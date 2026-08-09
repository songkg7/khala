from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

EXTRACTOR_VERSION = 1
MAX_IDENTIFIERS_PER_EVENT = 64
MAX_SCANNED_CONTENT_BYTES = 1_048_576

_MAX_IDENTIFIER_UTF8_BYTES = 512
_MAX_TOKEN_CHARACTERS = 1_024
_LINE_COLUMN_SUFFIX = re.compile(r"(?:(?::[0-9]+){1,2}|#L[0-9]+(?:C[0-9]+)?)$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_COMMON_FILENAMES = frozenset(
    {
        "Brewfile",
        "CHANGELOG",
        "CODEOWNERS",
        "Dockerfile",
        "Gemfile",
        "Jenkinsfile",
        "Justfile",
        "LICENSE",
        "LICENCE",
        "Makefile",
        "NOTICE",
        "Procfile",
        "README",
        "Rakefile",
        "Taskfile",
        "Vagrantfile",
    }
)
_TOKEN_PUNCTUATION = frozenset("._-/\\:+@~#%")
_WRAPPING_DELIMITERS = frozenset("\"'`<>[](){}=,;|")
_TRAILING_PROSE_PUNCTUATION = ".,;!?:"
_FORBIDDEN_IDENTIFIER_CHARACTERS = frozenset('<>"|?*#')


@dataclass(frozen=True)
class FileIdentifier:
    value: str
    basename: str
    mention_count: int

    def __post_init__(self) -> None:
        normalized = normalize_file_identifier(self.value)
        if normalized != self.value:
            raise ValueError("file identifier value must already be normalized")
        if self.basename != normalized.rsplit("/", 1)[-1]:
            raise ValueError("file identifier basename does not match its value")
        if type(self.mention_count) is not int or self.mention_count < 1:
            raise ValueError("file identifier mention count must be positive")


@dataclass(frozen=True)
class IdentifierExtraction:
    identifiers: tuple[FileIdentifier, ...]
    content_sha256: str
    truncated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.identifiers, tuple) or not all(
            isinstance(identifier, FileIdentifier) for identifier in self.identifiers
        ):
            raise TypeError("identifiers must be a tuple of FileIdentifier values")
        values = tuple(identifier.value for identifier in self.identifiers)
        if values != tuple(sorted(set(values))):
            raise ValueError("file identifiers must be unique and sorted")
        if len(values) > MAX_IDENTIFIERS_PER_EVENT:
            raise ValueError("too many file identifiers")
        if (
            not isinstance(self.content_sha256, str)
            or _SHA256.fullmatch(self.content_sha256) is None
        ):
            raise ValueError("content sha256 must be a lowercase hexadecimal digest")
        if type(self.truncated) is not bool:
            raise TypeError("identifier extraction truncated flag must be a boolean")


def normalize_file_identifier(value: object) -> str:
    """Normalize one portable, repository-relative file identifier.

    This is intentionally stricter than an operating system path parser. The
    identifier is safe to compare across clients and cannot escape a repository
    root or disguise a URL as a local file.
    """
    if not isinstance(value, str):
        raise TypeError("file identifier must be a string")
    if _contains_control(value):
        raise ValueError("file identifier must not contain control characters")

    normalized = unicodedata.normalize("NFC", value).replace("\\", "/")
    normalized = normalized.strip(" ")
    if _WINDOWS_DRIVE.match(normalized) is not None:
        raise ValueError("file identifier must not use a drive path")
    normalized = _LINE_COLUMN_SUFFIX.sub("", normalized)
    while normalized.startswith("./"):
        normalized = normalized[2:]

    if not normalized:
        raise ValueError("file identifier must not be empty")
    if normalized.startswith("/"):
        raise ValueError("file identifier must be relative")
    if normalized.startswith("~"):
        raise ValueError("file identifier must not use a home path")
    if _URI_SCHEME.match(normalized) is not None:
        raise ValueError("file identifier must not be a URI")
    if any(character in normalized for character in _FORBIDDEN_IDENTIFIER_CHARACTERS):
        raise ValueError("file identifier contains a forbidden character")
    if ":" in normalized:
        raise ValueError("file identifier contains an invalid location suffix")

    segments = normalized.split("/")
    if any(not segment or not segment.strip(" ") for segment in segments):
        raise ValueError("file identifier must not contain empty path segments")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("file identifier must not contain traversal segments")

    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("file identifier must be valid UTF-8") from error
    if len(encoded) > _MAX_IDENTIFIER_UTF8_BYTES:
        raise ValueError("file identifier exceeds 512 UTF-8 bytes")
    return normalized


def extract_file_identifiers(content: object) -> IdentifierExtraction:
    """Extract a deterministic, privacy-safe set of file mentions.

    The SHA-256 covers the complete content. Token scanning is limited to the
    first 1 MiB of UTF-8 data and each token is bounded, keeping extraction
    memory and matching work predictable even for adversarial tool output.
    """
    if not isinstance(content, str):
        raise TypeError("identifier extraction content must be a string")
    try:
        encoded_content = content.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("identifier extraction content must be valid UTF-8") from error

    content_sha256 = hashlib.sha256(encoded_content).hexdigest()
    scan_was_truncated = len(encoded_content) > MAX_SCANNED_CONTENT_BYTES
    scanned_content = encoded_content[:MAX_SCANNED_CONTENT_BYTES].decode("utf-8", errors="ignore")
    candidates, token_was_truncated = _candidate_tokens(scanned_content)

    mention_counts: dict[str, int] = {}
    identifier_cap_was_reached = False
    for candidate in candidates:
        cleaned = _clean_extracted_candidate(candidate)
        if not cleaned:
            continue
        try:
            normalized = normalize_file_identifier(cleaned)
        except (TypeError, ValueError):
            continue
        if not _looks_like_file_token(normalized):
            continue

        if normalized in mention_counts:
            mention_counts[normalized] += 1
            continue
        if len(mention_counts) < MAX_IDENTIFIERS_PER_EVENT:
            mention_counts[normalized] = 1
            continue

        identifier_cap_was_reached = True
        largest_retained = max(mention_counts)
        if normalized >= largest_retained:
            continue
        del mention_counts[largest_retained]
        mention_counts[normalized] = 1

    identifiers = tuple(
        FileIdentifier(
            value=value,
            basename=value.rsplit("/", 1)[-1],
            mention_count=mention_counts[value],
        )
        for value in sorted(mention_counts)
    )
    return IdentifierExtraction(
        identifiers=identifiers,
        content_sha256=content_sha256,
        truncated=(scan_was_truncated or token_was_truncated or identifier_cap_was_reached),
    )


def _candidate_tokens(content: str) -> tuple[tuple[str, ...], bool]:
    candidates: list[str] = []
    characters: list[str] = []
    token_overflow = False
    any_token_overflow = False

    def flush() -> None:
        nonlocal token_overflow, any_token_overflow
        if characters and not token_overflow:
            candidates.append("".join(characters))
        if token_overflow:
            any_token_overflow = True
        characters.clear()
        token_overflow = False

    for character in content:
        if _is_token_delimiter(character):
            flush()
            continue
        if token_overflow:
            continue
        if len(characters) >= _MAX_TOKEN_CHARACTERS:
            characters.clear()
            token_overflow = True
            continue
        characters.append(character)
    flush()
    return tuple(candidates), any_token_overflow


def _is_token_delimiter(character: str) -> bool:
    if character.isspace() or character in _WRAPPING_DELIMITERS:
        return True
    category = unicodedata.category(character)
    if category.startswith("C"):
        return True
    if category.startswith("P") and character not in _TOKEN_PUNCTUATION:
        return True
    return category.startswith("S") and character not in _TOKEN_PUNCTUATION


def _clean_extracted_candidate(candidate: str) -> str:
    return candidate.rstrip(_TRAILING_PROSE_PUNCTUATION)


def _looks_like_file_token(value: str) -> bool:
    basename = value.rsplit("/", 1)[-1]
    if "/" in value:
        return True
    if basename in _COMMON_FILENAMES:
        return True
    if basename.startswith(".") and len(basename) > 1:
        return True
    dot = basename.rfind(".")
    return 0 < dot < len(basename) - 1


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)
