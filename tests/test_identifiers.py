from __future__ import annotations

import hashlib
import unittest

from khala.identifiers import (
    EXTRACTOR_VERSION,
    MAX_IDENTIFIERS_PER_EVENT,
    MAX_SCANNED_CONTENT_BYTES,
    FileIdentifier,
    IdentifierExtraction,
    extract_file_identifiers,
    normalize_file_identifier,
)


class NormalizeFileIdentifierTest(unittest.TestCase):
    def test_normalizes_portable_paths_locations_and_unicode(self) -> None:
        cases = (
            ("./khala\\remote.py:12:7", "khala/remote.py"),
            ("././Dockerfile#L8C2", "Dockerfile"),
            ("src/e\u0301vent.py", "src/évent.py"),
            (" docs/My File.md ", "docs/My File.md"),
            (".gitignore", ".gitignore"),
            ("src/runner", "src/runner"),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_file_identifier(value), expected)

    def test_accepts_exact_512_utf8_byte_boundary(self) -> None:
        value = "a" * 509 + ".py"

        self.assertEqual(len(value.encode("utf-8")), 512)
        self.assertEqual(normalize_file_identifier(value), value)

    def test_non_string_values_are_type_errors(self) -> None:
        for value in (None, b"file.py", 42, ["file.py"]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_file_identifier(value)

    def test_rejects_non_relative_or_external_identifiers(self) -> None:
        invalid = (
            "/etc/passwd",
            "~/secret.py",
            "~other/secret.py",
            "C:\\work\\remote.py",
            "C:relative.py",
            "\\\\server\\share\\remote.py",
            "https://example.com/remote.py",
            "file:remote.py",
            "ssh://host/repo/file.py:12",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_file_identifier(value)

    def test_rejects_empty_traversal_and_ambiguous_segments(self) -> None:
        invalid = (
            "",
            "   ",
            ".",
            "..",
            "./",
            "src/",
            "src//remote.py",
            "src/./remote.py",
            "src/../remote.py",
            "src/ /remote.py",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_file_identifier(value)

    def test_rejects_controls_forbidden_characters_and_invalid_utf8(self) -> None:
        invalid = (
            "src/a\x00.py",
            "src/a\n.py",
            "src/a\t.py",
            "src/a\u200b.py",
            "src/a?.py",
            "src/a#fragment.py",
            "src/a|b.py",
            "src/a:b.py",
            "src/a\ud800.py",
        )
        for value in invalid:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                normalize_file_identifier(value)

    def test_rejects_identifier_larger_than_512_utf8_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "512 UTF-8 bytes"):
            normalize_file_identifier("a" * 510 + ".py")


class IdentifierValueObjectTest(unittest.TestCase):
    def test_file_identifier_is_frozen_and_self_consistent(self) -> None:
        identifier = FileIdentifier("src/app.py", "app.py", 2)

        self.assertEqual(identifier.value, "src/app.py")
        with self.assertRaises(AttributeError):
            identifier.value = "other.py"  # type: ignore[misc]

    def test_file_identifier_rejects_invalid_fields(self) -> None:
        invalid = (
            ("./src/app.py", "app.py", 1),
            ("src/app.py", "wrong.py", 1),
            ("src/app.py", "app.py", 0),
            ("src/app.py", "app.py", True),
        )
        for value, basename, count in invalid:
            with self.subTest(value=value, count=count), self.assertRaises(ValueError):
                FileIdentifier(value, basename, count)

    def test_extraction_value_rejects_malformed_contracts(self) -> None:
        first = FileIdentifier("a.py", "a.py", 1)
        second = FileIdentifier("b.py", "b.py", 1)
        valid_hash = "0" * 64
        with self.assertRaises(TypeError):
            IdentifierExtraction([first], valid_hash, False)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            IdentifierExtraction(("a.py",), valid_hash, False)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            IdentifierExtraction((second, first), valid_hash, False)
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            IdentifierExtraction((first, first), valid_hash, False)
        with self.assertRaisesRegex(ValueError, "sha256"):
            IdentifierExtraction((first,), "A" * 64, False)
        with self.assertRaises(TypeError):
            IdentifierExtraction((first,), valid_hash, 0)  # type: ignore[arg-type]

        too_many = tuple(
            FileIdentifier(f"{index:03}.py", f"{index:03}.py", 1)
            for index in range(MAX_IDENTIFIERS_PER_EVENT + 1)
        )
        with self.assertRaisesRegex(ValueError, "too many"):
            IdentifierExtraction(too_many, valid_hash, True)


class ExtractFileIdentifiersTest(unittest.TestCase):
    def test_extracts_paths_common_names_and_merged_mentions(self) -> None:
        content = (
            "Changed ./khala\\remote.py:12:3, khala/remote.py#L20 and "
            "`tests/test_remote.py`. Dockerfile. src/runner. "
            "Also src/e\u0301vent.py and src/évent.py."
        )

        result = extract_file_identifiers(content)

        self.assertEqual(EXTRACTOR_VERSION, 3)
        self.assertEqual(
            result.identifiers,
            (
                FileIdentifier("Dockerfile", "Dockerfile", 1),
                FileIdentifier("khala/remote.py", "remote.py", 2),
                FileIdentifier("src/runner", "runner", 1),
                FileIdentifier("src/évent.py", "évent.py", 2),
                FileIdentifier("tests/test_remote.py", "test_remote.py", 1),
            ),
        )
        self.assertEqual(
            result.content_sha256,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        self.assertFalse(result.truncated)

    def test_ignores_prose_and_unsafe_path_shaped_tokens(self) -> None:
        content = (
            "ordinary prose only; https://example.com/file.py "
            "/tmp/private.py ~/secret.py ../escape.py "
            "src/../escape.py C:\\private\\drive.py \\\\server\\share.py"
        )

        result = extract_file_identifiers(content)

        self.assertEqual(result.identifiers, ())
        self.assertFalse(result.truncated)

    def test_strips_trailing_literal_escapes_before_path_normalization(self) -> None:
        # Log text embeds literal "\n"; it must not become a fake "/n" path.
        result = extract_file_identifiers(
            "retries=120\\n rows=-20\\n CRLF.\\r\\n module.ptr ok src/x.py"
        )

        self.assertEqual(
            result.identifiers,
            (
                FileIdentifier("module.ptr", "module.ptr", 1),
                FileIdentifier("src/x.py", "x.py", 1),
            ),
        )
        self.assertFalse(result.truncated)

    def test_understands_wrappers_prose_punctuation_and_dotfiles(self) -> None:
        result = extract_file_identifiers(
            "[remote.py](remote.py), AGENTS.md; .gitignore! Makefile?"
        )

        self.assertEqual(
            result.identifiers,
            (
                FileIdentifier(".gitignore", ".gitignore", 1),
                FileIdentifier("AGENTS.md", "AGENTS.md", 1),
                FileIdentifier("Makefile", "Makefile", 1),
                FileIdentifier("remote.py", "remote.py", 2),
            ),
        )

    def test_cap_keeps_lexically_first_identifiers_and_marks_truncated(self) -> None:
        mentions = [f"pkg/file_{index:03}.py" for index in reversed(range(70))]
        mentions.extend(("pkg/file_000.py:9", "zzz.py"))

        result = extract_file_identifiers(" ".join(mentions))

        self.assertTrue(result.truncated)
        self.assertEqual(len(result.identifiers), MAX_IDENTIFIERS_PER_EVENT)
        self.assertEqual(result.identifiers[0].value, "pkg/file_000.py")
        self.assertEqual(result.identifiers[-1].value, "pkg/file_063.py")
        self.assertEqual(result.identifiers[0].mention_count, 2)

    def test_scan_bound_marks_truncated_but_hashes_full_content(self) -> None:
        content = "early.py " + "x" * MAX_SCANNED_CONTENT_BYTES + " late.py"

        result = extract_file_identifiers(content)

        self.assertTrue(result.truncated)
        self.assertEqual(
            result.identifiers,
            (FileIdentifier("early.py", "early.py", 1),),
        )
        self.assertEqual(
            result.content_sha256,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def test_overlong_token_is_skipped_and_reported_as_truncated(self) -> None:
        content = "early.py \x00" + "x" * 1_025 + " late.py"

        result = extract_file_identifiers(content)

        self.assertEqual(
            tuple(identifier.value for identifier in result.identifiers),
            ("early.py", "late.py"),
        )
        self.assertTrue(result.truncated)

    def test_empty_content_has_stable_full_hash(self) -> None:
        result = extract_file_identifiers("")

        self.assertEqual(result.identifiers, ())
        self.assertEqual(result.content_sha256, hashlib.sha256(b"").hexdigest())
        self.assertFalse(result.truncated)

    def test_non_string_and_surrogate_content_fail_fast(self) -> None:
        for value in (None, b"remote.py", 1):
            with self.subTest(value=value), self.assertRaises(TypeError):
                extract_file_identifiers(value)
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            extract_file_identifiers("remote.py\ud800")


if __name__ == "__main__":
    unittest.main()
