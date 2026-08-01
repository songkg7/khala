from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

# ponytail: regex-based redaction. Upgrade to AST-parsed detection in high-security deployments.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret)\b\s*[:=]\s*([^\s]+)"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
)


def redact(text: str) -> str:
    """Mask known secret patterns in-place. Non-destructive for clean text."""
    text = SECRET_PATTERNS[0].sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    for pattern in SECRET_PATTERNS[1:]:
        text = pattern.sub("[REDACTED]", text)
    return text