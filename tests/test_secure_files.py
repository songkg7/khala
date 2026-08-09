import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala.secure_files import atomic_write_private_text


class SecureFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "private" / "config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_atomic_write_replaces_complete_content(self):
        atomic_write_private_text(self.path, "first\n")
        atomic_write_private_text(self.path, "second\n")

        self.assertEqual(self.path.read_text(), "second\n")

    def test_failed_replace_preserves_previous_content_and_cleans_temp(self):
        atomic_write_private_text(self.path, "stable\n")
        if os.name == "posix":
            self.path.chmod(0o644)

        with patch("khala.secure_files.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                atomic_write_private_text(self.path, "partial\n")

        self.assertEqual(self.path.read_text(), "stable\n")
        self.assertEqual(list(self.path.parent.glob(".config.json.*.tmp")), [])
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_atomic_write_uses_private_modes(self):
        atomic_write_private_text(self.path, "private\n")

        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
