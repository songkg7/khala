import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala.codex_hooks import (
    KHALA_HOOK_STATUS_MESSAGE,
    codex_home,
    install_codex_stop_hook,
)


class CodexHookInstallerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.home = self.root / "codex"
        self.executable = self.root / "bin with spaces" / "khala"
        self.executable.parent.mkdir()
        self.executable.write_text("#!/bin/sh\nexit 0\n")
        self.executable.chmod(0o700)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _handlers(self) -> list[dict]:
        document = json.loads((self.home / "hooks.json").read_text())
        return [
            handler
            for group in document["hooks"]["Stop"]
            for handler in group["hooks"]
        ]

    def test_install_creates_shell_safe_stop_hook(self) -> None:
        result = install_codex_stop_hook(self.executable, home=self.home)

        self.assertTrue(result.changed)
        handlers = self._handlers()
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0]["statusMessage"], KHALA_HOOK_STATUS_MESSAGE)
        self.assertIn("'", handlers[0]["command"])
        self.assertTrue(handlers[0]["command"].endswith(" hook codex"))
        self.assertIn("hook codex", handlers[0]["commandWindows"])
        self.assertEqual(handlers[0]["timeout"], 30)
        self.assertEqual(
            result.public_report(),
            {
                "integration": "codex",
                "status": "installed",
                "hooks_file": str(self.home / "hooks.json"),
                "restart_required": True,
                "trust_required": True,
            },
        )

    def test_install_preserves_existing_hooks_and_is_idempotent(self) -> None:
        self.home.mkdir()
        existing = {
            "description": "keep me",
            "hooks": {
                "SessionStart": [{"hooks": [{"type": "command", "command": "one"}]}],
                "Stop": [{"hooks": [{"type": "command", "command": "orca"}]}],
            },
        }
        path = self.home / "hooks.json"
        path.write_text(json.dumps(existing))

        first = install_codex_stop_hook(self.executable, home=self.home)
        first_bytes = path.read_bytes()
        first_mtime = path.stat().st_mtime_ns
        second = install_codex_stop_hook(self.executable, home=self.home)

        document = json.loads(path.read_text())
        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(second.public_report()["status"], "already_installed")
        self.assertEqual(path.read_bytes(), first_bytes)
        self.assertEqual(path.stat().st_mtime_ns, first_mtime)
        self.assertEqual(document["description"], "keep me")
        self.assertEqual(document["hooks"]["SessionStart"], existing["hooks"]["SessionStart"])
        self.assertIn("orca", [handler["command"] for handler in self._handlers()])

    def test_reinstall_replaces_relocated_khala_handler_and_removes_duplicates(self) -> None:
        install_codex_stop_hook(self.executable, home=self.home)
        path = self.home / "hooks.json"
        document = json.loads(path.read_text())
        duplicate = dict(document["hooks"]["Stop"][0]["hooks"][0])
        document["hooks"]["Stop"].append({"hooks": [duplicate]})
        path.write_text(json.dumps(document))
        relocated = self.root / "new-bin" / "khala"
        relocated.parent.mkdir()
        relocated.write_text("#!/bin/sh\nexit 0\n")
        relocated.chmod(0o700)

        result = install_codex_stop_hook(relocated, home=self.home)

        khala_handlers = [
            handler
            for handler in self._handlers()
            if handler.get("statusMessage") == KHALA_HOOK_STATUS_MESSAGE
        ]
        self.assertTrue(result.changed)
        self.assertEqual(len(khala_handlers), 1)
        self.assertIn(str(relocated), khala_handlers[0]["command"])

    def test_malformed_hook_shapes_fail_without_overwriting(self) -> None:
        invalid_documents = (
            [],
            {"hooks": []},
            {"hooks": {"Stop": {}}},
            {"hooks": {"Stop": ["bad"]}},
            {"hooks": {"Stop": [{}]}},
        )
        self.home.mkdir()
        path = self.home / "hooks.json"
        for document in invalid_documents:
            with self.subTest(document=document):
                original = json.dumps(document).encode()
                path.write_bytes(original)
                with self.assertRaises((TypeError, ValueError)):
                    install_codex_stop_hook(self.executable, home=self.home)
                self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(os.name == "posix", "symbolic-link behavior is POSIX-specific")
    def test_install_rejects_hooks_symlink_without_mutating_target(self) -> None:
        self.home.mkdir()
        target = self.root / "target.json"
        original = b'{"hooks": {}}\n'
        target.write_bytes(original)
        (self.home / "hooks.json").symlink_to(target)

        with self.assertRaisesRegex(OSError, "symbolic link"):
            install_codex_stop_hook(self.executable, home=self.home)

        self.assertEqual(target.read_bytes(), original)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_installed_hooks_file_is_private(self) -> None:
        install_codex_stop_hook(self.executable, home=self.home)
        self.assertEqual(stat.S_IMODE((self.home / "hooks.json").stat().st_mode), 0o600)

    def test_codex_home_honors_environment_override(self) -> None:
        configured = self.root / "custom"
        self.assertEqual(codex_home({"CODEX_HOME": str(configured)}), configured)

    def test_missing_or_non_executable_binary_fails_before_writing(self) -> None:
        missing = self.root / "missing"
        with self.assertRaisesRegex(OSError, "does not exist"):
            install_codex_stop_hook(missing, home=self.home)
        self.assertFalse((self.home / "hooks.json").exists())

        if os.name == "posix":
            self.executable.chmod(0o600)
            with self.assertRaisesRegex(OSError, "not executable"):
                install_codex_stop_hook(self.executable, home=self.home)
            self.assertFalse((self.home / "hooks.json").exists())


if __name__ == "__main__":
    unittest.main()
