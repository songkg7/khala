import io
import json
import os
import stat
import sys
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
        self.assertTrue(
            handlers[0]["command"].endswith(" hook codex --codex-stop")
        )
        self.assertIn("hook codex --codex-stop", handlers[0]["commandWindows"])
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
        self.assertEqual(
            second.public_report(),
            {
                "integration": "codex",
                "status": "already_installed",
                "hooks_file": str(path),
                "restart_required": True,
                "trust_required": True,
            },
        )
        self.assertEqual(path.read_bytes(), first_bytes)
        self.assertEqual(path.stat().st_mtime_ns, first_mtime)
        self.assertEqual(document["description"], "keep me")
        self.assertEqual(document["hooks"]["SessionStart"], existing["hooks"]["SessionStart"])
        self.assertEqual(document["hooks"]["Stop"][0], existing["hooks"]["Stop"][0])
        self.assertIn("orca", [handler["command"] for handler in self._handlers()])

    def test_reinstall_replaces_relocated_khala_handler_and_removes_duplicates(self) -> None:
        install_codex_stop_hook(self.executable, home=self.home)
        path = self.home / "hooks.json"
        document = json.loads(path.read_text())
        duplicate = dict(document["hooks"]["Stop"][0]["hooks"][0])
        document["hooks"]["Stop"].append({"hooks": [duplicate]})
        document["hooks"]["Stop"].append(
            {"hooks": [{"type": "command", "command": "after-khala"}]}
        )
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
        stop_groups = json.loads(path.read_text())["hooks"]["Stop"]
        self.assertEqual(stop_groups[1]["hooks"], [])
        self.assertEqual(stop_groups[2]["hooks"][0]["command"], "after-khala")

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

    def test_install_upgrades_owned_pre_sentinel_handler_in_place(self) -> None:
        self.home.mkdir()
        path = self.home / "hooks.json"
        path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "/old/khala hook codex",
                                        "statusMessage": KHALA_HOOK_STATUS_MESSAGE,
                                    },
                                    {"type": "command", "command": "keep-position"},
                                ]
                            }
                        ]
                    }
                }
            )
        )

        install_codex_stop_hook(self.executable, home=self.home)

        handlers = self._handlers()
        self.assertEqual(len(handlers), 2)
        self.assertTrue(handlers[0]["command"].endswith("--codex-stop"))
        self.assertEqual(handlers[1]["command"], "keep-position")

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

    @unittest.skipUnless(os.name == "posix", "symbolic-link behavior is POSIX-specific")
    def test_executable_symlink_path_is_not_resolved(self) -> None:
        target = self.root / "tool-env" / "khala"
        target.parent.mkdir()
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o700)
        shim = self.root / "bin" / "khala"
        shim.parent.mkdir()
        shim.symlink_to(target)

        install_codex_stop_hook(shim, home=self.home)

        command = self._handlers()[0]["command"]
        self.assertIn(str(shim), command)
        self.assertNotIn(str(target), command)

    def test_install_cli_avoids_opening_product_store(self) -> None:
        from khala import main

        with (
            patch.dict(os.environ, {"CODEX_HOME": str(self.home)}),
            patch(
                "khala.codex_hooks.resolve_khala_executable",
                return_value=self.executable,
            ),
            patch("khala.cli.Store", side_effect=AssertionError("opened store")),
            patch.object(sys, "stdout", new_callable=io.StringIO) as out,
        ):
            code = main(["install", "codex"])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "installed")
        self.assertTrue((self.home / "hooks.json").is_file())


class CodexStopCLIContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "khala.db"
        from khala import config

        self.config_directory_patcher = patch.object(
            config, "CONFIG_DIR", self.root / "config"
        )
        self.config_file_patcher = patch.object(
            config, "CONFIG_FILE", self.root / "config" / "config.json"
        )
        self.config_directory_patcher.start()
        self.config_file_patcher.start()

    def tearDown(self) -> None:
        self.config_file_patcher.stop()
        self.config_directory_patcher.stop()
        self.temporary_directory.cleanup()

    def _run(self, *arguments: str, stdin: str) -> tuple[int, str, str]:
        from khala import main

        with (
            patch.object(sys, "stdin", io.StringIO(stdin)),
            patch.object(sys, "stdout", io.StringIO()) as stdout,
            patch.object(sys, "stderr", io.StringIO()) as stderr,
        ):
            code = main(["--db", str(self.database), *arguments])
            return code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _payload() -> str:
        return json.dumps(
            {
                "hook_event_name": "Stop",
                "session_id": "session-id",
                "transcript_path": "/tmp/rollout-session-id.jsonl",
                "cwd": "/workspace",
            }
        )

    def test_codex_stop_success_emits_one_neutral_json_object(self) -> None:
        operational = {"status": "synced", "added": 2}
        with patch("khala.hooks.sync_hook", return_value=operational) as sync:
            code, stdout, stderr = self._run(
                "hook", "codex", "--codex-stop", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(sync.call_args.kwargs["adapter_name"], "codex")

    def test_codex_stop_malformed_input_is_non_blocking(self) -> None:
        code, stdout, stderr = self._run(
            "hook", "codex", "--codex-stop", stdin="{not-json"
        )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertIn("Codex Stop sync failed", stderr)

    def test_codex_stop_sync_exception_is_non_blocking(self) -> None:
        with patch(
            "khala.hooks.sync_hook", side_effect=RuntimeError("sync unavailable")
        ):
            code, stdout, stderr = self._run(
                "hook", "codex", "--codex-stop", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertIn("sync unavailable", stderr)

    def test_codex_stop_store_initialization_exception_is_non_blocking(self) -> None:
        with patch("khala.cli.Store", side_effect=RuntimeError("database unavailable")):
            code, stdout, stderr = self._run(
                "hook", "codex", "--codex-stop", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertIn("database unavailable", stderr)

    def test_codex_stop_store_close_exception_is_non_blocking(self) -> None:
        from unittest.mock import Mock

        store = Mock()
        store.close.side_effect = RuntimeError("close unavailable")
        with (
            patch("khala.cli._open_store", return_value=store),
            patch("khala.hooks.sync_hook", return_value={"status": "synced"}),
        ):
            code, stdout, stderr = self._run(
                "hook", "codex", "--codex-stop", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertIn("close unavailable", stderr)

    def test_codex_stop_error_diagnostic_is_redacted(self) -> None:
        with patch(
            "khala.hooks.sync_hook",
            side_effect=RuntimeError("postgresql://alice:secret@db/khala"),
        ):
            code, stdout, stderr = self._run(
                "hook", "codex", "--codex-stop", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertNotIn("secret", stderr)
        self.assertIn("[REDACTED]", stderr)

    def test_generic_codex_hook_keeps_operational_json(self) -> None:
        operational = {"status": "synced", "added": 2}
        with patch("khala.hooks.sync_hook", return_value=operational):
            code, stdout, stderr = self._run(
                "hook", "codex", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), operational)
        self.assertEqual(stderr, "")


if __name__ == "__main__":
    unittest.main()
