import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import psycopg
from khala.codex_hooks import (
    KHALA_HOOK_STATUS_MESSAGE,
    KHALA_RECALL_STATUS_MESSAGE,
    KHALA_REMEMBER_STATUS_MESSAGE,
    codex_home,
    install_codex_hooks,
    install_codex_stop_hook,
    resolve_khala_executable,
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

    def _handlers(self, event: str = "Stop") -> list[dict]:
        document = json.loads((self.home / "hooks.json").read_text())
        return [
            handler
            for group in document["hooks"][event]
            for handler in group["hooks"]
        ]

    def test_install_creates_shell_safe_recall_and_stop_hooks(self) -> None:
        result = install_codex_hooks(self.executable, home=self.home)

        self.assertTrue(result.changed)
        recall_handlers = self._handlers("UserPromptSubmit")
        stop_handlers = self._handlers("Stop")
        self.assertEqual(len(recall_handlers), 1)
        self.assertEqual(len(stop_handlers), 1)

        recall = recall_handlers[0]
        self.assertEqual(recall["statusMessage"], KHALA_RECALL_STATUS_MESSAGE)
        self.assertIn("'", recall["command"])
        self.assertTrue(recall["command"].endswith(" hook codex --codex-recall"))
        self.assertIn("hook codex --codex-recall", recall["commandWindows"])
        self.assertEqual(recall["timeout"], 30)
        self.assertEqual(recall["additionalContextLimit"], 4000)

        stop = stop_handlers[0]
        self.assertEqual(stop["statusMessage"], KHALA_REMEMBER_STATUS_MESSAGE)
        self.assertIn("'", stop["command"])
        self.assertTrue(
            stop["command"].endswith(" hook codex --codex-stop")
        )
        self.assertIn("hook codex --codex-stop", stop["commandWindows"])
        self.assertEqual(stop["timeout"], 30)
        self.assertEqual(
            result.public_report(),
            {
                "integration": "codex",
                "status": "installed",
                "hooks_file": str(self.home / "hooks.json"),
                "events": ["UserPromptSubmit", "Stop"],
                "recall_enabled": True,
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
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "prompt-audit"}]}
                ],
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
                "events": ["UserPromptSubmit", "Stop"],
                "recall_enabled": True,
                "restart_required": True,
                "trust_required": True,
            },
        )
        self.assertEqual(path.read_bytes(), first_bytes)
        self.assertEqual(path.stat().st_mtime_ns, first_mtime)
        self.assertEqual(document["description"], "keep me")
        self.assertEqual(document["hooks"]["SessionStart"], existing["hooks"]["SessionStart"])
        self.assertEqual(
            document["hooks"]["UserPromptSubmit"][0],
            existing["hooks"]["UserPromptSubmit"][0],
        )
        self.assertEqual(document["hooks"]["Stop"][0], existing["hooks"]["Stop"][0])
        self.assertIn("orca", [handler["command"] for handler in self._handlers()])
        self.assertIn(
            "prompt-audit",
            [handler["command"] for handler in self._handlers("UserPromptSubmit")],
        )

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

    def test_reinstall_recognizes_owned_commands_when_status_text_changed(self) -> None:
        install_codex_hooks(self.executable, home=self.home)
        path = self.home / "hooks.json"
        document = json.loads(path.read_text())
        document["hooks"]["Stop"][-1]["hooks"][0]["statusMessage"] = "custom text"
        document["hooks"]["UserPromptSubmit"][-1]["hooks"][0]["statusMessage"] = (
            "custom recall text"
        )
        path.write_text(json.dumps(document))

        install_codex_hooks(self.executable, home=self.home)

        self.assertEqual(
            [
                handler["statusMessage"]
                for handler in self._handlers("UserPromptSubmit")
                if "--codex-recall" in handler["command"]
            ],
            [KHALA_RECALL_STATUS_MESSAGE],
        )
        self.assertEqual(
            [
                handler["statusMessage"]
                for handler in self._handlers("Stop")
                if "--codex-stop" in handler["command"]
            ],
            [KHALA_REMEMBER_STATUS_MESSAGE],
        )

    def test_unrelated_handler_with_same_status_message_is_preserved(self) -> None:
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
                                        "command": "/usr/bin/unrelated-sync",
                                        "statusMessage": KHALA_REMEMBER_STATUS_MESSAGE,
                                    }
                                ]
                            }
                        ]
                    }
                }
            )
        )

        install_codex_hooks(self.executable, home=self.home)

        commands = [handler["command"] for handler in self._handlers("Stop")]
        self.assertIn("/usr/bin/unrelated-sync", commands)
        self.assertEqual(sum("--codex-stop" in command for command in commands), 1)

    def test_malformed_hook_shapes_fail_without_overwriting(self) -> None:
        invalid_documents = (
            [],
            {"hooks": []},
            {"hooks": {"Stop": {}}},
            {"hooks": {"UserPromptSubmit": {}}},
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

    def test_executable_resolution_prefers_the_running_khala_entrypoint(self) -> None:
        other = self.root / "other-bin" / "khala"
        other.parent.mkdir()
        other.write_text("#!/bin/sh\nexit 0\n")
        other.chmod(0o700)

        with (
            patch.object(sys, "argv", [str(self.executable), "install", "codex"]),
            patch("khala.codex_hooks.shutil.which", return_value=str(other)),
        ):
            resolved = resolve_khala_executable()

        self.assertEqual(resolved, self.executable.absolute())

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

    def test_install_cli_registers_workspace_and_avoids_opening_product_store(self) -> None:
        from khala import config, main

        workspace = self.root / "workspace"
        workspace.mkdir()
        config_directory = self.root / "config"

        with (
            patch.dict(os.environ, {"CODEX_HOME": str(self.home)}),
            patch.object(config, "CONFIG_DIR", config_directory),
            patch.object(config, "CONFIG_FILE", config_directory / "config.json"),
            patch(
                "khala.codex_hooks.resolve_khala_executable",
                return_value=self.executable,
            ),
            patch("khala.cli.Store", side_effect=AssertionError("opened store")),
            patch.object(sys, "stdout", new_callable=io.StringIO) as out,
        ):
            code = main(["install", "codex", "--workspace", str(workspace)])
            configured = config.load_config()

        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["status"], "installed")
        self.assertEqual(report["workspace"], str(workspace.resolve()))
        self.assertTrue(report["recall_enabled"])
        self.assertEqual(configured.get("workspaces"), [str(workspace.resolve())])
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
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        config.save_config(
            {"workspaces": [str(self.workspace.resolve())], "remote": None}
        )

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

    def _payload(self) -> str:
        return json.dumps(
            {
                "hook_event_name": "Stop",
                "session_id": "session-id",
                "transcript_path": "/tmp/rollout-session-id.jsonl",
                "cwd": str(self.workspace),
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

    def test_disabled_workspace_skips_stop_before_opening_store(self) -> None:
        from khala import config

        config.save_config({"workspaces": [], "remote": None})
        with patch(
            "khala.cli._open_store",
            side_effect=AssertionError("store opened before workspace gate"),
        ):
            code, stdout, stderr = self._run(
                "hook", "codex", "--codex-stop", stdin=self._payload()
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertEqual(stderr, "")

    def test_codex_recall_adds_bounded_untrusted_context(self) -> None:
        hit = {
            "id": "event-1",
            "session_id": "session-1",
            "role": "assistant",
            "content": "Use the blue deployment path",
            "created_at": "2026-08-22T00:00:00+00:00",
            "agent": "codex",
            "user_name": "Haril",
            "user_email": "private@example.com",
            "adjacent": [{"role": "user", "content": "Which deployment path?"}],
        }
        from unittest.mock import Mock

        store = Mock()
        with (
            patch("khala.cli._open_store", return_value=store),
            patch(
                "khala.codex_recall.recall_for_prompt", return_value=[hit]
            ) as recall,
        ):
            code, stdout, stderr = self._run(
                "hook",
                "codex",
                "--codex-recall",
                stdin=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "session_id": "session-id",
                        "transcript_path": "/tmp/rollout-session-id.jsonl",
                        "cwd": str(self.workspace),
                        "prompt": "deploy api_key=super-secret",
                    }
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        response = json.loads(stdout)
        hook_output = response["hookSpecificOutput"]
        self.assertEqual(hook_output["hookEventName"], "UserPromptSubmit")
        context = hook_output["additionalContext"]
        self.assertIn("untrusted historical memory", context)
        self.assertIn("Use the blue deployment path", context)
        self.assertNotIn("private@example.com", context)
        self.assertEqual(recall.call_args.args[1], "deploy api_key=[REDACTED]")
        self.assertEqual(recall.call_args.kwargs["limit"], 3)
        self.assertEqual(recall.call_args.kwargs["role"], "assistant")
        store.close.assert_called_once_with()

    def test_codex_recall_without_hits_is_neutral(self) -> None:
        from unittest.mock import Mock

        with (
            patch("khala.cli._open_store", return_value=Mock()),
            patch("khala.codex_recall.recall_for_prompt", return_value=[]),
        ):
            code, stdout, stderr = self._run(
                "hook",
                "codex",
                "--codex-recall",
                stdin=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "cwd": str(self.workspace),
                        "prompt": "no matching memory",
                    }
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertEqual(stderr, "")

    def test_codex_recall_remote_unavailable_uses_local_context(self) -> None:
        from khala import config
        from khala.store import Store

        marker = "localfallbackcanary"
        store = Store(self.database)
        try:
            session_id = store.add_session("codex")
            store.add_event(
                session_id,
                "assistant",
                f"The deployment marker is {marker}",
            )
        finally:
            store.close()
        config.save_config(
            {
                "workspaces": [str(self.workspace.resolve())],
                "remote": "host=127.0.0.1 port=1 dbname=khala",
            }
        )

        with patch(
            "khala.remote.RemoteStore.ensure_schema",
            side_effect=psycopg.OperationalError("connection refused"),
        ):
            code, stdout, stderr = self._run(
                "hook",
                "codex",
                "--codex-recall",
                stdin=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "cwd": str(self.workspace),
                        "prompt": marker,
                    }
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        response = json.loads(stdout)
        context = response["hookSpecificOutput"]["additionalContext"]
        self.assertIn(marker, context)

    def test_codex_recall_failure_is_non_blocking_and_redacted(self) -> None:
        from unittest.mock import Mock

        with (
            patch("khala.cli._open_store", return_value=Mock()),
            patch(
                "khala.codex_recall.recall_for_prompt",
                side_effect=RuntimeError(
                    "postgresql://alice:" + "secret" + "@db/khala"
                ),
            ),
        ):
            code, stdout, stderr = self._run(
                "hook",
                "codex",
                "--codex-recall",
                stdin=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "cwd": str(self.workspace),
                        "prompt": "find memory",
                    }
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertIn("Codex recall failed", stderr)
        self.assertNotIn("secret", stderr)
        self.assertIn("[REDACTED]", stderr)

    def test_disabled_workspace_skips_recall_before_opening_store(self) -> None:
        from khala import config

        config.save_config({"workspaces": [], "remote": None})
        with patch(
            "khala.cli._open_store",
            side_effect=AssertionError("store opened before workspace gate"),
        ):
            code, stdout, stderr = self._run(
                "hook",
                "codex",
                "--codex-recall",
                stdin=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "cwd": str(self.workspace),
                        "prompt": "find memory",
                    }
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertEqual(stderr, "")

    def test_irrelevant_or_incomplete_codex_events_skip_before_store(self) -> None:
        cases = (
            (
                "--codex-stop",
                {
                    "hook_event_name": "Stop",
                    "cwd": str(self.workspace),
                },
            ),
            (
                "--codex-recall",
                {
                    "hook_event_name": "UserPromptSubmit",
                    "cwd": str(self.workspace),
                    "prompt": "   ",
                },
            ),
            (
                "--codex-recall",
                {
                    "hook_event_name": "Stop",
                    "cwd": str(self.workspace),
                    "prompt": "wrong event",
                },
            ),
        )
        for flag, payload in cases:
            with self.subTest(flag=flag, payload=payload), patch(
                "khala.cli._open_store",
                side_effect=AssertionError("store opened for ineligible event"),
            ):
                code, stdout, stderr = self._run(
                    "hook", "codex", flag, stdin=json.dumps(payload)
                )

            self.assertEqual(code, 0)
            self.assertEqual(stdout, "{}\n")
            self.assertEqual(stderr, "")

    def test_unresolvable_hook_cwd_skips_before_store(self) -> None:
        with (
            patch(
                "khala.codex_hook_runtime.resolve_workspace_path",
                return_value=None,
            ),
            patch(
                "khala.cli._open_store",
                side_effect=AssertionError("store opened for invalid cwd"),
            ),
        ):
            code, stdout, stderr = self._run(
                "hook",
                "codex",
                "--codex-recall",
                stdin=json.dumps(
                    {
                        "hook_event_name": "UserPromptSubmit",
                        "cwd": str(self.workspace),
                        "prompt": "find memory",
                    }
                ),
            )

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "{}\n")
        self.assertEqual(stderr, "")

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
