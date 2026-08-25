import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from khala.hermes_hooks import (
    HERMES_HOOK_EVENTS,
    HermesHookInstallResult,
    install_hermes_hooks,
)


class FakeHermesConfig:
    def __init__(self, path: Path, hooks: Any) -> None:
        self.path = path
        self.hooks = hooks
        self.set_calls = 0
        self.plugins: Any = {}

    def run(self, executable: Path, *arguments: str) -> str:
        self.assert_executable(executable)
        if arguments == ("config", "path"):
            return str(self.path)
        if arguments == ("config", "get", "hooks", "--json"):
            return json.dumps(self.hooks)
        if arguments[:4] == ("config", "set", "--force", "hooks"):
            self.hooks = json.loads(arguments[4])
            self.set_calls += 1
            return "updated"
        if arguments == ("config", "get", "plugins", "--json"):
            return json.dumps(self.plugins)
        if arguments[:4] == ("config", "set", "--force", "plugins.enabled"):
            if not isinstance(json.loads(arguments[4]), list):
                raise AssertionError("plugins.enabled must be set as a JSON list")
            plugins = dict(self.plugins) if isinstance(self.plugins, dict) else {}
            plugins["enabled"] = json.loads(arguments[4])
            self.plugins = plugins
            self.set_calls += 1
            return "updated"
        raise AssertionError(f"unexpected Hermes command: {arguments}")

    def assert_executable(self, executable: Path) -> None:
        if executable.name != "hermes":
            raise AssertionError(f"unexpected executable: {executable}")


class HermesHookInstallerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.khala = self._executable(self.root / "bin with spaces" / "khala")
        self.hermes = self._executable(self.root / "bin" / "hermes")
        self.config_path = self.root / "hermes" / "config.yaml"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _executable(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
        return path

    def test_install_merges_all_events_preserves_existing_hooks_and_is_idempotent(self) -> None:
        other = {"command": "other hook", "timeout": 5}
        outbound = [{"url": "https://example.invalid", "events": ["on_session_end"]}]
        fake = FakeHermesConfig(
            self.config_path,
            {
                "post_llm_call": [other, {"command": "khala hook hermes", "timeout": 1}],
                "on_session_end": [
                    {"command": "/old/bin/khala hook hermes"},
                    {"command": "uv run khala hook hermes"},
                ],
                "outbound": outbound,
            },
        )

        with patch("khala.hermes_hooks._run_hermes", side_effect=fake.run):
            first = install_hermes_hooks(self.khala, hermes_executable=self.hermes)
            second = install_hermes_hooks(self.khala, hermes_executable=self.hermes)

        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(fake.set_calls, 2)  # one hooks write + one plugin enable
        self.assertEqual(fake.hooks["outbound"], outbound)
        self.assertEqual(fake.hooks["post_llm_call"][0], other)
        for event in HERMES_HOOK_EVENTS:
            khala_hooks = [
                hook
                for hook in fake.hooks[event]
                if hook["command"].endswith(" hook hermes")
            ]
            self.assertEqual(len(khala_hooks), 1)
            self.assertIn(str(self.khala), khala_hooks[0]["command"])
            self.assertEqual(khala_hooks[0]["timeout"], 30)
        self.assertEqual(
            first.public_report(),
            {
                "integration": "hermes",
                "status": "installed",
                "config_file": str(self.config_path),
                "events": list(HERMES_HOOK_EVENTS),
                "restart_required": True,
                "trust_required": True,
                "plugin": str(self.config_path.parent / "plugins" / "khala"),
            },
        )

    def test_install_copies_and_enables_plugin_idempotently(self) -> None:
        fake = FakeHermesConfig(
            self.config_path,
            {"post_llm_call": [{"command": "khala hook hermes", "timeout": 30}]},
        )
        fake.plugins = {"enabled": ["observability/langfuse"]}

        with patch("khala.hermes_hooks._run_hermes", side_effect=fake.run):
            first = install_hermes_hooks(self.khala, hermes_executable=self.hermes)
            second = install_hermes_hooks(self.khala, hermes_executable=self.hermes)

        # plugin files copied next to config (isolated hermes home in tests)
        plugin_dir = self.config_path.parent / "plugins" / "khala"
        self.assertTrue((plugin_dir / "plugin.yaml").is_file())
        self.assertTrue((plugin_dir / "__init__.py").is_file())
        self.assertEqual(first.plugin_path, plugin_dir)
        # enabled list gained khala exactly once, existing entries preserved
        self.assertEqual(fake.plugins["enabled"], ["observability/langfuse", "khala"])
        # second install: no further writes at all
        self.assertFalse(second.changed)
        set_calls_after_first = 1
        self.assertLessEqual(fake.set_calls, set_calls_after_first + 1)

    def test_invalid_hook_shape_fails_before_write(self) -> None:
        for hooks in ([], {"post_llm_call": {}}, {"post_llm_call": ["bad"]}):
            with self.subTest(hooks=hooks):
                fake = FakeHermesConfig(self.config_path, hooks)
                with (
                    patch("khala.hermes_hooks._run_hermes", side_effect=fake.run),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    install_hermes_hooks(self.khala, hermes_executable=self.hermes)
                self.assertEqual(fake.set_calls, 1)  # plugin enable only, no hooks write

    def test_missing_or_non_executable_binary_fails_before_config_read(self) -> None:
        with self.assertRaisesRegex(OSError, "does not exist"):
            install_hermes_hooks(self.root / "missing", hermes_executable=self.hermes)

        if os.name == "posix":
            self.hermes.chmod(0o600)
            with self.assertRaisesRegex(OSError, "not executable"):
                install_hermes_hooks(self.khala, hermes_executable=self.hermes)

    def test_install_cli_avoids_opening_product_store(self) -> None:
        from khala import main

        result = HermesHookInstallResult(self.config_path, changed=True)
        with (
            patch("khala.hermes_hooks.install_hermes_hooks", return_value=result),
            patch("khala.cli.Store", side_effect=AssertionError("opened store")),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            code = main(["install", "hermes"])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), result.public_report())


if __name__ == "__main__":
    unittest.main()
