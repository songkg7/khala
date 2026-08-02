import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala.identity import resolve_user


class KhalaConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from khala import config as cfg

        # CONFIG_DIR/CONFIG_FILE are resolved at import; patch both constants.
        self.patcher = patch.object(cfg, "CONFIG_DIR", Path(self.tmp.name))
        self.patcher.start()
        self.file_patcher = patch.object(cfg, "CONFIG_FILE", Path(self.tmp.name) / "config.json")
        self.file_patcher.start()
        self.cfg = cfg

    def tearDown(self):
        self.file_patcher.stop()
        self.patcher.stop()
        self.tmp.cleanup()

    def test_workspace_add_and_is_enabled(self):
        self.cfg.save_config({"workspaces": ["/ws/a"], "remote": None})
        self.assertEqual(self.cfg.load_config()["workspaces"], ["/ws/a"])
        self.assertTrue(self.cfg.is_enabled_workspace("/ws/a/sub/dir"))
        self.assertFalse(self.cfg.is_enabled_workspace("/ws/b"))

    def test_config_default_when_missing(self):
        self.assertEqual(self.cfg.load_config(), {"workspaces": [], "remote": None})
        self.assertFalse(self.cfg.is_enabled_workspace("/anywhere"))

    def test_remote_dsn_roundtrip(self):
        self.assertEqual(self.cfg.get_remote_dsn(), None)
        self.assertFalse(self.cfg.is_connected())
        self.cfg.save_config({"workspaces": [], "remote": "postgresql://u:***@h:5432/db"})
        self.assertEqual(self.cfg.get_remote_dsn(), "postgresql://u:***@h:5432/db")
        self.assertTrue(self.cfg.is_connected())


class KhalaConfigSetGetTest(unittest.TestCase):
    """dot-path config get/set helpers."""

    def setUp(self):
        import khala.cli as cli

        self.cli = cli

    def test_set_and_get_scalar(self):
        d = {"workspaces": []}
        self.cli._config_set(d, "embedding.model", "openai/text-embedding-3-small")
        self.assertEqual(
            self.cli._config_get(d, "embedding.model"), "openai/text-embedding-3-small"
        )

    def test_set_list_index_and_get(self):
        d = {"workspaces": []}
        self.cli._config_set(d, "workspaces[0]", "/a")
        self.cli._config_set(d, "workspaces[1]", "/b")
        self.assertEqual(d["workspaces"], ["/a", "/b"])
        self.assertEqual(self.cli._config_get(d, "workspaces[1]"), "/b")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.cli._config_get({"a": {"b": 1}}, "a.c"))

    def test_set_json_value_via_get(self):
        d = {"embedding": {}}
        self.cli._config_set(d, "embedding.provider", "openrouter")
        self.assertEqual(self.cli._config_get(d, "embedding.provider"), "openrouter")


class KhalaIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _init_git(self, name: str, email: str) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.dir, check=True)
        subprocess.run(["git", "config", "user.name", name], cwd=self.dir, check=True)
        subprocess.run(["git", "config", "user.email", email], cwd=self.dir, check=True)

    def test_resolve_from_git_config(self):
        self._init_git("Alice Kim", "alice@example.com")
        user = resolve_user(self.dir)
        self.assertEqual(user.name, "Alice Kim")
        self.assertEqual(user.email, "alice@example.com")

    def test_fallback_to_env_user(self):
        with patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}):
            user = resolve_user(self.dir)
        self.assertEqual(user.name, os.environ.get("USER"))
        self.assertIsNone(user.email)

    def test_git_global_config_is_used(self):
        user = resolve_user(self.dir)
        self.assertIsNotNone(user.name)


if __name__ == "__main__":
    unittest.main()
