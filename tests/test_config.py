import os
import stat
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

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_load_tightens_existing_config_directory_when_file_is_missing(self):
        self.cfg.CONFIG_DIR.chmod(0o755)

        self.assertEqual(self.cfg.load_config(), {"workspaces": [], "remote": None})

        self.assertEqual(stat.S_IMODE(self.cfg.CONFIG_DIR.stat().st_mode), 0o700)

    def test_remote_dsn_roundtrip(self):
        self.assertEqual(self.cfg.get_remote_dsn(), None)
        self.assertFalse(self.cfg.is_connected())
        self.cfg.save_config({"workspaces": [], "remote": "postgresql://u:***@h:5432/db"})
        self.assertEqual(self.cfg.get_remote_dsn(), "postgresql://u:***@h:5432/db")
        self.assertTrue(self.cfg.is_connected())

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_save_and_load_tighten_config_permissions(self):
        self.cfg.CONFIG_DIR.chmod(0o755)
        self.cfg.CONFIG_FILE.write_text('{"workspaces": [], "remote": null}\n')
        self.cfg.CONFIG_FILE.chmod(0o644)

        self.assertEqual(self.cfg.load_config()["workspaces"], [])
        self.assertEqual(stat.S_IMODE(self.cfg.CONFIG_DIR.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.cfg.CONFIG_FILE.stat().st_mode), 0o600)

        self.cfg.save_config({"workspaces": ["/private"], "remote": None})
        self.assertEqual(stat.S_IMODE(self.cfg.CONFIG_FILE.stat().st_mode), 0o600)

    @unittest.skipUnless(os.name == "posix", "symbolic-link behavior is POSIX-specific")
    def test_load_rejects_config_symlink_without_mutating_target(self):
        target = Path(self.tmp.name).parent / f"{Path(self.tmp.name).name}-external.json"
        target.write_text('{"workspaces": ["/external"], "remote": null}\n')
        target.chmod(0o644)
        self.addCleanup(target.unlink)
        self.cfg.CONFIG_FILE.symlink_to(target)

        with self.assertRaisesRegex(OSError, "must not be a symbolic link"):
            self.cfg.load_config()

        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_mask_dsn_handles_uri_keyword_and_invalid_values(self):
        values = (
            "postgresql://alice:p%40ss@db.example:5432/khala?sslmode=require",
            "host=db.example dbname=khala user=alice password='secret with space'",
            "postgresql://alice@db.example/khala?password=query-secret",
            "host=db.example user=alice password=login-secret sslpassword=cert-secret",
        )
        for value in values:
            with self.subTest(value=value):
                masked = self.cfg.mask_dsn(value)
                self.assertIn("***", masked)
                self.assertNotIn("p%40ss", masked)
                self.assertNotIn("p@ss", masked)
                self.assertNotIn("secret with space", masked)
                self.assertNotIn("query-secret", masked)
                self.assertNotIn("cert-secret", masked)

        # oauth_client_secret is available only with newer libpq builds. Older
        # versions reject the key; either result must fail closed.
        oauth = self.cfg.mask_dsn(
            "host=db.example user=alice oauth_client_secret=oauth-secret"
        )
        self.assertNotIn("oauth-secret", oauth)
        self.assertTrue("***" in oauth or oauth == "[invalid PostgreSQL DSN]")

        invalid = "postgresql://alice:top-secret@["
        masked = self.cfg.mask_dsn(invalid)
        self.assertEqual(masked, "[invalid PostgreSQL DSN]")
        self.assertNotIn("top-secret", masked)

    def test_public_config_value_redacts_sensitive_keys(self):
        remote = self.cfg.public_config_value(
            "remote", "postgresql://alice:secret@db.example/khala"
        )
        self.assertNotIn("secret", remote)
        self.assertEqual(self.cfg.public_config_value("nested.password", "secret"), "[REDACTED]")
        self.assertEqual(self.cfg.public_config_value("embedding.model", "model"), "model")
        self.assertEqual(
            self.cfg.public_config_value(
                "embedding", {"model": "model", "api_key": "nested-secret"}
            ),
            {"model": "model", "api_key": "[REDACTED]"},
        )


class KhalaConfigSetGetTest(unittest.TestCase):
    """dot-path config get/set helpers."""

    def setUp(self):
        import khala.config as cfg

        self.cli = cfg

    def test_set_and_get_scalar(self):
        d = {"workspaces": []}
        self.cli.config_set(d, "embedding.model", "openai/text-embedding-3-small")
        self.assertEqual(
            self.cli.config_get(d, "embedding.model"), "openai/text-embedding-3-small"
        )

    def test_set_list_index_and_get(self):
        d = {"workspaces": []}
        self.cli.config_set(d, "workspaces[0]", "/a")
        self.cli.config_set(d, "workspaces[1]", "/b")
        self.assertEqual(d["workspaces"], ["/a", "/b"])
        self.assertEqual(self.cli.config_get(d, "workspaces[1]"), "/b")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.cli.config_get({"a": {"b": 1}}, "a.c"))

    def test_set_json_value_via_get(self):
        d = {"embedding": {}}
        self.cli.config_set(d, "embedding.provider", "openrouter")
        self.assertEqual(self.cli.config_get(d, "embedding.provider"), "openrouter")


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
        isolated_git = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        }
        with patch.dict(os.environ, isolated_git):
            user = resolve_user(self.dir)
        self.assertEqual(user.name, os.environ.get("USER"))
        self.assertIsNone(user.email)

    def test_git_global_config_is_used(self):
        user = resolve_user(self.dir)
        self.assertIsNotNone(user.name)


if __name__ == "__main__":
    unittest.main()
