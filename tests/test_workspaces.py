from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala import config
from khala.workspace_registry import add_workspace, remove_workspace
from khala.workspaces import (
    WORKSPACE_GIT_COMMON_DIRS_KEY,
    is_enabled_workspace,
    normalize_git_common_dirs,
    reconcile_git_common_dirs,
    resolve_git_common_dir,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for worktree tracking tests")
class GitWorktreeConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self._git("init", "-q", cwd=self.repository)
        self._git(
            "-c",
            "user.name=Khala Test",
            "-c",
            "user.email=khala@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
            cwd=self.repository,
        )

        self.config_directory = self.root / "config"
        self.config_directory_patcher = patch.object(
            config, "CONFIG_DIR", self.config_directory
        )
        self.config_file_patcher = patch.object(
            config, "CONFIG_FILE", self.config_directory / "config.json"
        )
        self.config_directory_patcher.start()
        self.config_file_patcher.start()

    def tearDown(self) -> None:
        self.config_file_patcher.stop()
        self.config_directory_patcher.stop()
        self.temporary_directory.cleanup()

    def _git(self, *arguments: str, cwd: Path) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

    def _add_worktree(self, name: str) -> Path:
        worktree = self.root / name
        self._git(
            "worktree",
            "add",
            "-q",
            "-b",
            f"branch-{name}",
            str(worktree),
            cwd=self.repository,
        )
        return worktree

    def test_registration_enables_existing_and_future_sibling_worktrees(self) -> None:
        registered = self._add_worktree("registered")
        existing_sibling = self._add_worktree("existing")

        config.save_config({"workspaces": [str(registered)], "remote": None})

        stored = config.load_config()
        common_dirs = stored[WORKSPACE_GIT_COMMON_DIRS_KEY]
        self.assertEqual(
            common_dirs,
            {str(registered.resolve()): str((self.repository / ".git").resolve())},
        )
        self.assertTrue(config.is_enabled_workspace(str(existing_sibling)))

        future_sibling = self._add_worktree("future")
        nested_directory = future_sibling / "nested"
        nested_directory.mkdir()
        self.assertTrue(config.is_enabled_workspace(str(nested_directory)))

        unrelated_repository = self.root / "unrelated"
        unrelated_repository.mkdir()
        self._git("init", "-q", cwd=unrelated_repository)
        self.assertFalse(config.is_enabled_workspace(str(unrelated_repository)))

    def test_legacy_path_only_config_enables_sibling_worktree(self) -> None:
        registered = self._add_worktree("legacy-registered")
        sibling = self._add_worktree("legacy-sibling")
        config.ensure_config_directory()
        config.CONFIG_FILE.write_text(
            json.dumps({"workspaces": [str(registered)], "remote": None})
        )

        loaded = config.load_config()
        self.assertNotIn(WORKSPACE_GIT_COMMON_DIRS_KEY, loaded)
        self.assertTrue(config.is_enabled_workspace(str(sibling)))

    def test_readding_legacy_workspace_backfills_git_identity(self) -> None:
        from khala import main

        registered = self._add_worktree("legacy-readded")
        config.ensure_config_directory()
        config.CONFIG_FILE.write_text(
            json.dumps({"workspaces": [str(registered.resolve())], "remote": None})
        )

        with patch.object(sys, "stdout", io.StringIO()) as stdout:
            code = main(
                [
                    "--db",
                    str(self.root / "khala.db"),
                    "workspace",
                    "add",
                    str(registered),
                ]
            )

        stored = config.load_config()
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["workspaces"],
            [str(registered.resolve())],
        )
        self.assertEqual(
            stored[WORKSPACE_GIT_COMMON_DIRS_KEY],
            {str(registered.resolve()): str((self.repository / ".git").resolve())},
        )

    def test_registration_survives_deletion_of_original_worktree(self) -> None:
        registered = self._add_worktree("ephemeral")
        config.save_config({"workspaces": [str(registered)], "remote": None})
        common_dirs = config.load_config()[WORKSPACE_GIT_COMMON_DIRS_KEY]

        self._git("worktree", "remove", str(registered), cwd=self.repository)
        config_data = config.load_config()
        config_data["remote"] = "postgresql://example.invalid/khala"
        config.save_config(config_data)

        self.assertEqual(
            config.load_config()[WORKSPACE_GIT_COMMON_DIRS_KEY],
            common_dirs,
        )
        future_sibling = self._add_worktree("replacement")
        self.assertTrue(config.is_enabled_workspace(str(future_sibling)))

    def test_removal_prunes_only_the_removed_registration(self) -> None:
        first = self._add_worktree("first")
        second = self._add_worktree("second")
        config.save_config({"workspaces": [str(first), str(second)], "remote": None})

        config_data = config.load_config()
        config_data["workspaces"] = [str(second)]
        config.save_config(config_data)

        common_dirs = config.load_config()[WORKSPACE_GIT_COMMON_DIRS_KEY]
        self.assertEqual(set(common_dirs), {str(second.resolve())})
        self.assertTrue(config.is_enabled_workspace(str(first)))

        config_data = config.load_config()
        config_data["workspaces"] = []
        config.save_config(config_data)

        self.assertNotIn(WORKSPACE_GIT_COMMON_DIRS_KEY, config.load_config())
        self.assertFalse(config.is_enabled_workspace(str(first)))
        self.assertFalse(config.is_enabled_workspace(str(second)))


class WorkspaceTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_non_git_workspace_keeps_path_prefix_behavior(self) -> None:
        registered = self.root / "registered"
        nested = registered / "nested"
        sibling = self.root / "sibling"
        nested.mkdir(parents=True)
        sibling.mkdir()

        self.assertTrue(is_enabled_workspace(nested, [str(registered)]))
        self.assertFalse(is_enabled_workspace(sibling, [str(registered)]))
        self.assertEqual(reconcile_git_common_dirs([str(registered)], None), {})

    def test_existing_identity_survives_git_lookup_failure(self) -> None:
        registered = self.root / "registered"
        common_dir = self.root / "repository" / ".git"
        registered.mkdir()
        common_dir.mkdir(parents=True)
        existing = {str(registered.resolve()): str(common_dir.resolve())}

        for error in (
            OSError("git unavailable"),
            subprocess.TimeoutExpired(["git"], timeout=5),
        ):
            with self.subTest(error=type(error).__name__):
                with patch("khala.workspaces.subprocess.run", side_effect=error):
                    self.assertEqual(
                        reconcile_git_common_dirs([str(registered)], existing),
                        existing,
                    )
                    self.assertTrue(
                        is_enabled_workspace(registered, [str(registered)], existing)
                    )
                    self.assertFalse(
                        is_enabled_workspace(self.root, [str(registered)], existing)
                    )

    def test_git_nonzero_and_empty_output_fail_safely(self) -> None:
        for completed in (
            subprocess.CompletedProcess(["git"], 1, stdout="failure\n", stderr=""),
            subprocess.CompletedProcess(["git"], 0, stdout="  \n", stderr=""),
        ):
            with self.subTest(returncode=completed.returncode, stdout=completed.stdout):
                with patch("khala.workspaces.subprocess.run", return_value=completed):
                    self.assertIsNone(resolve_git_common_dir(self.root))

    def test_relative_git_common_dir_is_resolved_from_workspace(self) -> None:
        completed = subprocess.CompletedProcess(
            ["git"],
            0,
            stdout="../shared.git\n",
            stderr="",
        )
        with patch("khala.workspaces.subprocess.run", return_value=completed):
            self.assertEqual(
                resolve_git_common_dir(self.root),
                (self.root / "../shared.git").resolve(),
            )

    def test_invalid_tracking_metadata_is_ignored(self) -> None:
        workspace = self.root / "workspace"
        common_dir = self.root / ".git"
        expected = {str(workspace.resolve()): str(common_dir.resolve())}
        self.assertEqual(
            normalize_git_common_dirs(
                {
                    str(workspace): str(common_dir),
                    1: str(common_dir),
                    "wrong-value-type": 1,
                }
            ),
            expected,
        )
        self.assertEqual(normalize_git_common_dirs([str(common_dir)]), {})

    @unittest.skipUnless(os.name == "posix", "symbolic-link loops are POSIX-specific")
    def test_invalid_canonical_path_fails_closed(self) -> None:
        loop = self.root / "loop"
        loop.symlink_to(loop)

        self.assertIsNone(resolve_git_common_dir(loop))
        self.assertFalse(is_enabled_workspace(loop, [str(self.root / "unrelated")]))


class WorkspaceRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
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

    def test_add_is_idempotent_and_remove_accepts_a_deleted_checkout(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()

        add_workspace(first)
        repeated = add_workspace(first)
        added = add_workspace(second)
        shutil.rmtree(first)
        removed = remove_workspace(first)

        self.assertEqual(repeated.workspaces, (str(first.resolve()),))
        self.assertEqual(
            added.workspaces,
            (str(first.resolve()), str(second.resolve())),
        )
        self.assertEqual(removed.workspaces, (str(second.resolve()),))
        self.assertEqual(config.load_config()["workspaces"], [str(second.resolve())])

    def test_add_rejects_missing_or_unresolvable_workspace(self) -> None:
        missing = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "not a directory"):
            add_workspace(missing)

        with (
            patch("khala.workspace_registry.resolve_workspace_path", return_value=None),
            self.assertRaisesRegex(ValueError, "cannot be resolved"),
        ):
            add_workspace(self.root)

    def test_invalid_persisted_workspace_collection_is_replaced(self) -> None:
        workspace = self.root / "workspace"
        workspace.mkdir()
        with (
            patch("khala.config.load_config", return_value={"workspaces": "invalid"}),
            patch("khala.config.save_config") as save_config,
        ):
            registration = add_workspace(workspace)

        self.assertEqual(registration.workspaces, (str(workspace.resolve()),))
        self.assertEqual(
            save_config.call_args.args[0]["workspaces"],
            [str(workspace.resolve())],
        )


if __name__ == "__main__":
    unittest.main()
