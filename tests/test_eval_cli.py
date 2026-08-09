from __future__ import annotations

import io
import json
import os
import sqlite3
import stat
import sys
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import psycopg
from khala import main
from khala.store import Store


class EvalCLIIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database = self.root / "khala.db"
        self.config_directory = self.root / "missing-config"

        from khala import config

        self.config = config
        self.config_directory_patcher = patch.object(config, "CONFIG_DIR", self.config_directory)
        self.config_file_patcher = patch.object(
            config, "CONFIG_FILE", self.config_directory / "config.json"
        )
        self.config_directory_patcher.start()
        self.config_file_patcher.start()

    def tearDown(self) -> None:
        self.config_file_patcher.stop()
        self.config_directory_patcher.stop()
        self.temporary_directory.cleanup()

    def _run(self, *arguments: str) -> tuple[int, str, str]:
        previous_stdout, sys.stdout = sys.stdout, io.StringIO()
        previous_stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            try:
                code = main(["--db", str(self.database), *arguments])
            except SystemExit as error:
                code = int(error.code or 0)
            return code, sys.stdout.getvalue(), sys.stderr.getvalue()
        finally:
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr

    def _seed_event(self, content: str = "deploy completed successfully") -> str:
        store = Store(self.database)
        try:
            session_id = store.add_session("test")
            return store.add_event(session_id, "assistant", content)
        finally:
            store.close()

    def _write_dataset(self, event_id: str) -> Path:
        dataset = self.root / "eval.jsonl"
        dataset.write_text(
            json.dumps({"query": "deploy", "gold": [event_id]}) + "\n",
            encoding="utf-8",
        )
        return dataset

    def test_baseline_early_dispatch_does_not_create_database_or_config(self) -> None:
        with (
            patch("khala.cli.Store", side_effect=AssertionError("opened product store")),
            patch.object(self.config, "ensure_config_directory") as ensure_config_directory,
        ):
            code, stdout, stderr = self._run("eval", "--baseline")

        self.assertEqual(code, 0, stderr)
        self.assertTrue(json.loads(stdout)["passed"])
        self.assertFalse(self.database.exists())
        self.assertFalse(self.config_directory.exists())
        ensure_config_directory.assert_not_called()

    def test_baseline_quality_regression_returns_exit_one(self) -> None:
        fixture = json.loads(
            resources.files("khala.evaluation")
            .joinpath("gold_baseline.json")
            .read_text(encoding="utf-8")
        )
        fixture["cases"][0]["query"] = "zzzznonexistenttoken"
        path = self.root / "regressed-baseline.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")

        code, stdout, stderr = self._run(
            "eval", "--baseline", "--baseline-file", str(path)
        )

        self.assertEqual(code, 1, stderr)
        report = json.loads(stdout)
        self.assertFalse(report["passed"])
        self.assertEqual(report["must_hit_failures"], ["q.atomic"])
        self.assertFalse(self.database.exists())
        self.assertFalse(self.config_directory.exists())

    def test_repeated_local_eval_does_not_write_recall_state(self) -> None:
        event_id = self._seed_event()
        dataset = self._write_dataset(event_id)

        with patch.object(self.config, "ensure_config_directory") as ensure_config_directory:
            first = self._run("eval", "--dataset", str(dataset), "--backend", "local", "--k", "1")
            second = self._run("eval", "--dataset", str(dataset), "--backend", "local", "--k", "1")

        self.assertEqual(first[0], 0, first[2])
        self.assertEqual(second[0], 0, second[2])
        first_report = json.loads(first[1])
        second_report = json.loads(second[1])
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["backend"], "local")
        self.assertEqual(first_report["mode"], "lexical")
        self.assertEqual(first_report["hit_rate"], 1.0)
        with sqlite3.connect(self.database) as connection:
            recall_rows = connection.execute(
                """
                SELECT (SELECT count(*) FROM recalls)
                     + (SELECT count(*) FROM recall_requests)
                """
            ).fetchone()[0]
        self.assertEqual(recall_rows, 0)
        self.assertFalse(self.config_directory.exists())
        ensure_config_directory.assert_not_called()

    def test_dataset_and_generate_are_mutually_exclusive(self) -> None:
        code, _, stderr = self._run(
            "eval", "--dataset", str(self.root / "eval.jsonl"), "--generate"
        )

        self.assertEqual(code, 2)
        self.assertIn("not allowed with argument", stderr)
        self.assertFalse(self.database.exists())

    def test_eval_requires_exactly_one_input_source(self) -> None:
        code, _, stderr = self._run("eval")

        self.assertEqual(code, 2)
        self.assertIn("one of the arguments", stderr)
        self.assertFalse(self.database.exists())

    def test_eval_rejects_out_of_range_k_and_sample_before_opening_database(self) -> None:
        dataset = self.root / "does-not-need-to-exist.jsonl"
        cases = (
            (("--k", "0"), "k must be between 1 and 100"),
            (("--k", "101"), "k must be between 1 and 100"),
            (("--sample", "0"), "sample must be between 1 and 100"),
            (("--sample", "101"), "sample must be between 1 and 100"),
        )

        for options, expected_error in cases:
            with self.subTest(options=options):
                code, stdout, stderr = self._run("eval", "--dataset", str(dataset), *options)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn(expected_error, stderr)
                self.assertFalse(self.database.exists())

    def test_output_requires_generate(self) -> None:
        code, _, stderr = self._run(
            "eval",
            "--dataset",
            str(self.root / "eval.jsonl"),
            "--output",
            str(self.root / "generated.jsonl"),
        )

        self.assertEqual(code, 2)
        self.assertIn("output requires --generate", stderr)
        self.assertFalse(self.database.exists())

    def test_sample_is_rejected_when_a_dataset_would_ignore_it(self) -> None:
        code, stdout, stderr = self._run(
            "eval",
            "--dataset",
            str(self.root / "eval.jsonl"),
            "--sample",
            "10",
        )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("sample applies only to --generate", stderr)
        self.assertFalse(self.database.exists())

    def test_malformed_dataset_fails_without_recall_writes(self) -> None:
        self._seed_event()
        dataset = self.root / "malformed.jsonl"
        dataset.write_text("not JSON\n", encoding="utf-8")

        code, stdout, stderr = self._run("eval", "--dataset", str(dataset))

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("invalid JSON", stderr)
        with sqlite3.connect(self.database) as connection:
            recall_rows = connection.execute(
                """
                SELECT (SELECT count(*) FROM recalls)
                     + (SELECT count(*) FROM recall_requests)
                """
            ).fetchone()[0]
        self.assertEqual(recall_rows, 0)

    def test_natural_language_punctuation_is_safe_for_local_fts(self) -> None:
        event_id = self._seed_event("atomic sync commits events and watermark together")
        dataset = self._write_dataset(event_id)
        dataset.write_text(
            json.dumps({"query": "what is atomic sync?", "gold": [event_id]}) + "\n",
            encoding="utf-8",
        )

        code, stdout, stderr = self._run(
            "eval", "--dataset", str(dataset), "--k", "1"
        )

        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["hit_rate"], 1.0)

    def test_sqlite_schema_error_is_reported_without_a_traceback(self) -> None:
        sqlite3.connect(self.database).close()
        dataset = self._write_dataset("event-1")

        code, stdout, stderr = self._run("eval", "--dataset", str(dataset))

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.startswith("khala: "))
        self.assertNotIn("Traceback", stderr)

    def test_invalid_dataset_fails_before_remote_connection(self) -> None:
        dataset = self.root / "malformed.jsonl"
        dataset.write_text("not JSON\n", encoding="utf-8")
        with (
            patch("khala.cli._remote_dsn", return_value="postgresql://example/db"),
            patch(
                "khala.remote.RemoteStore.ping",
                side_effect=AssertionError("remote connection should not be attempted"),
            ) as ping,
        ):
            code, stdout, stderr = self._run(
                "eval", "--dataset", str(dataset), "--backend", "remote"
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("invalid JSON", stderr)
        ping.assert_not_called()

    def test_generate_fails_for_missing_key_before_opening_database(self) -> None:
        with (
            patch("khala.evaluation.llm._api_key", return_value=None),
            patch(
                "khala.evaluation.backends.LocalEvalBackend",
                side_effect=AssertionError("database should not be opened"),
            ) as local_backend,
        ):
            code, stdout, stderr = self._run("eval", "--generate")

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("OPENROUTER_API_KEY is not set", stderr)
        local_backend.assert_not_called()

    def test_judge_and_read_workloads_are_bounded_before_backend_open(self) -> None:
        dataset = self._write_dataset("event-1")
        cases = (
            ("MAX_JUDGE_RESULTS", "--judge", "judge workload"),
            ("MAX_READ_RESULTS", "--read", "read output"),
        )
        for constant, option, message in cases:
            with self.subTest(option=option), patch(
                f"khala.eval_cli.{constant}", 1
            ), patch(
                "khala.evaluation.backends.LocalEvalBackend",
                side_effect=AssertionError("backend should not be opened"),
            ) as local_backend:
                code, stdout, stderr = self._run(
                    "eval", "--dataset", str(dataset), "--k", "2", option
                )

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn(message, stderr)
            local_backend.assert_not_called()

    def test_remote_semantic_query_workload_is_bounded_before_connect(self) -> None:
        dataset = self.root / "two-cases.jsonl"
        dataset.write_text(
            json.dumps({"query": "first", "gold": ["event-1"]})
            + "\n"
            + json.dumps({"query": "second", "gold": ["event-2"]})
            + "\n",
            encoding="utf-8",
        )
        with (
            patch("khala.cli._remote_dsn", return_value="postgresql://example/db"),
            patch("khala.eval_cli.MAX_REMOTE_SEMANTIC_QUERIES", 1),
            patch(
                "khala.remote.RemoteStore.ping",
                side_effect=AssertionError("remote should not be opened"),
            ) as ping,
        ):
            code, stdout, stderr = self._run(
                "eval",
                "--dataset",
                str(dataset),
                "--backend",
                "remote",
                "--mode",
                "semantic",
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("remote semantic workload", stderr)
        ping.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_generated_dataset_is_private_without_tightening_its_parent(self) -> None:
        self._seed_event("deploy completed successfully with deterministic content")
        output_directory = self.root / "shared-output"
        output_directory.mkdir(mode=0o755)
        output_directory.chmod(0o755)
        generated = output_directory / "generated.jsonl"

        with patch(
            "khala.eval_cli.make_question_generator",
            return_value=lambda _content: "deploy",
        ):
            code, stdout, stderr = self._run(
                "eval",
                "--generate",
                "--sample",
                "1",
                "--k",
                "1",
                "--output",
                str(generated),
            )

        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["hit_rate"], 1.0)
        self.assertEqual(stat.S_IMODE(generated.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(output_directory.stat().st_mode), 0o755)

    def test_baseline_rejects_options_that_would_be_silently_ignored(self) -> None:
        options_to_reject = (
            ("--backend", "local"),
            ("--backend", "remote"),
            ("--mode", "lexical"),
            ("--mode", "semantic"),
            ("--k", "5"),
            ("--sample", "20"),
        )
        for options in options_to_reject:
            with self.subTest(options=options):
                code, stdout, stderr = self._run("eval", "--baseline", *options)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("baseline", stderr)
                self.assertFalse(self.database.exists())

    def test_local_semantic_eval_is_rejected_without_creating_state(self) -> None:
        code, stdout, stderr = self._run(
            "eval",
            "--dataset",
            str(self.root / "eval.jsonl"),
            "--backend",
            "local",
            "--mode",
            "semantic",
        )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("local eval supports only lexical mode", stderr)
        self.assertFalse(self.database.exists())
        self.assertFalse(self.config_directory.exists())

    def test_remote_connection_failure_never_falls_back_to_local(self) -> None:
        dsn = "postgresql://example.invalid/khala"
        dataset = self._write_dataset("event-1")
        with (
            patch("khala.cli._remote_dsn", return_value=dsn),
            patch(
                "khala.remote.RemoteStore.ping",
                side_effect=psycopg.OperationalError("database unavailable"),
            ),
            patch(
                "khala.evaluation.backends.LocalEvalBackend",
                side_effect=AssertionError("unexpected local fallback"),
            ) as local_backend,
        ):
            code, stdout, stderr = self._run(
                "eval",
                "--dataset",
                str(dataset),
                "--backend",
                "remote",
                "--mode",
                "hybrid",
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("PostgreSQL OperationalError", stderr)
        self.assertFalse(self.database.exists())
        local_backend.assert_not_called()

    def test_remote_eval_without_connection_does_not_fall_back_to_local(self) -> None:
        with (
            patch("khala.cli._remote_dsn", return_value=None),
            patch(
                "khala.evaluation.backends.LocalEvalBackend",
                side_effect=AssertionError("unexpected local fallback"),
            ) as local_backend,
        ):
            code, stdout, stderr = self._run(
                "eval",
                "--dataset",
                str(self.root / "eval.jsonl"),
                "--backend",
                "remote",
            )

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("remote eval requires `khala connect` first", stderr)
        self.assertFalse(self.database.exists())
        local_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
