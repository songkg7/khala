from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
import unittest
import urllib.error
from importlib import resources
from pathlib import Path
from typing import Any, cast
from unittest import mock

from khala import eval_cli as eval_cli_module
from khala.evaluation import dataset as dataset_module
from khala.evaluation import llm as llm_module
from khala.evaluation.backends import LocalEvalBackend, RemoteEvalBackend, select_backend
from khala.evaluation.baseline import BASELINE_RESOURCE, run_baseline
from khala.evaluation.dataset import load_jsonl, write_jsonl
from khala.evaluation.metrics import (
    graded_ndcg_at_k,
    hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    unique_ranked_ids,
)
from khala.evaluation.models import EvalCase, Judgment
from khala.evaluation.runner import evaluate_queries
from khala.local_search import build_match_query
from khala.models import ArchivedSession
from khala.store import Store


def _read_baseline_fixture() -> dict[str, Any]:
    text = (
        resources.files("khala.evaluation")
        .joinpath(BASELINE_RESOURCE)
        .read_text(encoding="utf-8")
    )
    value = json.loads(text)
    if not isinstance(value, dict):
        raise AssertionError("the packaged baseline must be a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class EvaluationModelsAndMetricsTest(unittest.TestCase):
    def test_eval_case_and_judgment_reject_invalid_values(self) -> None:
        for query, gold in [
            ("", ("event",)),
            ("   ", ("event",)),
            ("query", ()),
            ("query", ("",)),
            ("query", ("event", "event")),
        ]:
            with self.subTest(query=query, gold=gold):
                with self.assertRaises(ValueError):
                    EvalCase(query, gold)

        for score, reason in [(-1, "bad"), (3, "bad"), (0, ""), (1, "  ")]:
            with self.subTest(score=score, reason=reason):
                with self.assertRaises(ValueError):
                    Judgment(score, reason)

        invalid_cases = [
            (cast(str, 1), ("event",)),
            ("query", cast(tuple[str, ...], ["event"])),
            ("query", (cast(str, 1),)),
        ]
        for query, gold in invalid_cases:
            with self.subTest(query=query, gold=gold):
                with self.assertRaises(TypeError):
                    EvalCase(query, gold)

        with self.assertRaisesRegex(TypeError, "score must be an integer"):
            Judgment(cast(int, True), "reason")
        with self.assertRaisesRegex(TypeError, "reason must be a string"):
            Judgment(1, cast(str, None))
        self.assertEqual(Judgment(1, "x" * 500).score, 1)
        with self.assertRaisesRegex(ValueError, "reason cannot exceed"):
            Judgment(1, "x" * 501)

    def test_binary_metrics_are_rank_sensitive_and_deduplicate_results(self) -> None:
        results = ["irrelevant", "gold-a", "gold-a", "gold-b"]
        gold = {"gold-a", "gold-b"}

        self.assertEqual(unique_ranked_ids(results, 3), ["irrelevant", "gold-a", "gold-b"])
        self.assertEqual(hit_rate(results, gold, 3), 1.0)
        self.assertEqual(recall_at_k(results, gold, 3), 1.0)
        self.assertEqual(reciprocal_rank(results, gold, 3), 0.5)
        expected_ndcg = (1 / math.log2(3) + 1 / math.log2(4)) / (
            1 + 1 / math.log2(3)
        )
        self.assertAlmostEqual(ndcg_at_k(results, gold, 3), expected_ndcg)

        self.assertEqual(hit_rate([], gold, 3), 0.0)
        self.assertEqual(recall_at_k(results, set(), 3), 0.0)
        self.assertEqual(reciprocal_rank(["other"], gold, 1), 0.0)
        self.assertEqual(ndcg_at_k(["other"], set(), 1), 0.0)

    def test_duplicates_cannot_consume_the_cutoff_or_inflate_recall(self) -> None:
        results = ["gold-a", "gold-a", "gold-b"]
        gold = {"gold-a", "gold-b"}

        self.assertEqual(unique_ranked_ids(results, 2), ["gold-a", "gold-b"])
        self.assertEqual(recall_at_k(results, gold, 2), 1.0)
        self.assertEqual(ndcg_at_k(results, gold, 2), 1.0)

    def test_metric_inputs_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "k must be positive"):
            unique_ranked_ids(["event"], 0)
        for function in (hit_rate, reciprocal_rank, ndcg_at_k):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "k must be positive"):
                    function(["event"], {"event"}, 0)

        with self.assertRaisesRegex(ValueError, "k must be positive"):
            graded_ndcg_at_k([2], 0)
        with self.assertRaisesRegex(ValueError, "only 0, 1, or 2"):
            graded_ndcg_at_k([2, 3], 2)

    def test_graded_ndcg_rewards_correct_order(self) -> None:
        self.assertEqual(graded_ndcg_at_k([2, 1, 0], 3), 1.0)
        self.assertLess(graded_ndcg_at_k([0, 1, 2], 3), 1.0)
        self.assertEqual(graded_ndcg_at_k([0, 0], 2), 0.0)


class EvaluationRunnerTest(unittest.TestCase):
    def test_evaluate_deduplicates_rows_before_metrics_judging_and_collection(self) -> None:
        recall_calls: list[tuple[str, int]] = []
        judge_calls: list[tuple[str, str]] = []

        def recall(query: str, limit: int) -> list[dict[str, str]]:
            recall_calls.append((query, limit))
            return [
                {"id": "gold", "content": "first copy"},
                {"id": "gold", "content": "duplicate copy"},
                {"id": "other", "content": "false positive"},
            ]

        def judge(query: str, content: str) -> Judgment:
            judge_calls.append((query, content))
            if content == "first copy":
                return Judgment(2, "direct")
            return Judgment(1, "tangential")

        report = evaluate_queries(
            [EvalCase("needle", ("gold",))],
            recall,
            k=3,
            judge=judge,
            collect=True,
            judge_model="strict-judge",
        )

        self.assertEqual(recall_calls, [("needle", 3)])
        self.assertEqual(
            judge_calls,
            [("needle", "first copy"), ("needle", "false positive")],
        )
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertEqual(report["mrr"], 1.0)
        self.assertEqual(report["ndcg"], 1.0)
        self.assertEqual(report["judge"], "strict-judge")
        self.assertEqual(report["judge_precision_at_k"], 0.6667)
        self.assertEqual(report["judge_gold_agreement"], 0.5)
        self.assertEqual(
            [row["id"] for row in report["rows"][0]["results"]],
            ["gold", "other"],
        )
        self.assertEqual(report["rows"][0]["results"][0]["reason"], "direct")

    def test_judge_precision_uses_requested_k_when_backend_returns_fewer_rows(self) -> None:
        report = evaluate_queries(
            [EvalCase("needle", ("gold",))],
            lambda _query, _limit: [{"id": "gold", "content": "answer"}],
            k=3,
            judge=lambda _query, _content: Judgment(2, "relevant"),
        )

        self.assertEqual(report["judge_precision_at_k"], 0.3333)
        self.assertEqual(report["judge_gold_agreement"], 1.0)
        self.assertEqual(report["judge_ndcg"], 1.0)

    def test_collection_truncates_content_and_works_without_a_judge(self) -> None:
        long_content = "x" * 350
        report = evaluate_queries(
            [EvalCase("needle", ("gold",))],
            lambda _query, _limit: [{"id": "gold", "content": long_content}],
            collect=True,
        )

        result = report["rows"][0]["results"][0]
        self.assertEqual(result["content"], "x" * 300)
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "")
        self.assertNotIn("judge", report)

    def test_collect_preserves_provenance_fields_when_present(self) -> None:
        row = {
            "id": "event-1",
            "session_id": "session-1",
            "role": "assistant",
            "created_at": "2026-01-01T00:00:00Z",
            "content": "answer text",
            "semantic_rank": 1,
            "semantic_distance": 0.12,
            "lexical_rank": 2,
            "rank": 0.5,
        }
        report = evaluate_queries(
            [EvalCase("needle", ("event-1",))],
            lambda _query, _limit: [row],
            collect=True,
        )

        result = report["rows"][0]["results"][0]
        self.assertEqual(result["id"], "event-1")
        self.assertEqual(result["session_id"], "session-1")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["created_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(result["semantic_rank"], 1)
        self.assertEqual(result["semantic_distance"], 0.12)
        self.assertEqual(result["lexical_rank"], 2)
        self.assertEqual(result["rank"], 0.5)

    def test_collect_omits_missing_provenance_as_none(self) -> None:
        row = {
            "id": "event-1",
            "session_id": "session-1",
            "role": "user",
            "content": "question",
        }
        report = evaluate_queries(
            [EvalCase("needle", ("event-1",))],
            lambda _query, _limit: [row],
            collect=True,
        )

        result = report["rows"][0]["results"][0]
        self.assertIsNone(result["semantic_rank"])
        self.assertIsNone(result["semantic_distance"])
        self.assertIsNone(result["lexical_rank"])
        self.assertIsNone(result["rank"])

    def test_empty_judged_results_have_zero_judge_metrics(self) -> None:
        report = evaluate_queries(
            [EvalCase("needle", ("gold",))],
            lambda _query, _limit: [],
            judge=lambda _query, _content: Judgment(0, "unused"),
        )

        self.assertEqual(report["judge_ndcg"], 0.0)
        self.assertEqual(report["judge_precision_at_k"], 0.0)
        self.assertEqual(report["judge_gold_agreement"], 0.0)

    def test_runner_validates_cases_k_and_result_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_queries([], lambda _query, _limit: [])
        with self.assertRaisesRegex(ValueError, "k must be positive"):
            evaluate_queries(
                [EvalCase("needle", ("gold",))],
                lambda _query, _limit: [],
                k=0,
            )

        for row in ({}, {"id": ""}, {"id": 1}):
            with self.subTest(row=row):
                with self.assertRaisesRegex(TypeError, "non-empty string"):
                    evaluate_queries(
                        [EvalCase("needle", ("gold",))],
                        lambda _query, _limit, row=row: [row],
                    )

    def test_runner_requires_string_content_when_disclosing_or_judging(self) -> None:
        case = [EvalCase("needle", ("gold",))]

        def invalid_recall(_query: str, _limit: int) -> list[dict[str, Any]]:
            return [{"id": "gold", "content": None}]

        with self.assertRaisesRegex(TypeError, "judged recall result content"):
            evaluate_queries(
                case,
                invalid_recall,
                judge=lambda _query, _content: Judgment(1, "unused"),
            )
        with self.assertRaisesRegex(TypeError, "collected recall result content"):
            evaluate_queries(case, invalid_recall, collect=True)

    def test_runner_stops_at_k_unique_results(self) -> None:
        report = evaluate_queries(
            [EvalCase("needle", ("first",))],
            lambda _query, _limit: [
                {"id": "first", "content": "one"},
                {"id": "second", "content": "two"},
                {"id": "third", "content": "three"},
            ],
            k=2,
            collect=True,
        )

        self.assertEqual(
            [row["id"] for row in report["rows"][0]["results"]],
            ["first", "second"],
        )


class EvaluationWorkloadTest(unittest.TestCase):
    def test_judge_read_and_remote_semantic_bounds_are_inclusive(self) -> None:
        common = {
            "k": 1,
            "backend": "local",
            "mode": "lexical",
        }
        with mock.patch.object(eval_cli_module, "MAX_JUDGE_RESULTS", 2):
            eval_cli_module._validate_workload(
                case_count=2, judge=True, read=False, **common
            )
            with self.assertRaisesRegex(ValueError, "judge workload"):
                eval_cli_module._validate_workload(
                    case_count=3, judge=True, read=False, **common
                )

        with mock.patch.object(eval_cli_module, "MAX_READ_RESULTS", 2):
            eval_cli_module._validate_workload(
                case_count=2, judge=False, read=True, **common
            )
            with self.assertRaisesRegex(ValueError, "read output"):
                eval_cli_module._validate_workload(
                    case_count=3, judge=False, read=True, **common
                )

        remote = {
            "k": 1,
            "judge": False,
            "read": False,
            "backend": "remote",
            "mode": "semantic",
        }
        with mock.patch.object(eval_cli_module, "MAX_REMOTE_SEMANTIC_QUERIES", 2):
            eval_cli_module._validate_workload(case_count=2, **remote)
            with self.assertRaisesRegex(ValueError, "remote semantic workload"):
                eval_cli_module._validate_workload(case_count=3, **remote)


class EvaluationDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_jsonl_load_and_private_round_trip(self) -> None:
        source = self.directory / "source.jsonl"
        source.write_text(
            '{"query": "  첫 질문  ", "gold": ["event-a", "event-b"]}\n'
            '{"query": "second", "gold": ["event-c"]}\n',
            encoding="utf-8",
        )

        cases = load_jsonl(source)

        self.assertEqual(
            cases,
            [
                EvalCase("첫 질문", ("event-a", "event-b")),
                EvalCase("second", ("event-c",)),
            ],
        )
        output = self.directory / "nested" / "output.jsonl"
        write_jsonl(output, iter(cases))
        self.assertEqual(load_jsonl(output), cases)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_loader_rejects_malformed_or_empty_datasets(self) -> None:
        malformed = self.directory / "malformed.jsonl"
        malformed.write_text('{"query":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            load_jsonl(malformed)

        empty = self.directory / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dataset is empty"):
            load_jsonl(empty)

        blank = self.directory / "blank.jsonl"
        blank.write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "blank lines are not allowed"):
            load_jsonl(blank)

        with self.assertRaises(FileNotFoundError):
            load_jsonl(self.directory / "missing.jsonl")

    def test_loader_rejects_invalid_rows_and_bounds(self) -> None:
        invalid_rows: list[tuple[str, Any]] = [
            ("not-object", []),
            ("query-type", {"query": 1, "gold": ["event"]}),
            ("gold-type", {"query": "query", "gold": "event"}),
            ("gold-item-type", {"query": "query", "gold": [1]}),
            ("missing-field", {"query": "query"}),
            ("extra-field", {"query": "query", "gold": ["event"], "extra": True}),
            ("empty-query", {"query": " ", "gold": ["event"]}),
            ("empty-gold", {"query": "query", "gold": []}),
            ("blank-gold", {"query": "query", "gold": [" "]}),
            ("padded-gold", {"query": "query", "gold": [" event"]}),
            ("duplicate-gold", {"query": "query", "gold": ["event", "event"]}),
            (
                "long-query",
                {"query": "q" * (dataset_module.MAX_QUERY_LENGTH + 1), "gold": ["event"]},
            ),
            (
                "too-many-gold",
                {
                    "query": "query",
                    "gold": [str(index) for index in range(dataset_module.MAX_GOLD_IDS + 1)],
                },
            ),
            (
                "long-event-id",
                {
                    "query": "query",
                    "gold": ["e" * (dataset_module.MAX_EVENT_ID_LENGTH + 1)],
                },
            ),
        ]
        path = self.directory / "invalid.jsonl"
        for name, row in invalid_rows:
            with self.subTest(name=name):
                _write_json(path, row)
                with self.assertRaises(ValueError):
                    load_jsonl(path)

        path.write_text(
            '{"query":"first","query":"second","gold":["event"]}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON field: query"):
            load_jsonl(path)

    def test_loader_enforces_file_and_case_limits(self) -> None:
        path = self.directory / "bounded.jsonl"
        path.write_text('{"query":"q","gold":["e"]}\n', encoding="utf-8")
        with mock.patch.object(dataset_module, "MAX_DATASET_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "exceeds 1 bytes"):
                load_jsonl(path)

        path.write_text(
            '{"query":"q1","gold":["e1"]}\n'
            '{"query":"q2","gold":["e2"]}\n',
            encoding="utf-8",
        )
        with mock.patch.object(dataset_module, "MAX_CASES", 1):
            with self.assertRaisesRegex(ValueError, "exceeds 1 cases"):
                load_jsonl(path)

    def test_writer_rejects_empty_input_and_symbolic_link_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty eval dataset"):
            write_jsonl(self.directory / "empty-output.jsonl", [])

        if os.name != "posix":
            return
        target = self.directory / "target.jsonl"
        target.write_text("original", encoding="utf-8")
        link = self.directory / "output-link.jsonl"
        link.symlink_to(target)
        with self.assertRaisesRegex(OSError, "symbolic link"):
            write_jsonl(link, [EvalCase("query", ("event",))])
        self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_writer_enforces_generated_dataset_limits(self) -> None:
        path = self.directory / "generated.jsonl"
        cases = [EvalCase("first", ("event-a",)), EvalCase("second", ("event-b",))]
        with mock.patch.object(dataset_module, "MAX_CASES", 1):
            with self.assertRaisesRegex(ValueError, "generated eval dataset exceeds 1 cases"):
                write_jsonl(path, cases)

        with mock.patch.object(dataset_module, "MAX_DATASET_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "generated eval dataset exceeds 1 bytes"):
                write_jsonl(path, cases[:1])

        oversized = EvalCase(
            "q" * (dataset_module.MAX_QUERY_LENGTH + 1),
            ("event",),
        )
        with self.assertRaisesRegex(ValueError, "query is too long"):
            write_jsonl(path, [oversized])


class EvaluationLlmTest(unittest.TestCase):
    @staticmethod
    def _urlopen_response(value: Any) -> mock.MagicMock:
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(value).encode(
            "utf-8"
        )
        return response

    def test_chat_parses_openrouter_response_and_sends_privacy_setting(self) -> None:
        response = self._urlopen_response(
            {"choices": [{"message": {"content": "  answer  "}}]}
        )
        with mock.patch.object(llm_module.urllib.request, "urlopen", return_value=response) as call:
            result = llm_module._chat("prompt", api_key="secret", model="model-id")

        self.assertEqual(result, "answer")
        request = call.call_args.args[0]
        self.assertEqual(call.call_args.kwargs["timeout"], 60)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "model-id")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(payload["provider"], {"data_collection": "deny"})
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    def test_chat_rejects_invalid_openrouter_shapes(self) -> None:
        invalid_values = [
            [],
            {},
            {"choices": "not-a-list"},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": []}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": " "}}]},
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                response = self._urlopen_response(value)
                with mock.patch.object(
                    llm_module.urllib.request, "urlopen", return_value=response
                ):
                    with self.assertRaises(ValueError):
                        llm_module._chat("prompt", api_key="secret", model="model")

    def test_chat_sanitizes_transport_and_decode_errors(self) -> None:
        http_error = urllib.error.HTTPError(
            llm_module.CHAT_URL,
            429,
            "contains provider details",
            cast(Any, None),
            None,
        )
        with mock.patch.object(
            llm_module.urllib.request, "urlopen", side_effect=http_error
        ):
            with self.assertRaisesRegex(ValueError, "eval LLM HTTP 429") as caught:
                llm_module._chat("prompt", api_key="secret", model="model")
        self.assertNotIn("provider details", str(caught.exception))

        for side_effect in (OSError("secret transport details"),):
            with mock.patch.object(
                llm_module.urllib.request, "urlopen", side_effect=side_effect
            ):
                with self.assertRaisesRegex(ValueError, "unreadable response") as caught:
                    llm_module._chat("prompt", api_key="secret", model="model")
            self.assertNotIn("secret transport details", str(caught.exception))

        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"not-json"
        with mock.patch.object(llm_module.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "unreadable response"):
                llm_module._chat("prompt", api_key="secret", model="model")

        oversized = mock.MagicMock()
        oversized.__enter__.return_value.read.return_value = b"x" * (
            llm_module.MAX_RESPONSE_BYTES + 1
        )
        with mock.patch.object(llm_module.urllib.request, "urlopen", return_value=oversized):
            with self.assertRaisesRegex(ValueError, "response is too large"):
                llm_module._chat("prompt", api_key="secret", model="model")

    def test_judgment_parser_accepts_fenced_json_and_rejects_invalid_fields(self) -> None:
        judgment = llm_module._parse_judgment(
            '```json\n{"score": 2, "reason": "direct answer"}\n```'
        )
        self.assertEqual(judgment, Judgment(2, "direct answer"))

        invalid = [
            "not JSON",
            '{"score":',
            '{"score": true, "reason": "bad"}',
            '{"score": 1.5, "reason": "bad"}',
            '{"score": 3, "reason": "bad"}',
            '{"score": 1}',
            '{"score": 1, "reason": 4}',
            '{"score": 1, "reason": "   "}',
            '{"score": 1, "reason": "ok", "extra": true}',
            '{"score": 1, "score": 2, "reason": "duplicate"}',
            'prefix {"score": 1, "reason": "wrapped"}',
            '```python\n{"score": 1, "reason": "wrong fence"}\n```',
            '```json\n{"score": 1, "reason": "missing close"}',
        ]
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    llm_module._parse_judgment(text)

    def test_api_key_selection_is_explicit_and_fails_closed(self) -> None:
        with mock.patch.object(llm_module, "_api_key") as fallback:
            self.assertEqual(llm_module._key("explicit"), "explicit")
            fallback.assert_not_called()

        with mock.patch.object(llm_module, "_api_key", return_value="fallback"):
            self.assertEqual(llm_module._key(None), "fallback")
        with mock.patch.object(llm_module, "_api_key", return_value=None):
            with self.assertRaisesRegex(ValueError, llm_module.API_KEY_ENV):
                llm_module._key(None)

    def test_generate_question_validates_input_model_and_output_length(self) -> None:
        content = "x" * 1_000
        with mock.patch.object(llm_module, "_chat", return_value="find this") as chat:
            question = llm_module.generate_question(content, api_key="secret")

        self.assertEqual(question, "find this")
        prompt = chat.call_args.args[0]
        self.assertIn("x" * 800, prompt)
        self.assertNotIn("x" * 801, prompt)
        self.assertEqual(chat.call_args.kwargs["api_key"], "secret")
        self.assertEqual(chat.call_args.kwargs["model"], llm_module.GENERATOR_MODEL)

        with self.assertRaisesRegex(ValueError, "empty content"):
            llm_module.generate_question("   ", api_key="secret")
        with mock.patch.object(llm_module, "_chat", return_value="q" * 501):
            with self.assertRaisesRegex(ValueError, "unexpectedly long"):
                llm_module.generate_question("memory", api_key="secret")

    def test_question_generator_resolves_the_key_before_sampling(self) -> None:
        with (
            mock.patch.object(llm_module, "_key", return_value="resolved") as resolve,
            mock.patch.object(llm_module, "generate_question", return_value="query") as generate,
        ):
            generator = llm_module.make_question_generator()
            result = generator("memory")

        self.assertEqual(result, "query")
        resolve.assert_called_once_with(None)
        generate.assert_called_once_with("memory", api_key="resolved")

    def test_make_judge_truncates_inputs_and_strictly_parses_chat_output(self) -> None:
        with mock.patch.object(
            llm_module,
            "_chat",
            return_value='{"score": 1, "reason": "partial"}',
        ) as chat:
            judge = llm_module.make_judge("secret")
            judgment = judge("q" * 500, "c" * 1_300)

        self.assertEqual(judgment, Judgment(1, "partial"))
        prompt = chat.call_args.args[0]
        self.assertIn("q" * 400, prompt)
        self.assertNotIn("q" * 401, prompt)
        self.assertIn("c" * 1_200, prompt)
        self.assertNotIn("c" * 1_201, prompt)
        self.assertEqual(chat.call_args.kwargs["api_key"], "secret")
        self.assertEqual(chat.call_args.kwargs["model"], llm_module.JUDGE_MODEL)

        with mock.patch.object(llm_module, "_chat", return_value="not JSON"):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                llm_module.make_judge("secret")("query", "content")


class GoldBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run_fixture(self, value: dict[str, Any]) -> dict[str, Any]:
        path = self.directory / "baseline.json"
        _write_json(path, value)
        return run_baseline(path)

    def test_packaged_baseline_is_deterministic_and_side_effect_free(self) -> None:
        first = run_baseline()
        second = run_baseline()

        self.assertEqual(first, second)
        self.assertEqual(first["baseline"], "phase3-local-lexical-v1")
        self.assertEqual(first["queries"], 6)
        self.assertEqual(first["k"], 3)
        for metric in ("hit_rate", "recall", "mrr", "ndcg"):
            self.assertEqual(first[metric], 1.0)
        self.assertEqual(first["must_hit_failures"], [])
        self.assertEqual(first["recall_state_writes"], 0)
        self.assertTrue(first["passed"])
        self.assertEqual(len(first["dataset_sha256"]), 64)

    def test_regressed_query_fails_the_per_case_and_aggregate_gates(self) -> None:
        value = _read_baseline_fixture()
        value["cases"][0]["query"] = "zzzznonexistenttoken"

        report = self._run_fixture(value)

        self.assertFalse(report["passed"])
        self.assertEqual(report["must_hit_failures"], ["q.atomic"])
        self.assertLess(report["hit_rate"], report["minimum"]["hit_rate"])
        self.assertEqual(report["recall_state_writes"], 0)

    def test_baseline_rejects_duplicate_and_unknown_logical_keys(self) -> None:
        def duplicate_session(value: dict[str, Any]) -> None:
            value["sessions"].append(copy.deepcopy(value["sessions"][0]))

        def duplicate_event(value: dict[str, Any]) -> None:
            value["events"].append(copy.deepcopy(value["events"][0]))

        def unknown_session(value: dict[str, Any]) -> None:
            value["events"][0]["session"] = "missing"

        def duplicate_case(value: dict[str, Any]) -> None:
            value["cases"].append(copy.deepcopy(value["cases"][0]))

        def duplicate_relevant(value: dict[str, Any]) -> None:
            value["cases"][0]["relevant"].append(value["cases"][0]["relevant"][0])

        def unknown_relevant(value: dict[str, Any]) -> None:
            value["cases"][0]["relevant"] = ["missing"]

        invalid_fixtures = [
            ("duplicate session", duplicate_session, "duplicate baseline session key"),
            ("duplicate event", duplicate_event, "duplicate baseline event key"),
            ("unknown session", unknown_session, "unknown session"),
            ("duplicate case", duplicate_case, "duplicate baseline case key"),
            ("duplicate relevant", duplicate_relevant, "duplicate relevant"),
            ("unknown relevant", unknown_relevant, "unknown events"),
        ]
        for name, mutate, message in invalid_fixtures:
            with self.subTest(name=name):
                value = _read_baseline_fixture()
                mutate(value)
                with self.assertRaisesRegex(ValueError, message):
                    self._run_fixture(value)

    def test_baseline_fixture_schema_and_thresholds_are_strict(self) -> None:
        invalid: list[tuple[str, Any, str]] = [
            ("schema_version", 2, "schema_version"),
            ("k", True, "baseline k"),
            ("minimum", {"hit_rate": 1.0}, "baseline minimum"),
        ]
        for field, replacement, message in invalid:
            with self.subTest(field=field):
                value = _read_baseline_fixture()
                value[field] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    self._run_fixture(value)

        value = _read_baseline_fixture()
        value["minimum"]["mrr"] = 1.1
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self._run_fixture(value)

        path = self.directory / "invalid-json.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            run_baseline(path)

    def test_custom_fixture_hashes_exact_input_bytes(self) -> None:
        value = _read_baseline_fixture()
        path = self.directory / "baseline-with-spacing.json"
        text = json.dumps(value, indent=4) + "\n"
        path.write_text(text, encoding="utf-8")

        report = run_baseline(path)

        self.assertEqual(
            report["dataset_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest()
        )


class LocalEvaluationBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _recall_count(database: Path) -> int:
        db = sqlite3.connect(database)
        try:
            return int(
                db.execute(
                    """
                    SELECT (SELECT count(*) FROM recalls)
                         + (SELECT count(*) FROM recall_requests)
                    """
                ).fetchone()[0]
            )
        finally:
            db.close()

    def test_natural_language_query_is_literal_deduplicated_and_stopword_safe(self) -> None:
        self.assertEqual(
            build_match_query("What is Atomic sync? atomic"),
            '"Atomic" OR "sync"',
        )
        self.assertEqual(build_match_query("what is it?"), '"what" OR "is" OR "it"')
        self.assertIsNone(build_match_query("?!"))

    def test_local_backend_search_and_sampling_are_reproducible_and_read_only(self) -> None:
        database = self.directory / "directory with spaces" / "khala eval.db"
        store = Store(database)
        session = store.add_session("test")
        expected = store.add_event(
            session,
            "assistant",
            "read only needle memory with enough content for deterministic sampling",
            created_at="2026-01-01T00:00:01Z",
        )
        store.add_event(session, "user", "short needle", created_at="2026-01-01T00:00:02Z")
        store.add_event(
            session,
            "system",
            "read only needle system content long enough but excluded from sampling",
            created_at="2026-01-01T00:00:03Z",
        )
        store.close()
        before = hashlib.sha256(database.read_bytes()).hexdigest()

        backend = LocalEvalBackend(database)
        try:
            first = backend.search("read only needle", 10)
            second = backend.search("read only needle", 10)
            samples = backend.sample_events(10)
            self.assertEqual(first, second)
            self.assertIn(expected, [str(row["id"]) for row in first])
            self.assertEqual(samples, [{"id": expected, "content": first[0]["content"]}])
            with self.assertRaises(sqlite3.OperationalError):
                backend.db.execute(
                    "INSERT INTO recalls(event_id, count, last_recalled_at) VALUES (?, 1, ?)",
                    (expected, "now"),
                )
        finally:
            backend.close()

        self.assertEqual(self._recall_count(database), 0)
        self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), before)

    def test_store_search_has_no_usage_side_effect_but_recall_records_usage(self) -> None:
        store = Store(self.directory / "store.db")
        self.addCleanup(store.close)
        session = store.add_session("test")
        event_id = store.add_event(session, "assistant", "needle memory")

        first = store.search("needle")
        second = store.search("needle")
        self.assertEqual(first, second)
        self.assertEqual(int(first[0]["recall_count"]), 0)
        self.assertEqual(
            int(store.db.execute("SELECT count(*) FROM recalls").fetchone()[0]), 0
        )
        self.assertEqual(
            int(store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0]),
            0,
        )

        recalled = store.recall("needle")
        self.assertEqual([row["id"] for row in recalled], [event_id])
        self.assertEqual(
            int(
                store.db.execute(
                    "SELECT count FROM recalls WHERE event_id = ?", (event_id,)
                ).fetchone()[0]
            ),
            1,
        )
        self.assertEqual(int(store.search("needle")[0]["recall_count"]), 1)
        self.assertEqual(
            int(store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0]),
            1,
        )
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            store.recall("needle", limit=0)

    def test_local_backend_and_selection_fail_fast_on_invalid_configuration(self) -> None:
        missing = self.directory / "missing.db"
        with self.assertRaisesRegex(FileNotFoundError, "eval database not found"):
            LocalEvalBackend(missing)
        with self.assertRaisesRegex(ValueError, "unsupported eval backend"):
            select_backend(
                database=missing,
                dsn=None,
                requested="automatic",
                mode="lexical",
            )
        with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
            select_backend(
                database=missing,
                dsn="postgresql://example/db",
                requested="remote",
                mode="automatic",
            )
        with self.assertRaisesRegex(ValueError, "only lexical"):
            select_backend(
                database=missing,
                dsn=None,
                requested="local",
                mode="semantic",
            )
        with self.assertRaisesRegex(ValueError, "requires `khala connect`"):
            select_backend(
                database=missing,
                dsn=None,
                requested="remote",
                mode="lexical",
            )

    def test_sampling_validates_positive_limit(self) -> None:
        store = Store(self.directory / "sample.db")
        self.addCleanup(store.close)
        with self.assertRaisesRegex(ValueError, "sample limit must be positive"):
            store.sample_events(0)


class RemoteEvaluationBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Path(self.temporary_directory.name) / "khala.db"

    def test_initialization_reads_then_pings_and_checks_archive_convergence(self) -> None:
        calls = mock.Mock()
        remote = mock.MagicMock()
        tombstones: list[ArchivedSession] = [
            {
                "id": "archived-session",
                "archived_at": "2026-01-01T00:00:00Z",
                "duplicate_of": "canonical-session",
            }
        ]
        remote.ping.side_effect = lambda: calls.ping()
        remote.archives_converged.side_effect = (
            lambda value: calls.archives_converged(value) or True
        )

        def read_tombstones(database: Path) -> list[ArchivedSession]:
            calls.read_tombstones(database)
            return tombstones

        def create_remote(dsn: str) -> mock.MagicMock:
            calls.create_remote(dsn)
            return remote

        with (
            mock.patch(
                "khala.remote.RemoteStore",
                side_effect=create_remote,
            ) as remote_store,
            mock.patch(
                "khala.archive_convergence.read_local_archive_tombstones",
                side_effect=read_tombstones,
            ) as reader,
        ):
            backend = RemoteEvalBackend(
                "postgresql://example/khala",
                "hybrid",
                self.database,
            )

        remote_store.assert_called_once_with("postgresql://example/khala")
        reader.assert_called_once_with(self.database)
        self.assertEqual(backend.mode, "hybrid")
        self.assertEqual(
            calls.mock_calls,
            [
                mock.call.read_tombstones(self.database),
                mock.call.create_remote("postgresql://example/khala"),
                mock.call.ping(),
                mock.call.archives_converged(tombstones),
            ],
        )

    def test_legacy_duplicate_fails_before_remote_search_without_mutating_sqlite(
        self,
    ) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.executescript(
                """
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    agent TEXT NOT NULL,
                    external_id TEXT
                );
                INSERT INTO sessions(id, agent, external_id) VALUES
                    ('older', 'codex', 'duplicate'),
                    ('newer', 'codex', 'duplicate');
                """
            )
        before = self.database.read_bytes()
        real_connect = sqlite3.connect
        remote = mock.MagicMock()

        with (
            mock.patch("khala.remote.RemoteStore", return_value=remote) as remote_store,
            mock.patch("khala.remote.search_by_mode") as search,
            mock.patch(
                "khala.archive_convergence.sqlite3.connect",
                wraps=real_connect,
            ) as connect,
            self.assertRaisesRegex(RuntimeError, "local archive migration is required"),
        ):
            RemoteEvalBackend(
                "postgresql://example/khala",
                "lexical",
                self.database,
            )

        remote_store.assert_not_called()
        remote.ping.assert_not_called()
        remote.archives_converged.assert_not_called()
        remote.sample_events.assert_not_called()
        search.assert_not_called()
        connect.assert_called_once_with(
            f"{self.database.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        self.assertEqual(self.database.read_bytes(), before)

    def test_remote_archive_mismatch_fails_before_search_or_sampling(self) -> None:
        remote = mock.MagicMock()
        remote.archives_converged.return_value = False
        tombstones: list[ArchivedSession] = [
            {
                "id": "archived-session",
                "archived_at": "2026-01-01T00:00:00Z",
                "duplicate_of": None,
            }
        ]

        with (
            mock.patch("khala.remote.RemoteStore", return_value=remote),
            mock.patch(
                "khala.archive_convergence.read_local_archive_tombstones",
                return_value=tombstones,
            ),
            mock.patch("khala.remote.search_by_mode") as search,
            self.assertRaisesRegex(RuntimeError, "remote archive state is not converged"),
        ):
            RemoteEvalBackend(
                "postgresql://example/khala",
                "semantic",
                self.database,
            )

        remote.ping.assert_called_once_with()
        remote.archives_converged.assert_called_once_with(tombstones)
        remote.sample_events.assert_not_called()
        search.assert_not_called()

    def test_matching_and_empty_tombstones_allow_search_and_sampling(self) -> None:
        cases: list[list[ArchivedSession]] = [
            [],
            [
                {
                    "id": "archived-session",
                    "archived_at": "2026-01-01T00:00:00Z",
                    "duplicate_of": "canonical-session",
                }
            ],
        ]
        for tombstones in cases:
            with self.subTest(tombstones=tombstones):
                remote = mock.MagicMock()
                remote.archives_converged.return_value = True
                remote.sample_events.return_value = [{"id": "sample"}]
                expected_search = [{"id": "match"}]
                with (
                    mock.patch("khala.remote.RemoteStore", return_value=remote),
                    mock.patch(
                        "khala.archive_convergence.read_local_archive_tombstones",
                        return_value=tombstones,
                    ),
                    mock.patch(
                        "khala.remote.search_by_mode",
                        return_value=expected_search,
                    ) as search,
                ):
                    backend = RemoteEvalBackend(
                        "postgresql://example/khala",
                        "hybrid",
                        self.database,
                    )
                    self.assertEqual(backend.search("needle", 5), expected_search)
                    self.assertEqual(backend.sample_events(3), [{"id": "sample"}])

                remote.archives_converged.assert_called_once_with(tombstones)
                search.assert_called_once_with(remote, "needle", "hybrid", 5)
                remote.sample_events.assert_called_once_with(3, mode="hybrid")


if __name__ == "__main__":
    unittest.main()
