from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from khala.evaluation.backends import select_backend, validate_backend_request
from khala.evaluation.baseline import run_baseline
from khala.evaluation.dataset import load_jsonl, write_jsonl
from khala.evaluation.llm import (
    GENERATOR_MODEL,
    JUDGE_MODEL,
    make_judge,
    make_question_generator,
)
from khala.evaluation.models import EvalCase
from khala.evaluation.runner import evaluate_queries

MAX_K = 100
MAX_SAMPLE = 100
MAX_JUDGE_RESULTS = 1_000
MAX_READ_RESULTS = 10_000
MAX_REMOTE_SEMANTIC_QUERIES = 1_000


def add_eval_parser(commands: Any) -> None:
    eval_command = commands.add_parser("eval", help="Evaluate recall without usage writes")
    inputs = eval_command.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path, help="Validated JSONL query/gold dataset")
    inputs.add_argument(
        "--generate",
        action="store_true",
        help="Generate a synthetic dataset from sampled events using an LLM",
    )
    inputs.add_argument(
        "--baseline", action="store_true", help="Run the checked-in deterministic gold gate"
    )
    eval_command.add_argument("--baseline-file", type=Path, help=argparse.SUPPRESS)
    eval_command.add_argument("--backend", choices=("local", "remote"))
    eval_command.add_argument(
        "--mode", choices=("lexical", "semantic", "hybrid")
    )
    eval_command.add_argument("--k", type=int)
    eval_command.add_argument("--sample", type=int)
    eval_command.add_argument("--output", type=Path, help="Write generated JSONL privately")
    eval_command.add_argument(
        "--judge", action="store_true", help="Add strict LLM relevance judging"
    )
    eval_command.add_argument(
        "--read", action="store_true", help="Include top-k content and judge reasons"
    )


def _fingerprint(cases: list[EvalCase]) -> str:
    value = [{"query": case.query, "gold": list(case.gold)} for case in cases]
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_workload(
    *,
    case_count: int,
    k: int,
    judge: bool,
    read: bool,
    backend: str,
    mode: str,
) -> None:
    result_budget = case_count * k
    if judge and result_budget > MAX_JUDGE_RESULTS:
        raise ValueError(
            f"judge workload exceeds {MAX_JUDGE_RESULTS} possible results; "
            "use a smaller dataset or k"
        )
    if read and result_budget > MAX_READ_RESULTS:
        raise ValueError(
            f"read output exceeds {MAX_READ_RESULTS} possible results; "
            "use a smaller dataset or k"
        )
    if (
        backend == "remote"
        and mode in {"semantic", "hybrid"}
        and case_count > MAX_REMOTE_SEMANTIC_QUERIES
    ):
        raise ValueError(
            "remote semantic workload exceeds "
            f"{MAX_REMOTE_SEMANTIC_QUERIES} queries; use a smaller dataset"
        )


def run_eval_command(
    args: Any, *, database: Path, dsn: str | None = None
) -> tuple[dict[str, Any], int]:
    if args.baseline:
        if any(
            value is not None
            for value in (args.backend, args.mode, args.k, args.sample)
        ):
            raise ValueError("baseline backend, mode, k, and sample are fixed by its fixture")
        if args.output or args.judge or args.read:
            raise ValueError("baseline does not accept output, judge, or read options")
        report = run_baseline(args.baseline_file)
        return report, 0 if report["passed"] else 1
    if args.baseline_file:
        raise ValueError("baseline-file requires --baseline")
    k = 5 if args.k is None else args.k
    sample = 20 if args.sample is None else args.sample
    backend_name = args.backend or "local"
    mode = args.mode or "lexical"
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}")
    if not 1 <= sample <= MAX_SAMPLE:
        raise ValueError(f"sample must be between 1 and {MAX_SAMPLE}")
    if args.dataset is not None and args.sample is not None:
        raise ValueError("sample applies only to --generate")
    if args.output and not args.generate:
        raise ValueError("output requires --generate")

    validate_backend_request(dsn=dsn, requested=backend_name, mode=mode)
    cases = load_jsonl(args.dataset.expanduser()) if args.dataset else None
    _validate_workload(
        case_count=len(cases) if cases is not None else sample,
        k=k,
        judge=bool(args.judge),
        read=bool(args.read),
        backend=backend_name,
        mode=mode,
    )
    judge = make_judge() if args.judge else None
    question_generator = make_question_generator() if args.generate else None
    backend = select_backend(
        database=database,
        dsn=dsn,
        requested=backend_name,
        mode=mode,
    )
    try:
        generated = cases is None
        if cases is None:
            if question_generator is None:
                raise RuntimeError("eval generation was not initialized")
            sampled = backend.sample_events(sample)
            if not sampled:
                raise ValueError("no eligible events were available for eval generation")
            cases = []
            for row in sampled:
                event_id = row.get("id")
                content = row.get("content")
                if not isinstance(event_id, str) or not isinstance(content, str):
                    raise TypeError("sampled event id and content must be strings")
                cases.append(EvalCase(question_generator(content), (event_id,)))
            if args.output:
                write_jsonl(args.output.expanduser(), cases)

        report = evaluate_queries(
            cases,
            backend.search,
            k=k,
            judge=judge,
            collect=args.read,
            judge_model=JUDGE_MODEL if judge else None,
        )
        report.update(
            {
                "backend": backend.name,
                "mode": backend.mode,
                "dataset_sha256": _fingerprint(cases),
            }
        )
        if generated:
            report["generator"] = GENERATOR_MODEL
        return report, 0
    finally:
        backend.close()
