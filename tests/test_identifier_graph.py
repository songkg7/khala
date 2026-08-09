from __future__ import annotations

import unittest

from khala.identifier_graph import (
    GraphCoverage,
    GraphEvidence,
    GraphRelated,
    GraphResolution,
    IdentifierGraphQuery,
    assemble_identifier_graph,
)


class IdentifierGraphDomainTest(unittest.TestCase):
    def test_report_exposes_evidence_and_co_reference_without_dependency_claim(self) -> None:
        query = IdentifierGraphQuery("./khala\\remote.py:42", limit=2, context_limit=1)
        report = assemble_identifier_graph(
            "local",
            query,
            resolution=GraphResolution("exact", "khala/remote.py"),
            evidence=(
                GraphEvidence(
                    "event-1",
                    "session-1",
                    "assistant",
                    "touched khala/remote.py",
                    "2026-08-09T00:00:00Z",
                    2,
                ),
            ),
            related=(GraphRelated("tests/test_remote.py", "test_remote.py", 1),),
            coverage=GraphCoverage(3, 0, 0),
            evidence_has_more=False,
            related_has_more=False,
            contexts={"event-1": ({"id": "before", "role": "user", "content": "context"},)},
        )

        self.assertEqual(report["query"]["value"], "khala/remote.py")
        self.assertTrue(report["found"])
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["evidence"][0]["edge_kind"], "references")
        self.assertEqual(report["related"][0]["edge_kind"], "co_referenced")
        self.assertNotIn("depends_on", repr(report))

    def test_ambiguous_basename_is_explicit_and_has_no_evidence(self) -> None:
        report = assemble_identifier_graph(
            "remote",
            IdentifierGraphQuery("remote.py"),
            resolution=GraphResolution(
                "ambiguous",
                candidates=("khala/remote.py", "tests/remote.py"),
            ),
            evidence=(),
            related=(),
            coverage=GraphCoverage(2, 1, 1),
            evidence_has_more=False,
            related_has_more=False,
        )

        self.assertFalse(report["found"])
        self.assertTrue(report["ambiguous"])
        self.assertFalse(report["coverage"]["complete"])

    def test_result_truncation_is_explicit_and_marks_report_incomplete(self) -> None:
        report = assemble_identifier_graph(
            "local",
            IdentifierGraphQuery("a.py", limit=1, context_limit=0),
            resolution=GraphResolution("exact", "a.py"),
            evidence=(GraphEvidence("e", "s", "user", "a.py", "now", 1),),
            related=(GraphRelated("b.py", "b.py", 1),),
            coverage=GraphCoverage(2, 0, 0),
            evidence_has_more=True,
            related_has_more=True,
        )

        self.assertFalse(report["coverage"]["complete"])
        self.assertTrue(report["coverage"]["has_more"])
        self.assertTrue(report["coverage"]["evidence_has_more"])
        self.assertTrue(report["coverage"]["related_has_more"])

    def test_query_and_repository_contracts_fail_fast(self) -> None:
        for query in (
            lambda: IdentifierGraphQuery("../secret.py"),
            lambda: IdentifierGraphQuery("a.py", limit=0),
            lambda: IdentifierGraphQuery("a.py", context_limit=21),
        ):
            with self.subTest(query=query), self.assertRaises((TypeError, ValueError)):
                query()

        base = {
            "backend": "local",
            "query": IdentifierGraphQuery("a.py", limit=1),
            "resolution": GraphResolution("not_found"),
            "evidence": (),
            "related": (),
            "coverage": GraphCoverage(0, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        invalid = (
            {"resolution": GraphResolution("ambiguous", candidates=("a.py",))},
            {"resolution": GraphResolution("exact", "b.py")},
            {"coverage": GraphCoverage(0, 0, 1)},
            {
                "resolution": GraphResolution("exact", "a.py"),
                "evidence": (
                    GraphEvidence("e", "s", "user", "x", "now", 1),
                    GraphEvidence("e2", "s", "user", "x", "now", 1),
                ),
            },
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                assemble_identifier_graph(**{**base, **override})  # type: ignore[arg-type]

    def test_cross_row_contracts_fail_closed(self) -> None:
        evidence = GraphEvidence("e", "s", "user", "a.py", "2026-08-09", 1)
        base = {
            "backend": "local",
            "query": IdentifierGraphQuery("a.py", limit=2, context_limit=1),
            "resolution": GraphResolution("exact", "a.py"),
            "evidence": (evidence,),
            "related": (),
            "coverage": GraphCoverage(1, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        invalid = (
            {"evidence": ()},
            {"evidence_has_more": 1},
            {"evidence_has_more": True},
            {
                "query": IdentifierGraphQuery("a.py", limit=1, context_limit=1),
                "evidence_has_more": True,
            },
            {"related_has_more": 1},
            {"related_has_more": True},
            {
                "evidence": (
                    GraphEvidence("older", "s", "user", "x", "2026-08-08", 1),
                    evidence,
                ),
                "coverage": GraphCoverage(2, 0, 0),
            },
            {
                "evidence": (evidence, evidence),
                "coverage": GraphCoverage(2, 0, 0),
            },
            {"related": (GraphRelated("a.py", "a.py", 1),)},
            {"related": (GraphRelated("b.py", "b.py", 2),)},
            {
                "coverage": GraphCoverage(3, 0, 0),
                "related": (GraphRelated("b.py", "b.py", 2),),
            },
            {
                "related": (
                    GraphRelated("c.py", "c.py", 1),
                    GraphRelated("b.py", "b.py", 2),
                )
            },
            {"contexts": {"unknown": ()}},
            {"contexts": {"e": ({"id": "e", "role": "user", "content": "x"},)}},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                assemble_identifier_graph(**{**base, **override})  # type: ignore[arg-type]

    def test_resolution_basename_and_candidate_contracts_fail_closed(self) -> None:
        base = {
            "backend": "remote",
            "query": IdentifierGraphQuery("a.py"),
            "evidence": (),
            "related": (),
            "coverage": GraphCoverage(1, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        invalid = (
            GraphResolution("basename", "directory/b.py"),
            GraphResolution("basename", "./directory/a.py"),
            GraphResolution("ambiguous", candidates=("directory/a.py", "directory/b.py")),
            GraphResolution("ambiguous", candidates=("./directory/a.py", "other/a.py")),
            GraphResolution(
                "ambiguous",
                candidates=("directory/a.py", "other/a.py"),
                candidates_truncated=1,  # type: ignore[arg-type]
            ),
        )
        for resolution in invalid:
            with self.subTest(resolution=resolution), self.assertRaises((TypeError, ValueError)):
                assemble_identifier_graph(  # type: ignore[arg-type]
                    **{**base, "resolution": resolution}
                )

    def test_assembler_types_and_unresolved_contracts_fail_fast(self) -> None:
        missing = {
            "backend": "local",
            "query": IdentifierGraphQuery("a.py"),
            "resolution": GraphResolution("not_found"),
            "evidence": (),
            "related": (),
            "coverage": GraphCoverage(0, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        evidence = GraphEvidence("e", "s", "user", "a.py", "now", 1)
        invalid = (
            {"backend": "cloud"},
            {"query": object()},
            {"resolution": object()},
            {"coverage": object()},
            {"evidence": (evidence,)},
            {"resolution": GraphResolution("ambiguous", candidates=("one/a.py", "two/a.py"))},
            {"contexts": []},
            {
                "resolution": GraphResolution("exact", "a.py"),
                "evidence": (object(),),
                "coverage": GraphCoverage(1, 0, 0),
            },
            {
                "resolution": GraphResolution("exact", "a.py"),
                "evidence": (evidence,),
                "related": (object(),),
                "coverage": GraphCoverage(1, 0, 0),
            },
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                assemble_identifier_graph(  # type: ignore[arg-type]
                    **{**missing, **override}
                )

    def test_resolution_shape_contracts_fail_fast(self) -> None:
        base = {
            "backend": "local",
            "query": IdentifierGraphQuery("a.py"),
            "evidence": (),
            "related": (),
            "coverage": GraphCoverage(2, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        too_many = tuple(f"dir-{index:03d}/a.py" for index in range(101))
        invalid = (
            GraphResolution("unsupported"),  # type: ignore[arg-type]
            GraphResolution("ambiguous", candidates=("two/a.py", "one/a.py")),
            GraphResolution("ambiguous", candidates=("one/a.py", "one/a.py")),
            GraphResolution("ambiguous", candidates=too_many),
            GraphResolution("exact"),
            GraphResolution("exact", "a.py", ("one/a.py",)),
            GraphResolution("not_found", "a.py"),
            GraphResolution(
                "ambiguous",
                candidates=("one/a.py", "two/a.py"),
                candidates_truncated=True,
            ),
            GraphResolution("not_found", candidates=("one/a.py",)),
        )
        for resolution in invalid:
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                assemble_identifier_graph(  # type: ignore[arg-type]
                    **{**base, "resolution": resolution}
                )

    def test_evidence_and_context_row_contracts_fail_fast(self) -> None:
        valid = GraphEvidence("e", "s", "user", "a.py", "now", 1)
        base = {
            "backend": "remote",
            "query": IdentifierGraphQuery("a.py", limit=2, context_limit=2),
            "resolution": GraphResolution("exact", "a.py"),
            "evidence": (valid,),
            "related": (),
            "coverage": GraphCoverage(2, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        invalid = (
            {"evidence": (GraphEvidence("", "s", "user", "x", "now", 1),)},
            {
                "evidence": (
                    GraphEvidence(1, "s", "user", "x", "now", 1),  # type: ignore[arg-type]
                )
            },
            {
                "evidence": (
                    GraphEvidence("e", "s", "user", object(), "now", 1),  # type: ignore[arg-type]
                )
            },
            {"evidence": (GraphEvidence("e", "s", "user", "x" * 501, "now", 1),)},
            {"evidence": (GraphEvidence("e", "s", "user", "x", "now", 0),)},
            {
                "query": IdentifierGraphQuery("a.py", context_limit=1),
                "contexts": {
                    "e": (
                        {"id": "before", "role": "user", "content": "x"},
                        {"id": "after", "role": "assistant", "content": "x"},
                    )
                },
            },
            {"contexts": {"e": (object(),)}},
            {"contexts": {"e": ({"id": "", "role": "user", "content": "x"},)}},
            {"contexts": {"e": ({"id": "e", "role": "user", "content": "x"},)}},
            {
                "contexts": {
                    "e": (
                        {"id": "same", "role": "user", "content": "x"},
                        {"id": "same", "role": "user", "content": "x"},
                    )
                }
            },
            {"contexts": {"e": ({"id": "x", "role": "", "content": "x"},)}},
            {"contexts": {"e": ({"id": "x", "role": "user", "content": object()},)}},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                assemble_identifier_graph(  # type: ignore[arg-type]
                    **{**base, **override}
                )

    def test_related_row_contracts_fail_fast(self) -> None:
        evidence = GraphEvidence("e", "s", "user", "a.py", "now", 1)
        base = {
            "backend": "local",
            "query": IdentifierGraphQuery("a.py", limit=2),
            "resolution": GraphResolution("exact", "a.py"),
            "evidence": (evidence,),
            "coverage": GraphCoverage(1, 0, 0),
            "evidence_has_more": False,
            "related_has_more": False,
        }
        invalid = (
            (GraphRelated("./b.py", "b.py", 1),),
            (GraphRelated("b.py", "wrong.py", 1),),
            (GraphRelated("b.py", "b.py", 0),),
            (
                GraphRelated("b.py", "b.py", 1),
                GraphRelated("b.py", "b.py", 1),
            ),
        )
        for related in invalid:
            with self.subTest(related=related), self.assertRaises(ValueError):
                assemble_identifier_graph(**{**base, "related": related})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
