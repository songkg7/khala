import unittest

from khala.recall_results import RecallHit, deduplicate_recall_results, is_tool_call_summary


def hit(**fields: str) -> RecallHit:
    return RecallHit(**fields)  # type: ignore[arg-type]


class RecallResultsTest(unittest.TestCase):
    def test_tool_call_summary_prefix_is_detected(self):
        self.assertTrue(is_tool_call_summary("[tool_calls: terminal]"))
        self.assertFalse(is_tool_call_summary("terminal"))
        self.assertFalse(is_tool_call_summary(""))
        self.assertFalse(is_tool_call_summary(None))
        self.assertFalse(is_tool_call_summary(42))

    def test_deduplicates_by_role_and_content_keeping_first_ranked(self):
        rows = [
            hit(id="first", role="assistant", content="same"),
            hit(id="second", role="assistant", content="same"),
            hit(id="user-copy", role="user", content="same"),
            hit(id="unique", role="assistant", content="unique"),
        ]

        result = deduplicate_recall_results(rows, limit=10)

        self.assertEqual([row.get("id") for row in result], ["first", "user-copy", "unique"])

    def test_honors_the_limit_after_deduplication(self):
        rows = [
            hit(id="copy-1", role="assistant", content="same"),
            hit(id="copy-2", role="assistant", content="same"),
            hit(id="copy-3", role="assistant", content="same"),
            hit(id="other", role="assistant", content="other"),
        ]

        self.assertEqual(
            [row.get("id") for row in deduplicate_recall_results(rows, limit=1)],
            ["copy-1"],
        )
        self.assertEqual(
            [row.get("id") for row in deduplicate_recall_results(rows, limit=2)],
            ["copy-1", "other"],
        )

    def test_incomplete_rows_are_never_collapsed_together(self):
        rows = [
            hit(id="only-id-1"),
            hit(id="only-id-2"),
            hit(id="full", role="assistant", content="x"),
        ]

        result = deduplicate_recall_results(rows, limit=10)

        self.assertEqual([row.get("id") for row in result], ["only-id-1", "only-id-2", "full"])


if __name__ == "__main__":
    unittest.main()
