from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from khala import Store
from khala.adapters.codex import CodexAdapter, codex_sessions_dir
from khala.sync import sync_session

SESSION_ID = "019fe634-c29f-7dc0-be9d-da1c0c3625ca"
ROLLOUT_STEM = f"rollout-2026-08-11T08-35-58-{SESSION_ID}"


def _message(role: str, text: str, timestamp: str) -> dict[str, Any]:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "timestamp": timestamp,
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": text}],
        },
    }


class CodexHookAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sessions_dir = self.root / "sessions"
        self.rollout_dir = self.sessions_dir / "2026" / "08" / "11"
        self.rollout_dir.mkdir(parents=True)
        self.rollout_path = self.rollout_dir / f"{ROLLOUT_STEM}.jsonl"
        self.adapter = CodexAdapter(self.sessions_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self.rollout_path.write_text(
            "".join(f"{json.dumps(record)}\n" for record in records)
        )

    def _stop_payload(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "hook_event_name": "Stop",
            "session_id": SESSION_ID,
            "transcript_path": str(self.rollout_path),
            "cwd": "/workspace/frogfish",
            "stop_hook_active": False,
            "last_assistant_message": "response",
        }
        payload.update(overrides)
        return payload

    def test_default_sessions_directory_honors_environment_precedence(self) -> None:
        explicit_sessions = self.root / "explicit-sessions"
        custom_home = self.root / "custom-codex-home"

        self.assertEqual(
            codex_sessions_dir(
                {
                    "CODEX_SESSIONS_DIR": str(explicit_sessions),
                    "CODEX_HOME": str(custom_home),
                }
            ),
            explicit_sessions,
        )
        self.assertEqual(
            codex_sessions_dir({"CODEX_HOME": str(custom_home)}),
            custom_home / "sessions",
        )
        self.assertEqual(codex_sessions_dir({}), Path.home() / ".codex" / "sessions")

        with patch.dict(os.environ, {"CODEX_HOME": str(custom_home)}, clear=True):
            self.assertEqual(CodexAdapter().sessions_dir, custom_home / "sessions")

    def test_official_stop_payload_resolves_full_rollout_external_id(self) -> None:
        self._write_records(
            [
                {
                    "timestamp": "2026-08-11T08:35:58Z",
                    "type": "session_meta",
                    "payload": {"id": SESSION_ID, "cwd": "/workspace/frogfish"},
                }
            ]
        )

        session_id, cwd = self.adapter.handle_hook(self._stop_payload())

        self.assertEqual(session_id, ROLLOUT_STEM)
        self.assertEqual(cwd, "/workspace/frogfish")
        self.assertIn(session_id, self.adapter.list_sessions())

    def test_transcript_accepts_legacy_full_stem_claim(self) -> None:
        self._write_records([])

        session_id, _ = self.adapter.handle_hook(
            {
                "rollout_id": ROLLOUT_STEM,
                "transcript_path": str(self.rollout_path),
            }
        )

        self.assertEqual(session_id, ROLLOUT_STEM)

    def test_missing_transcript_path_preserves_legacy_identifiers(self) -> None:
        (self.sessions_dir / "legacy-rollout.jsonl").write_text("")
        (self.sessions_dir / "legacy-id.jsonl").write_text("")
        self.assertEqual(
            self.adapter.handle_hook({"session_id": "legacy-rollout", "cwd": "/ws"}),
            ("legacy-rollout", "/ws"),
        )
        self.assertEqual(
            self.adapter.handle_hook({"rollout_id": "legacy-id", "cwd": "/ws"}),
            ("legacy-id", "/ws"),
        )

    def test_missing_transcript_path_resolves_bare_uuid_without_empty_session(self) -> None:
        self._write_records([])

        self.assertEqual(
            self.adapter.handle_hook({"session_id": SESSION_ID, "cwd": "/ws"}),
            (ROLLOUT_STEM, "/ws"),
        )
        self.assertEqual(
            self.adapter.handle_hook({"session_id": "unknown", "cwd": "/ws"}),
            (None, "/ws"),
        )

    def test_missing_rollout_fails_fast_instead_of_reporting_empty_sync(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Codex rollout not found"):
            self.adapter.read_batch("missing-rollout")

    def test_rejects_malformed_or_out_of_session_transcript_paths(self) -> None:
        self._write_records([])
        outside = self.root / f"{ROLLOUT_STEM}.jsonl"
        outside.write_text("")
        symlink_escape = self.rollout_dir / (
            f"rollout-2026-08-11T09-00-00-{SESSION_ID}.jsonl"
        )
        symlink_escape.symlink_to(outside)
        mismatched = self.rollout_dir / (
            "rollout-2026-08-11T08-35-58-019fe634-c29f-7dc0-be9d-da1c0c3625cb.jsonl"
        )
        mismatched.write_text("")
        wrong_suffix = self.rollout_dir / f"{ROLLOUT_STEM}.txt"
        wrong_suffix.write_text("")

        invalid_paths: tuple[object, ...] = (
            str(outside),
            str(symlink_escape),
            str(mismatched),
            str(wrong_suffix),
            str(self.rollout_dir / f"missing-{SESSION_ID}.jsonl"),
            f"2026/08/11/{ROLLOUT_STEM}.jsonl",
            42,
            "",
        )
        for transcript_path in invalid_paths:
            with self.subTest(transcript_path=transcript_path):
                session_id, cwd = self.adapter.handle_hook(
                    self._stop_payload(transcript_path=transcript_path)
                )
                self.assertIsNone(session_id)
                self.assertEqual(cwd, "/workspace/frogfish")

    def test_rejects_transcript_for_a_different_hook_session(self) -> None:
        self._write_records([])

        session_id, _ = self.adapter.handle_hook(
            self._stop_payload(session_id="019fe634-c29f-7dc0-be9d-da1c0c3625cb")
        )

        self.assertIsNone(session_id)

    def test_rejects_transcript_without_a_valid_claimed_session(self) -> None:
        self._write_records([])

        for payload in (
            {"transcript_path": str(self.rollout_path)},
            {"session_id": "not-a-uuid", "transcript_path": str(self.rollout_path)},
        ):
            with self.subTest(payload=payload):
                session_id, _ = self.adapter.handle_hook(payload)
                self.assertIsNone(session_id)

        self.assertEqual(self.adapter.handle_hook({}), (None, ""))

    def test_ambiguous_uuid_fallback_is_rejected(self) -> None:
        self._write_records([])
        duplicate_dir = self.sessions_dir / "2026" / "08" / "12"
        duplicate_dir.mkdir()
        (duplicate_dir / f"{ROLLOUT_STEM}.jsonl").write_text("")

        session_id, _ = self.adapter.handle_hook({"session_id": SESSION_ID})

        self.assertIsNone(session_id)
        self.assertEqual(
            self.adapter.handle_hook({"session_id": ROLLOUT_STEM}), (None, "")
        )
        transcript_session_id, _ = self.adapter.handle_hook(self._stop_payload())
        self.assertIsNone(transcript_session_id)
        with self.assertRaisesRegex(FileNotFoundError, "ambiguous"):
            self.adapter.read_batch(ROLLOUT_STEM)


class CodexTurnSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sessions_dir = self.root / "sessions"
        rollout_dir = self.sessions_dir / "2026" / "08" / "11"
        rollout_dir.mkdir(parents=True)
        self.rollout_path = rollout_dir / f"{ROLLOUT_STEM}.jsonl"
        self.adapter = CodexAdapter(self.sessions_dir)
        self.store = Store(self.root / "khala.db")
        self.payload = {
            "hook_event_name": "Stop",
            "session_id": SESSION_ID,
            "transcript_path": str(self.rollout_path),
            "cwd": "/workspace/frogfish",
        }

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _append_record(self, record: dict[str, Any], *, complete: bool = True) -> None:
        with self.rollout_path.open("a") as stream:
            stream.write(json.dumps(record))
            if complete:
                stream.write("\n")

    def _sync_hook_session(self) -> dict[str, Any]:
        session_id, _ = self.adapter.handle_hook(self.payload)
        self.assertEqual(session_id, ROLLOUT_STEM)
        assert session_id is not None
        return sync_session(self.store, self.adapter, session_id)

    def test_repeated_stop_hooks_sync_each_complete_turn_exactly_once(self) -> None:
        self._append_record(
            {
                "timestamp": "2026-08-11T08:35:58Z",
                "type": "session_meta",
                "payload": {"id": SESSION_ID, "cwd": "/workspace/frogfish"},
            }
        )
        self._append_record(_message("user", "first prompt", "2026-08-11T08:36:00Z"))
        self._append_record(
            _message("assistant", "first response", "2026-08-11T08:36:01Z")
        )

        first = self._sync_hook_session()
        duplicate_first = self._sync_hook_session()

        self.assertEqual(first["added"], 2)
        self.assertEqual(duplicate_first["added"], 0)

        self._append_record(
            _message("user", "second prompt\nwith detail", "2026-08-11T08:37:00Z")
        )
        self._append_record(
            _message("assistant", "second response\ncomplete", "2026-08-11T08:37:01Z"),
            complete=False,
        )

        partial = self._sync_hook_session()
        repeated_partial = self._sync_hook_session()
        self.assertEqual(partial["added"], 1)
        self.assertEqual(repeated_partial["added"], 0)
        self.assertEqual(self.store.get_watermark("codex", ROLLOUT_STEM), 4)

        with self.rollout_path.open("a") as stream:
            stream.write("\n")

        completed = self._sync_hook_session()
        duplicate_completed = self._sync_hook_session()
        self.assertEqual(completed["added"], 1)
        self.assertEqual(duplicate_completed["added"], 0)
        self.assertEqual(self.store.get_watermark("codex", ROLLOUT_STEM), 5)

        events = self.store.replay(first["khala_session"])
        self.assertEqual(
            [(event["role"], event["content"]) for event in events],
            [
                ("user", "first prompt"),
                ("assistant", "first response"),
                ("user", "second prompt\nwith detail"),
                ("assistant", "second response\ncomplete"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
