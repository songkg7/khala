import os
import sqlite3
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from khala import Store, load_adapter, read_jsonl, redact
from khala.identifier_graph import IdentifierGraphQuery


class KhalaStoreTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "khala.db"
        self.store = Store(self.db)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_add_session_and_events(self):
        session = self.store.add_session("codex", "ext-1", {"env": "prod"})
        self.assertIsInstance(session, str)
        self.assertEqual(len(session), 36)  # uuid
        self.store.add_event(session, "user", "hello", metadata={"ref": "1"})
        self.store.add_event(session, "assistant", "world", created_at="2026-01-01T00:00:00Z")
        self.assertEqual(len(self.store.replay(session)), 2)
        self.assertEqual(self.store.status()["sessions"], 1)
        self.assertEqual(self.store.status()["events"], 2)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_database_file_is_private(self):
        self.assertEqual(stat.S_IMODE(self.db.stat().st_mode), 0o600)

    def test_memory_store_does_not_create_filesystem_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = Path.cwd()
            os.chdir(directory)
            try:
                memory_store = Store(Path(":memory:"))
                self.addCleanup(memory_store.close)
                self.assertFalse(Path(":memory:").exists())
            finally:
                os.chdir(previous_cwd)

        self.store.close()
        self.db.chmod(0o644)
        self.store = Store(self.db)

        self.assertEqual(stat.S_IMODE(self.db.stat().st_mode), 0o600)

    def test_add_event_invalid_session(self):
        with self.assertRaises(ValueError, msg="unknown session"):
            self.store.add_event("no-such-session", "user", "x")

    def test_add_event_invalid_role(self):
        session = self.store.add_session("test")
        with self.assertRaises(ValueError, msg="unsupported role"):
            self.store.add_event(session, "invalid", "x")

    def test_add_event_rejects_postgres_incompatible_nul(self):
        session = self.store.add_session("test")
        with self.assertRaisesRegex(ValueError, "NUL"):
            self.store.add_event(session, "user", "valid prefix\x00poison")
        self.assertEqual(self.store.replay(session), [])
        self.assertEqual(self.store.status()["pending_push_total"], 0)

    def test_recall_empty(self):
        session = self.store.add_session("test")
        self.store.add_event(session, "user", "needle in haystack")
        self.assertEqual(self.store.recall("nonexistent"), [])
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0],
            1,
        )
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            self.store.recall("needle", limit=0)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0],
            1,
        )

    def test_recall_limit_and_weighting(self):
        session = self.store.add_session("test")
        e1 = self.store.add_event(session, "user", "alpha beta gamma delta")
        self.store.add_event(session, "assistant", "alpha only")
        self.store.add_event(session, "system", "delta omega")

        r1 = self.store.recall("alpha OR delta", limit=3)
        self.assertEqual(len(r1), 3)

        # Boost e1 by recalling it
        self.store.recall("alpha beta gamma", limit=1)
        r2 = self.store.recall("alpha", limit=3)
        self.assertEqual(r2[0]["recall_count"], 2)
        self.assertEqual(r2[0]["id"], e1)

    def test_recall_hyphen_query_is_tokenized_as_literal_text(self):
        """FTS5 syntax characters in natural-language queries stay literal."""
        session = self.store.add_session("test")
        self.store.add_event(session, "assistant", "cross-agent recall target")
        hits = self.store.recall("cross-agent")
        self.assertTrue(any("cross-agent" in h["content"] for h in hits))

    def test_replay_empty_and_nonexistent(self):
        self.assertEqual(self.store.replay("no-such"), [])
        session = self.store.add_session("test")
        self.assertEqual(self.store.replay(session), [])

    def test_graph_finds_same_session_neighbors(self):
        session = self.store.add_session("test")
        e1 = self.store.add_event(session, "user", "needle in haystack")
        e2 = self.store.add_event(session, "assistant", "response about a thing")
        g = self.store.graph(e1)
        self.assertTrue(g["found"])
        self.assertEqual(g["event"]["id"], e1)
        ids = [n["id"] for n in g["same_session"]]
        self.assertIn(e2, ids)

    def test_graph_missing_event(self):
        g = self.store.graph("no-such-event")
        self.assertFalse(g["found"])

    def test_identifier_graph_indexes_new_events_and_reports_bounded_evidence(self):
        session = self.store.add_session("test")
        self.store.add_event(
            session,
            "user",
            "why change the remote store?",
            created_at="2026-01-01T00:00:00Z",
        )
        first = self.store.add_event(
            session,
            "assistant",
            "changed khala/remote.py with khala/store.py and khala/cli.py",
            created_at="2026-01-01T00:00:01Z",
        )
        self.store.add_event(
            session,
            "user",
            "verify the change",
            created_at="2026-01-01T00:00:02Z",
        )
        second = self.store.add_event(
            session,
            "assistant",
            "rechecked khala/remote.py:42",
            created_at="2026-01-01T00:00:03Z",
        )

        report = self.store.identifier_graph(
            IdentifierGraphQuery("./khala\\remote.py", limit=1, context_limit=1)
        )

        self.assertTrue(report["found"])
        self.assertEqual(report["resolution"]["status"], "exact")
        self.assertEqual(report["evidence"][0]["event_id"], second)
        self.assertEqual(len(report["evidence"][0]["same_session"]), 1)
        self.assertTrue(report["coverage"]["has_more"])
        self.assertTrue(report["coverage"]["evidence_has_more"])
        self.assertTrue(report["coverage"]["related_has_more"])
        self.assertFalse(report["coverage"]["complete"])
        self.assertEqual(report["coverage"]["indexed_active_events"], 4)
        self.assertEqual(
            [(row["value"], row["co_referenced_events"]) for row in report["related"]],
            [("khala/cli.py", 1)],
        )
        self.assertNotIn("depends_on", repr(report))
        self.assertNotEqual(first, second)

    def test_identifier_index_failure_rolls_back_event_and_outbox(self):
        session = self.store.add_session("test")
        self.store.db.execute(
            """
            CREATE TRIGGER reject_identifier_state
            BEFORE INSERT ON identifier_index_state BEGIN
                SELECT RAISE(ABORT, 'identifier indexing failed');
            END
            """
        )
        self.store.db.commit()

        with self.assertRaisesRegex(ValueError, "event insert failed") as caught:
            self.store.add_event(session, "assistant", "changed khala/store.py")

        self.assertIsInstance(caught.exception.__cause__, sqlite3.IntegrityError)
        self.assertEqual(self.store.replay(session), [])
        self.assertEqual(self.store.status()["pending_push_total"], 0)

    def test_identifier_backfill_is_explicit_and_bounded(self):
        session = self.store.add_session("legacy")
        self.store.db.execute(
            """
            INSERT INTO events(id, session_id, role, content, created_at, metadata)
            VALUES ('legacy-file-event', ?, 'assistant', 'read legacy.py',
                    '2026-01-01T00:00:00Z', '{}')
            """,
            (session,),
        )
        self.store.db.commit()

        before = self.store.identifier_graph(IdentifierGraphQuery("legacy.py"))
        self.assertFalse(before["found"])
        self.assertEqual(before["coverage"]["pending_active_events"], 1)

        self.assertEqual(self.store.backfill_identifiers(limit=1), 1)
        after = self.store.identifier_graph(IdentifierGraphQuery("legacy.py"))

        self.assertTrue(after["found"])
        self.assertEqual(after["coverage"]["pending_active_events"], 0)
        self.assertEqual(self.store.backfill_identifiers(limit=1), 0)

    def test_graph_and_attach_bound_neighbor_content(self):
        session = self.store.add_session("test")
        hit_id = self.store.add_event(session, "user", "h" * 600)
        self.store.add_event(session, "assistant", "n" * 600)

        graph = self.store.graph(hit_id, limit=1)
        attached = self.store.attach_neighbors([{"id": hit_id, "session_id": session}], per_hit=1)

        self.assertEqual(len(graph["event"]["content"]), 200)
        self.assertEqual(len(graph["same_session"][0]["content"]), 500)
        self.assertEqual(len(attached[0]["adjacent"][0]["content"]), 500)

    def test_attach_neighbors_excludes_hit_and_centers_window(self):
        session = self.store.add_session("test")
        self.store.add_event(session, "system", "before-a")
        self.store.add_event(session, "user", "before-b")
        hit_id = self.store.add_event(session, "user", "alpha")
        self.store.add_event(session, "assistant", "after-a")
        self.store.add_event(session, "assistant", "after-b")
        hits = [{"id": hit_id, "session_id": session, "role": "user", "content": "alpha"}]

        # Contiguous window centered on the hit, hit itself excluded.
        adj = self.store.attach_neighbors(hits)[0]["adjacent"]
        self.assertEqual([a["content"] for a in adj], ["before-b", "after-a", "after-b"])
        self.assertNotIn("alpha", [a["content"] for a in adj])

        # A larger window still excludes the hit.
        wide = self.store.attach_neighbors(hits, per_hit=4)[0]["adjacent"]
        self.assertNotIn("alpha", [a["content"] for a in wide])
        self.assertEqual(len(wide), 4)

    def test_attach_neighbors_handles_missing_session(self):
        hits = [{"id": "x", "role": "user", "content": "alpha"}]
        self.assertEqual(self.store.attach_neighbors(hits)[0]["adjacent"], [])

    def test_graph_and_attach_use_canonical_order_with_interleaved_rowids(self):
        target_session = self.store.add_session("target")
        other_session = self.store.add_session("other")
        rows = [
            ("target-z", target_session, "assistant", "tie-after", "2026-01-01T00:00:02Z"),
            ("other-1", other_session, "user", "other one", "2026-01-01T00:00:00Z"),
            ("target-before", target_session, "user", "before", "2026-01-01T00:00:01Z"),
            ("target-after", target_session, "assistant", "after", "2026-01-01T00:00:03Z"),
            ("other-2", other_session, "user", "other two", "2026-01-01T00:00:01Z"),
            ("target-m", target_session, "user", "hit", "2026-01-01T00:00:02Z"),
            ("other-3", other_session, "user", "other three", "2026-01-01T00:00:02Z"),
            ("target-a", target_session, "assistant", "tie-before", "2026-01-01T00:00:02Z"),
        ]
        self.store.db.executemany(
            """
            INSERT INTO events(id, session_id, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, '{}')
            """,
            rows,
        )
        self.store.db.commit()

        graph = self.store.graph("target-m", limit=4)
        attached = self.store.attach_neighbors(
            [{"id": "target-m", "session_id": target_session}], per_hit=4
        )

        expected_ids = ["target-before", "target-a", "target-z", "target-after"]
        expected_content = ["before", "tie-before", "tie-after", "after"]
        self.assertEqual([row["id"] for row in graph["same_session"]], expected_ids)
        self.assertEqual([row["content"] for row in attached[0]["adjacent"]], expected_content)

    def test_adjacency_keyset_queries_use_composite_index(self):
        session = self.store.add_session("test")
        event_ids = [
            self.store.add_event(
                session,
                "user",
                f"event-{index}",
                created_at=f"2026-01-01T00:00:0{index}Z",
            )
            for index in range(3)
        ]
        statements: list[str] = []
        self.store.db.set_trace_callback(statements.append)
        try:
            self.store.graph(event_ids[1], limit=2)
        finally:
            self.store.db.set_trace_callback(None)

        keyset_queries = [
            statement for statement in statements if "(event.created_at, event.id)" in statement
        ]
        self.assertEqual(len(keyset_queries), 2)
        self.assertEqual(sum(" < " in query for query in keyset_queries), 1)
        self.assertEqual(sum(" > " in query for query in keyset_queries), 1)
        for query in keyset_queries:
            with self.subTest(query=query):
                plan = " ".join(
                    str(row[3])
                    for row in self.store.db.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
                )
                self.assertIn("USING INDEX events_session_created_id", plan)
                self.assertIn("created_at", plan)

    def test_graph_and_attach_backfill_boundaries_to_exact_limit(self):
        session = self.store.add_session("test")
        event_ids = [
            self.store.add_event(
                session,
                "user",
                f"event-{index}",
                created_at=f"2026-01-01T00:00:0{index}Z",
            )
            for index in range(5)
        ]
        content_by_id = {event_id: f"event-{index}" for index, event_id in enumerate(event_ids)}
        cases = (
            (event_ids[0], event_ids[1:4]),
            (event_ids[-1], event_ids[1:4]),
        )

        for hit_id, expected_ids in cases:
            with self.subTest(hit_id=hit_id):
                graph = self.store.graph(hit_id, limit=3)
                attached = self.store.attach_neighbors(
                    [{"id": hit_id, "session_id": session}], per_hit=3
                )
                graph_ids = [row["id"] for row in graph["same_session"]]
                attached_content = [row["content"] for row in attached[0]["adjacent"]]

                self.assertEqual(graph_ids, expected_ids)
                self.assertEqual(len(attached_content), 3)
                self.assertEqual(
                    attached_content,
                    [content_by_id[event_id] for event_id in expected_ids],
                )

    def test_attach_neighbors_rejects_spoofed_session_id(self):
        actual_session = self.store.add_session("actual")
        spoofed_session = self.store.add_session("spoofed")
        hit_id = self.store.add_event(actual_session, "user", "private hit")
        self.store.add_event(spoofed_session, "user", "spoofed before")
        self.store.add_event(spoofed_session, "assistant", "spoofed after")

        attached = self.store.attach_neighbors(
            [{"id": hit_id, "session_id": spoofed_session}], per_hit=3
        )

        self.assertEqual(attached[0]["adjacent"], [])

    def test_graph_and_attach_exclude_archived_sessions(self):
        session = self.store.add_session("archived")
        hit_id = self.store.add_event(session, "user", "private hit")
        self.store.add_event(session, "assistant", "private neighbor")
        self.store.db.execute(
            "UPDATE sessions SET archived_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00Z", session),
        )
        self.store.db.commit()

        graph = self.store.graph(hit_id)
        attached = self.store.attach_neighbors([{"id": hit_id, "session_id": session}])

        self.assertFalse(graph["found"])
        self.assertEqual(attached[0]["adjacent"], [])

    def test_archived_sessions_returns_sorted_content_free_tombstones(self):
        active = self.store.add_session("active")
        duplicate = self.store.add_session("duplicate")
        archived = self.store.add_session("archived")
        self.store.db.executemany(
            """
            UPDATE sessions
            SET archived_at = ?, duplicate_of = ?
            WHERE id = ?
            """,
            [
                ("2026-01-02T00:00:00+00:00", active, duplicate),
                ("2026-01-01T00:00:00+00:00", None, archived),
            ],
        )
        self.store.db.commit()

        tombstones = self.store.archived_sessions()

        self.assertEqual(
            tombstones,
            sorted(
                [
                    {
                        "id": duplicate,
                        "archived_at": "2026-01-02T00:00:00+00:00",
                        "duplicate_of": active,
                    },
                    {
                        "id": archived,
                        "archived_at": "2026-01-01T00:00:00+00:00",
                        "duplicate_of": None,
                    },
                ],
                key=lambda row: row["id"],
            ),
        )
        self.assertTrue(
            all(set(row) == {"id", "archived_at", "duplicate_of"} for row in tombstones)
        )

    def test_graph_and_attach_reject_nonpositive_limits_before_lookup(self):
        for limit in (0, -1):
            with self.subTest(limit=limit, operation="graph"):
                with self.assertRaises(ValueError):
                    self.store.graph("no-such-event", limit=limit)
            with self.subTest(limit=limit, operation="attach"):
                with self.assertRaises(ValueError):
                    self.store.attach_neighbors([], per_hit=limit)

    def test_import_events_valid(self):
        events = [
            {"role": "user", "content": "first", "created_at": "1", "metadata": {"k": "v"}},
            {"role": "assistant", "content": "second", "created_at": "2"},
            {"role": "tool", "content": "result"},
        ]
        session = self.store.import_events("hermes", events)
        replay = self.store.replay(session)
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay[0]["role"], "user")
        self.assertEqual(replay[0]["content"], "first")
        self.assertEqual(self.store.status()["sessions"], 1)

    def test_import_events_atomic_rollback(self):
        with self.assertRaises(ValueError, msg="unsupported role"):
            self.store.import_events(
                "broken",
                [
                    {"role": "user", "content": "ok"},
                    {"role": "invalid", "content": "bad"},
                ],
            )
        status = self.store.status()
        self.assertEqual(status["sessions"], 0)
        self.assertEqual(status["events"], 0)
        self.assertEqual(status["pending_push"], 0)

    def test_import_empty_events(self):
        session = self.store.import_events("test", [])
        self.assertEqual(self.store.replay(session), [])
        self.assertEqual(self.store.status()["sessions"], 1)

    def test_nested_transaction_rolls_back_inner_without_committing_outer(self):
        with self.assertRaisesRegex(RuntimeError, "outer failure"):
            with self.store.transaction():
                first = self.store.add_session("outer")
                with self.assertRaisesRegex(ValueError, "unsupported role"):
                    self.store.add_event(first, "invalid", "bad")
                second = self.store.add_session("outer")
                self.assertNotEqual(first, second)
                self.store.enqueue_backlog()
                raise RuntimeError("outer failure")

        self.assertEqual(self.store.status()["sessions"], 0)

    def test_external_session_identity_is_unique(self):
        self.store.add_session("agent", "external")
        with self.assertRaisesRegex(ValueError, "session already exists"):
            self.store.add_session("agent", "external")

    def test_null_external_session_identity_can_repeat(self):
        first = self.store.add_session("agent")
        second = self.store.add_session("agent")
        self.assertNotEqual(first, second)

    def test_legacy_duplicate_sessions_are_quarantined_without_event_loss(self):
        legacy_path = Path(self.tmp.name) / "duplicate-sessions.db"
        db = sqlite3.connect(legacy_path)
        db.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                user_name TEXT,
                user_email TEXT
            );
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO sessions VALUES
                ('older', 'agent', 'external', '1', '{}', NULL, NULL),
                ('newer', 'agent', 'external', '2', '{}', 'Alice', 'a@example.com');
            INSERT INTO events VALUES
                ('event-1', 'older', 'user', 'one', '1', '{}'),
                ('event-2', 'newer', 'assistant', 'two', '2', '{}');
            """
        )
        db.commit()
        db.close()

        migrated = Store(legacy_path)
        try:
            sessions = migrated.db.execute(
                """
                SELECT id, user_name, user_email, archived_at, duplicate_of
                FROM sessions ORDER BY id
                """
            ).fetchall()
            event_sessions = migrated.db.execute(
                "SELECT id, session_id FROM events ORDER BY id"
            ).fetchall()
            self.assertEqual(len(sessions), 2)
            active = next(row for row in sessions if row["archived_at"] is None)
            archived = next(row for row in sessions if row["archived_at"] is not None)
            self.assertEqual(str(active["id"]), "older")
            self.assertEqual(str(active["user_name"]), "Alice")
            self.assertEqual(str(archived["id"]), "newer")
            self.assertEqual(str(archived["duplicate_of"]), "older")
            self.assertEqual(
                [(str(row["id"]), str(row["session_id"])) for row in event_sessions],
                [("event-1", "older"), ("event-2", "newer")],
            )
            status = migrated.status()
            self.assertEqual(status["sessions"], 1)
            self.assertEqual(status["events"], 1)
            self.assertEqual(status["archived_sessions"], 1)
            self.assertEqual(status["archived_events"], 1)
            migrated.enqueue_backlog()
            self.assertEqual(migrated.pending_remote(), ["event-1"])
            with self.assertRaisesRegex(ValueError, "archived session"):
                migrated.add_event("newer", "user", "must stay quarantined")
            self.assertEqual(migrated.pending_remote(), ["event-1"])
        finally:
            migrated.close()

    def test_concurrent_store_open_serializes_forward_migrations(self):
        legacy_path = Path(self.tmp.name) / "concurrent-migration.db"
        db = sqlite3.connect(legacy_path)
        db.execute(
            """
            CREATE TABLE remote_outbox (
                event_id TEXT PRIMARY KEY,
                queued_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                last_error TEXT
            )
            """
        )
        db.commit()
        db.close()
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def open_store() -> None:
            try:
                barrier.wait(timeout=5)
                opened = Store(legacy_path)
                opened.close()
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=open_store) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        migrated = Store(legacy_path)
        try:
            columns = {
                str(row[1]) for row in migrated.db.execute("PRAGMA table_info(remote_outbox)")
            }
            self.assertTrue({"last_attempt_at", "dead_lettered_at", "lease_owner"} <= columns)
            self.assertIsNotNone(
                migrated.db.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'sync_event_ledger'
                    """
                ).fetchone()
            )
        finally:
            migrated.close()

    def test_khalai_status_does_not_claim_remote_embedding_count(self):
        self.assertIsNone(self.store.status(mode="khalai")["pending_embedding"])

    def test_status_empty(self):
        status = self.store.status()
        self.assertEqual(status["sessions"], 0)
        self.assertEqual(status["events"], 0)
        self.assertEqual(status["pending_push"], 0)


class KhalaRedactTest(unittest.TestCase):
    def test_empty_and_clean(self):
        self.assertEqual(redact(""), "")
        self.assertEqual(redact("hello world"), "hello world")

    def test_keys_and_tokens(self):
        self.assertEqual(redact("api_key=sk-abc123"), "api_key=[REDACTED]")
        self.assertEqual(redact("password= hunter2"), "password=[REDACTED]")
        self.assertEqual(redact("secret=ghp_abcdefghijklmnopqrstuvxyz"), "secret=[REDACTED]")
        result = redact("Authorization=Bearer abcdefghijklmnop1234")
        self.assertIn("[REDACTED]", result)
        self.assertNotIn("Bearer", result)

    def test_github_token(self):
        self.assertEqual(redact("github_pat_abcABC123_xyzABC123"), "[REDACTED]")
        self.assertEqual(redact("ghp_abcdEFGH1234_5678abcd"), "[REDACTED]")

    def test_multiple_occurrences(self):
        text = "api_key=secret password=pass123 Bearer token123"
        result = redact(text)
        self.assertNotIn("secret", result)
        self.assertNotIn("pass123", result)
        self.assertNotIn("token123", result)
        self.assertIn("api_key=[REDACTED]", result)
        self.assertIn("password=[REDACTED]", result)

    def test_no_false_positive_on_url(self):
        self.assertEqual(
            redact("https://example.com/api_key=test"), redact("https://example.com/api_key=test")
        )

    def test_uri_password(self):
        result = redact("failed postgresql://alice:uri-secret@db.example/khala")
        self.assertNotIn("uri-secret", result)
        self.assertIn("alice:[REDACTED]@", result)


class KhalaAdapterTest(unittest.TestCase):
    def test_load_adapter_invalid_spec(self):
        with self.assertRaises(ValueError, msg="adapter must be MODULE:CALLABLE"):
            load_adapter("justmodule")

    def test_load_adapter_nonexistent_module(self):
        with self.assertRaises(ImportError):
            load_adapter("nonexistent_module:nonexistent")

    def test_load_adapter_not_callable(self):
        with self.assertRaises(ValueError, msg="is not callable"):
            load_adapter("json:__name__")

    def test_load_adapter_valid(self):
        adapter = load_adapter("json:loads")
        self.assertTrue(callable(adapter))

    def test_read_jsonl(self):
        lines = ['{"role": "user", "content": "hi"}', '{"role": "assistant", "content": "hello"}']
        result = list(read_jsonl(lines))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")

    def test_read_jsonl_skip_blank(self):
        lines = [
            '{"role": "user", "content": "hi"}',
            "",
            "  ",
            '{"role": "user", "content": "bye"}',
        ]
        result = list(read_jsonl(lines))
        self.assertEqual(len(result), 2)

    def test_read_jsonl_not_dict(self):
        with self.assertRaises(ValueError, msg="is not an object"):
            list(read_jsonl(['"just a string"']))

    def test_read_jsonl_invalid_json(self):
        import json as _json

        with self.assertRaises(_json.JSONDecodeError):
            list(read_jsonl(["not json"]))


if __name__ == "__main__":
    unittest.main()
