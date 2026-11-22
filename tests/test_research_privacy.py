import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


class ResearchPrivacyTest(unittest.TestCase):
    def test_overlap_report_omits_actor_identifiers(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_overlap.py"
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "session.db"
            out = Path(tmp) / "overlap_stats.json"
            with closing(sqlite3.connect(db)) as conn:
                conn.executescript("""
                    CREATE TABLE sessions (
                        id, agent, user_name, user_email, archived_at, duplicate_of
                    );
                    CREATE TABLE events (id, session_id, role, content, created_at);
                    INSERT INTO sessions VALUES (
                        1, 'test', 'Private Name', 'actor@example.test', NULL, NULL
                    );
                    INSERT INTO events VALUES
                        (1, 1, 'user', 'private sample prompt', '2026-01-01'),
                        (2, 1, 'user', 'private sample prompt', '2026-01-01');
                """)
            run = subprocess.run(
                [sys.executable, str(script), "--db", str(db), "--out", str(out)],
                capture_output=True,
                text=True,
                check=True,
            )
            saved = out.read_text()
        report = json.loads(saved)
        self.assertEqual(report, json.loads(run.stdout))
        self.assertEqual(report["scope"]["actor_count"], 1)
        for private in (
            "top_actors", "actor@example.test", "Private Name", "private sample prompt"
        ):
            self.assertNotIn(private, saved)
