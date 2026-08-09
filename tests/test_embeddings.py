import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TEST_API_KEY_ENV = {"OPENROUTER_API_KEY": "sk-test-1234567890abcdef"}


class KhalaEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        import khala.embeddings as emb

        self.emb = emb

    def test_api_key_set_respects_env(self):
        with patch.object(self.emb, "_load_dotenv", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(self.emb.api_key_set())
            with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
                self.assertTrue(self.emb.api_key_set())

    def test_dotenv_supplies_key_when_env_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("OPENROUTER_API_KEY=sk-test-dotenv-1234567890\nOTHER=x\n")
            with patch.dict(os.environ, {}, clear=True):
                self.emb._load_dotenv(env_file)
                self.assertTrue(self.emb.api_key_set())

    def test_env_wins_over_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("OPENROUTER_API_KEY=sk-test-dotenv-1234567890\n")
            with patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "sk-test-env-1234567890"},
                clear=False,
            ):
                self.emb._load_dotenv(env_file)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "sk-test-env-1234567890")

    def test_content_hash_stable(self):
        self.assertEqual(self.emb.content_hash("hello"), self.emb.content_hash("hello"))
        self.assertNotEqual(self.emb.content_hash("hello"), self.emb.content_hash("world"))

    def test_embed_no_key_raises(self):
        with patch.object(self.emb, "_load_dotenv", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(self.emb.EmbeddingError):
                    self.emb.embed(["some text"])

    def test_embed_truncates_overlong_text(self):
        long_text = "x" * (self.emb.MAX_TEXT_LEN * 2)
        sent = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"data": [{"embedding": [0.5] * 1536, "index": 0}]}).encode()

        def fake_open(req, timeout=None):
            body = json.loads(req.data.decode())
            sent["shape"] = len(body["input"][0])
            return Resp()

        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_open):
                out = self.emb.embed([long_text])
        self.assertEqual(out[0][0], 0.5)
        self.assertEqual(sent["shape"], self.emb.MAX_TEXT_LEN)

    def test_embed_empty_inputs(self):
        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=AssertionError("no network call")):
                self.assertEqual(self.emb.embed([]), [])
                self.assertEqual(self.emb.embed([""]), [])

    def test_embed_batch_dims_and_ordering(self):
        payload = {
            "data": [
                {"embedding": [0.1] * 1536, "index": 0},
                {"embedding": [0.2] * 1536, "index": 1},
            ]
        }

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(payload).encode()

        sent = {}

        def fake_open(req, timeout=None):
            sent["timeout"] = timeout
            sent["headers"] = {k: v for k, v in req.headers.items()}
            body = json.loads(req.data.decode())
            sent["body"] = body
            return Resp()

        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_open):
                out = self.emb.embed(["a", "b"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [0.1] * 1536)
        self.assertEqual(out[1], [0.2] * 1536)
        self.assertEqual(sent["body"]["model"], "openai/text-embedding-3-small")
        self.assertEqual(sent["body"]["dimensions"], 1536)
        self.assertNotIn("sk-test", json.dumps(sent["body"]))
        headers_lc = {k.lower(): v for k, v in sent["headers"].items()}
        self.assertEqual(headers_lc.get("x-title"), "khala")
        self.assertEqual(headers_lc.get("http-referer"), "https://github.com/songkg7/khala")
        self.assertTrue(headers_lc["authorization"].startswith("Bearer "))


if __name__ == "__main__":
    unittest.main()
