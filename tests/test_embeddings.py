import json
import os
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

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

    def _response(self, payload):
        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(payload).encode()

        return Resp()

    def test_embed_rejects_oversized_text_without_truncating_or_sending(self):
        private_content = "private:" + ("x" * self.emb.MAX_TEXT_UTF8_BYTES)

        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch(
                "urllib.request.urlopen", side_effect=AssertionError("no network call")
            ):
                with self.assertRaises(self.emb.EmbeddingError) as raised:
                    self.emb.embed([private_content])
        self.assertNotIn(private_content, str(raised.exception))

    def test_embed_input_limit_is_utf8_bytes(self):
        exact_limit = "가" * (self.emb.MAX_TEXT_UTF8_BYTES // 3)
        response = self._response(
            {"data": [{"embedding": [0.5] * self.emb.DIMENSIONS, "index": 0}]}
        )

        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", return_value=response) as request:
                self.assertEqual(self.emb.embed([exact_limit])[0][0], 0.5)
            with patch(
                "urllib.request.urlopen", side_effect=AssertionError("no network call")
            ):
                with self.assertRaises(self.emb.EmbeddingError):
                    self.emb.embed([exact_limit + "가"])
        request.assert_called_once()

    def test_embed_empty_inputs(self):
        with patch.object(self.emb, "_api_key", side_effect=AssertionError("no key lookup")):
            self.assertEqual(self.emb.embed([]), [])

        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("no network call")
        ):
            with self.assertRaisesRegex(
                self.emb.EmbeddingError, "non-empty strings"
            ):
                self.emb.embed([""])
            with self.assertRaisesRegex(
                self.emb.EmbeddingError, "non-empty strings"
            ):
                self.emb.embed(["valid", ""])

    def test_embed_rejects_non_string_and_invalid_utf8_before_key_lookup(self):
        with patch.object(self.emb, "_api_key", side_effect=AssertionError("no key lookup")):
            with self.assertRaisesRegex(
                self.emb.EmbeddingError, "non-empty strings"
            ):
                self.emb.embed(cast(list[str], [123]))
            with self.assertRaisesRegex(self.emb.EmbeddingError, "valid UTF-8"):
                self.emb.embed(["\ud800"])

    def test_embed_batch_dims_and_ordering(self):
        payload = {
            "data": [
                {"embedding": [0.2] * 1536, "index": 1},
                {"embedding": [0.1] * 1536, "index": 0},
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
        self.assertEqual(out[0][0], 0.1)
        self.assertEqual(out[1][0], 0.2)
        self.assertTrue(all(value == 0.1 for value in out[0]))
        self.assertTrue(all(value == 0.2 for value in out[1]))
        self.assertEqual(sent["body"]["model"], "openai/text-embedding-3-small")
        self.assertEqual(sent["body"]["dimensions"], 1536)
        self.assertNotIn("sk-test", json.dumps(sent["body"]))
        headers_lc = {k.lower(): v for k, v in sent["headers"].items()}
        self.assertEqual(headers_lc.get("x-title"), "khala")
        self.assertEqual(headers_lc.get("http-referer"), "https://github.com/songkg7/khala")
        self.assertTrue(headers_lc["authorization"].startswith("Bearer "))

    def test_token_usage_accumulates_and_resets(self):
        payload = {
            "data": [
                {"embedding": [0.1] * 1536, "index": 0},
                {"embedding": [0.2] * 1536, "index": 1},
            ],
            "usage": {"prompt_tokens": 11, "total_tokens": 13},
        }
        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=lambda req, timeout=None: self._response(payload)):
                self.emb.embedding_token_usage()  # reset
                self.emb.embed(["a", "b"])
        self.assertEqual(
            self.emb.embedding_token_usage(),
            {"prompt_tokens": 11, "total_tokens": 13},
        )
        # reader resets the counter
        self.assertEqual(self.emb.embedding_token_usage(), {"prompt_tokens": 0, "total_tokens": 0})

    def test_token_usage_absent_or_invalid_is_ignored(self):
        payload = {
            "data": [{"embedding": [0.3] * 1536, "index": 0}],
            "usage": {"prompt_tokens": "many", "total_tokens": -5},
        }
        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=lambda req, timeout=None: self._response(payload)):
                self.emb.embedding_token_usage()
                self.emb.embed(["a"])
        self.assertEqual(self.emb.embedding_token_usage(), {"prompt_tokens": 0, "total_tokens": 0})

    def test_embed_splits_batches_by_input_count(self):
        batch_sizes = []

        def fake_open(req, timeout=None):
            texts = json.loads(req.data.decode())["input"]
            batch_sizes.append(len(texts))
            return self._response(
                {
                    "data": [
                        {"embedding": [float(index)] * 1536, "index": index}
                        for index in range(len(texts))
                    ]
                }
            )

        texts = [f"text-{index}" for index in range(self.emb.MAX_BATCH_INPUTS + 1)]
        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_open):
                vectors = self.emb.embed(texts)

        self.assertEqual(batch_sizes, [self.emb.MAX_BATCH_INPUTS, 1])
        self.assertEqual(len(vectors), len(texts))

    def test_embed_splits_batches_by_total_utf8_bytes(self):
        batch_sizes = []

        def fake_open(req, timeout=None):
            texts = json.loads(req.data.decode())["input"]
            batch_sizes.append(len(texts))
            return self._response(
                {
                    "data": [
                        {"embedding": [0.25] * 1536, "index": index}
                        for index in range(len(texts))
                    ]
                }
            )

        full_size_text = "x" * self.emb.MAX_TEXT_UTF8_BYTES
        inputs_per_byte_batch = (
            self.emb.MAX_BATCH_UTF8_BYTES // self.emb.MAX_TEXT_UTF8_BYTES
        )
        texts = [full_size_text] * (inputs_per_byte_batch + 1)
        with patch.dict(os.environ, TEST_API_KEY_ENV, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_open):
                vectors = self.emb.embed(texts)

        self.assertLess(inputs_per_byte_batch, self.emb.MAX_BATCH_INPUTS)
        self.assertEqual(batch_sizes, [inputs_per_byte_batch, 1])
        self.assertEqual(len(vectors), len(texts))

    def test_request_batch_enforces_empty_count_and_byte_limits_before_network(self):
        cases = {
            "empty": [],
            "count": ["x"] * (self.emb.MAX_BATCH_INPUTS + 1),
            "bytes": ["x" * self.emb.MAX_TEXT_UTF8_BYTES]
            * ((self.emb.MAX_BATCH_UTF8_BYTES // self.emb.MAX_TEXT_UTF8_BYTES) + 1),
        }

        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("no network call")
        ):
            for name, texts in cases.items():
                with self.subTest(name=name):
                    with self.assertRaises(self.emb.EmbeddingError):
                        self.emb._request_batch(texts, "api-key")

    def test_request_batch_partition_rejects_mismatched_size_metadata(self):
        with self.assertRaisesRegex(self.emb.EmbeddingError, "size count"):
            list(self.emb._request_batches(["text"], []))

    def test_request_batch_requires_complete_unique_integer_indexes(self):
        vector = [0.5] * self.emb.DIMENSIONS
        cases = {
            "missing": [{"embedding": vector}, {"embedding": vector, "index": 1}],
            "duplicate": [
                {"embedding": vector, "index": 0},
                {"embedding": vector, "index": 0},
            ],
            "out of range": [
                {"embedding": vector, "index": 0},
                {"embedding": vector, "index": 2},
            ],
            "string index": [
                {"embedding": vector, "index": "0"},
                {"embedding": vector, "index": 1},
            ],
            "boolean index": [
                {"embedding": vector, "index": False},
                {"embedding": vector, "index": 1},
            ],
            "short response": [{"embedding": vector, "index": 0}],
        }

        for name, data in cases.items():
            with self.subTest(name=name):
                with patch(
                    "urllib.request.urlopen", return_value=self._response({"data": data})
                ):
                    with self.assertRaises(self.emb.EmbeddingError):
                        self.emb._request_batch(["first", "second"], "api-key")

    def test_request_batch_rejects_invalid_vectors_without_content_leak(self):
        private_content = "private-embedding-source"
        valid = [0.5] * self.emb.DIMENSIONS
        cases = {
            "item": [None],
            "embedding type": [{"embedding": "not-a-list", "index": 0}],
            "dimension": [{"embedding": valid[:-1], "index": 0}],
            "non numeric": [
                {"embedding": valid[:-1] + [private_content], "index": 0}
            ],
            "boolean": [{"embedding": valid[:-1] + [True], "index": 0}],
            "not finite": [{"embedding": valid[:-1] + [float("inf")], "index": 0}],
        }

        for name, data in cases.items():
            with self.subTest(name=name):
                with patch(
                    "urllib.request.urlopen", return_value=self._response({"data": data})
                ):
                    with self.assertRaises(self.emb.EmbeddingError) as raised:
                        self.emb._request_batch([private_content], "api-key")
                self.assertNotIn(private_content, str(raised.exception))

    def test_request_batch_rejects_invalid_json_shape(self):
        invalid_body = Mock()
        invalid_body.__enter__ = Mock(return_value=invalid_body)
        invalid_body.__exit__ = Mock(return_value=False)
        invalid_body.read.return_value = b"not-json"
        with patch("urllib.request.urlopen", return_value=invalid_body):
            with self.assertRaises(self.emb.EmbeddingError):
                self.emb._request_batch(["text"], "api-key")

        non_bytes_body = Mock()
        non_bytes_body.__enter__ = Mock(return_value=non_bytes_body)
        non_bytes_body.__exit__ = Mock(return_value=False)
        non_bytes_body.read.return_value = "not bytes"
        with patch("urllib.request.urlopen", return_value=non_bytes_body):
            with self.assertRaises(self.emb.EmbeddingError):
                self.emb._request_batch(["text"], "api-key")

        with self.assertRaises(self.emb.EmbeddingError):
            self.emb._decode_response(object())

        with patch(
            "urllib.request.urlopen", return_value=self._response([{"data": []}])
        ):
            with self.assertRaises(self.emb.EmbeddingError):
                self.emb._request_batch(["text"], "api-key")

    def test_network_error_does_not_include_original_error_or_content(self):
        private_content = "private-network-content"
        with patch("urllib.request.urlopen", side_effect=OSError(private_content)):
            with self.assertRaises(self.emb.EmbeddingError) as raised:
                self.emb._request_batch([private_content], "api-key")
        self.assertNotIn(private_content, str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_http_error_exposes_status_only(self):
        private_content = "private-http-content"
        http_error = urllib.error.HTTPError(
            self.emb.EMBEDDINGS_URL,
            429,
            private_content,
            hdrs=Message(),
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(self.emb.EmbeddingError, "HTTP 429") as raised:
                self.emb._request_batch([private_content], "api-key")
        self.assertNotIn(private_content, str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
