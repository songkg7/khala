import unittest
from pathlib import Path


class DockerComposeSecurityTest(unittest.TestCase):
    def test_postgres_uses_explicit_password_and_localhost_binding(self):
        compose = (Path(__file__).parents[1] / "docker-compose.yml").read_text()

        self.assertIn("KHALA_POSTGRES_PASSWORD:?", compose)
        self.assertNotIn("POSTGRES_PASSWORD: khala", compose)
        self.assertIn('127.0.0.1:${KHALA_POSTGRES_PORT:-5432}:5432', compose)


if __name__ == "__main__":
    unittest.main()
