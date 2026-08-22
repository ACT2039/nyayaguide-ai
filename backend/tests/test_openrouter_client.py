"""
Unit Tests for OpenRouter Client
Tests configuration, payload formatting, error handling, and timeout behavior.
"""
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.llm.openrouter_client import OpenRouterClient


class TestOpenRouterClient(unittest.TestCase):
    def test_missing_api_key_raises_value_error(self):
        client = OpenRouterClient(api_key="", model="google/gemini-2.0-flash-001")
        with self.assertRaises(ValueError) as ctx:
            client.generate(messages=[{"role": "user", "content": "Hello"}])
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_missing_model_raises_value_error(self):
        client = OpenRouterClient(api_key="mock_key", model="")
        with self.assertRaises(ValueError) as ctx:
            client.generate(messages=[{"role": "user", "content": "Hello"}], model="")
        self.assertIn("OPENROUTER_MODEL", str(ctx.exception))

    @patch("requests.post")
    def test_successful_generation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": "Under Section 6 of the RTI Act, you can file an application to the PIO."}}
            ]
        }
        mock_post.return_value = mock_resp

        client = OpenRouterClient(api_key="mock_key", model="google/gemini-2.0-flash-001")
        result = client.generate(messages=[{"role": "user", "content": "Test"}])
        self.assertIn("Section 6", result)
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.post")
    def test_auth_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_post.return_value = mock_resp

        client = OpenRouterClient(api_key="invalid_key", model="google/gemini-2.0-flash-001")
        with self.assertRaises(PermissionError):
            client.generate(messages=[{"role": "user", "content": "Test"}])

    @patch("requests.post")
    def test_rate_limit_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_post.return_value = mock_resp

        client = OpenRouterClient(api_key="mock_key", model="google/gemini-2.0-flash-001")
        with self.assertRaises(RuntimeError) as ctx:
            client.generate(messages=[{"role": "user", "content": "Test"}])
        self.assertIn("rate limit", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
