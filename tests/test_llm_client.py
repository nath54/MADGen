"""
Unit tests for LLMClient initialization, health checking, and network error handling.
"""

# Import Modules
import unittest

from src.llm.client import LLMClient


class TestLLMClient(unittest.TestCase):
    """
    Validation suite for LLM HTTP client connectivity and headers.
    """

    def test_client_initialization(self) -> None:
        """
        Verify client initializes headers, strips trailing slashes, and sets timeouts.
        """

        client: LLMClient = LLMClient(
            base_url="http://localhost:8080/v1/",
            api_key="test_secret_token",
            timeout_s=30.0,
        )

        self.assertEqual(client.base_url, "http://localhost:8080/v1")
        self.assertEqual(client.headers["Content-Type"], "application/json")
        self.assertEqual(client.headers["Authorization"], "Bearer test_secret_token")
        self.assertEqual(client.timeout_s, 30.0)

    def test_is_healthy_unreachable_endpoint(self) -> None:
        """
        Verify is_healthy returns False cleanly when the target port is unreachable.
        """

        # Point to an unused invalid port
        client: LLMClient = LLMClient(
            base_url="http://127.0.0.1:59999/v1",
            timeout_s=1.0,
        )

        healthy: bool = client.is_healthy()
        self.assertFalse(healthy)


if __name__ == "__main__":
    unittest.main()
