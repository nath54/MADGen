"""
HTTP client for communicating with llama.cpp and OpenAI-compatible inference servers.

Reuses and adapts the client implementation from LLMTests with connection health
checking, timeout configuration, and structured tool calling support.
"""

# Import Modules
import typing

import logging
import httpx

from src.llm.llm_types import (
    ChatMessage,
    ToolDefinition,
    ChatCompletionResponse,
)

logger: logging.Logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with llama-server or OpenAI-compatible LLM endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "",
        timeout_s: float = 60.0,
    ) -> None:
        """
        Initialize the LLM API client.

        Args:
            base_url (str): Server endpoint root URL.
            api_key (str): Optional bearer authorization token.
            timeout_s (float): Request timeout limit in seconds.
        """

        self.base_url: str = base_url.rstrip("/")
        self.timeout_s: float = timeout_s
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def is_healthy(self) -> bool:
        """
        Check whether the LLM server is accessible and responding.

        Returns:
            bool: True if server responded with 200 OK, False otherwise.
        """

        try:
            with httpx.Client(timeout=2.0) as client:
                resp: httpx.Response = client.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                )
                return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def send_chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: typing.Any = None,
        model: str = "default",
        temperature: float = 0.7,
    ) -> ChatCompletionResponse:
        """
        Send a chat completion request to the server.

        Args:
            messages (list[ChatMessage]): Conversation history turns.
            tools (list[ToolDefinition] | None): Optional list of callable tools.
            tool_choice (typing.Any): Tool selection strategy or specific function.
            model (str): Model name or alias.
            temperature (float): Sampling temperature.

        Returns:
            ChatCompletionResponse: Parsed server response.

        Raises:
            httpx.HTTPStatusError: If server returned an error status code.
            httpx.RequestError: If network communication failed.
        """

        payload: dict[str, typing.Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        with httpx.Client(timeout=self.timeout_s) as client:
            response: httpx.Response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            resp_data: ChatCompletionResponse = response.json()
            return resp_data
