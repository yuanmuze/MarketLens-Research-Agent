"""OpenAI-compatible LLM provider with HTTP-level error handling.

Uses the openai SDK for request/response including tool calling.
Does NOT log API keys or Authorization headers.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from marketlens.agent.providers.base import LLMClient

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(LLMClient):
    """LLM client using the openai SDK for OpenAI-compatible APIs.

    Handles: tool call response parsing, timeouts, auth errors (401),
    rate limits (429), server errors (500), and malformed responses.

    Environment variables:
      MARKETLENS_AGENT_API_KEY  — required, API key
      MARKETLENS_AGENT_BASE_URL — optional, defaults to https://api.openai.com/v1
      MARKETLENS_AGENT_MODEL    — optional, defaults to gpt-4.1-mini
      MARKETLENS_AGENT_TIMEOUT_SECONDS — optional, defaults to 30
    """

    def __init__(self) -> None:
        """Initialize from environment variables."""
        self._api_key = os.environ.get("MARKETLENS_AGENT_API_KEY", "")
        self._base_url = os.environ.get(
            "MARKETLENS_AGENT_BASE_URL", "https://api.openai.com/v1"
        )
        self._model = os.environ.get("MARKETLENS_AGENT_MODEL", "gpt-4.1-mini")
        self._timeout_s = float(
            os.environ.get("MARKETLENS_AGENT_TIMEOUT_SECONDS", "30")
        )

        if not self._api_key:
            raise ValueError(
                "MARKETLENS_AGENT_API_KEY is not set. "
                "Set this environment variable to use a real LLM."
            )

        self._client: Any = None

    @property
    def model_name(self) -> str:
        """Return the active model identifier."""
        return self._model

    def _get_client(self) -> Any:
        """Lazy-init the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_s,
            )
        return self._client

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Send messages and parse the response including tool calls.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions.
            timeout_s: Per-request timeout.

        Returns:
            Dict with "content" (str) and optional "tool_calls" list.

        Raises:
            ConnectionError: Network or auth errors.
            TimeoutError: Request exceeded timeout_s.
            RuntimeError: Unexpected response format.
        """
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                timeout=min(timeout_s, self._timeout_s),
            )
        except Exception as e:
            error_str = str(e).lower()
            # Never log the full exception — may contain API key in URL params
            if "401" in error_str or "unauthorized" in error_str or "403" in error_str:
                raise ConnectionError("LLM authentication failed (401/403). Check MARKETLENS_AGENT_API_KEY.") from e
            if "429" in error_str or "rate limit" in error_str:
                raise ConnectionError("LLM rate limited (429). Retry later.") from e
            if "500" in error_str or "server error" in error_str or "502" in error_str or "503" in error_str:
                raise ConnectionError(f"LLM server error: {e}") from e
            if "timeout" in error_str or "timed out" in error_str:
                raise TimeoutError(f"LLM request timed out after {timeout_s}s") from e
            raise ConnectionError(f"LLM request failed: {type(e).__name__}") from e

        choice = resp.choices[0] if resp.choices else None
        if choice is None:
            raise RuntimeError("LLM response has no choices")

        msg = choice.message
        content = msg.content or ""

        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        return {"content": content, "tool_calls": tool_calls}
