"""LLM provider protocol — injectable, testable interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract LLM client for the agent orchestrator.

    Implementations must handle: sending prompts, receiving structured
    responses (tool calls), and timeout/error handling.

    Tests inject FakeLLMClient; production uses OpenAI-compatible APIs.
    """

    @abstractmethod
    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Send messages to the LLM and return response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            tools: OpenAI-format tool definitions.
            timeout_s: Request timeout in seconds.

        Returns:
            Response dict with at least "content" (str) and optional
            "tool_calls" (list of {"name": str, "arguments": dict}).

        Raises:
            TimeoutError: If the request exceeds timeout_s.
            ConnectionError: If the provider is unreachable.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model identifier."""
        ...


class FakeLLMClient(LLMClient):
    """Deterministic fake LLM for testing.

    Returns scripted responses from a predefined list. Each send()
    consumes one response from the script.
    """

    def __init__(self, script: list[dict[str, Any]], model_name: str = "fake") -> None:
        """Initialize with a scripted response list.

        Args:
            script: List of response dicts consumed in order.
            model_name: Reported model name.
        """
        self._script = script
        self._pos = 0
        self._model_name = model_name
        self.calls: list[dict[str, Any]] = []  # Record of all calls

    @property
    def model_name(self) -> str:
        """Return the fake model name."""
        return self._model_name

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Return the next scripted response.

        Args:
            messages: Ignored (fake doesn't read them).
            tools: Ignored (fake returns scripted tool calls).
            timeout_s: Ignored.

        Returns:
            Next scripted response dict.
        """
        self.calls.append({"messages": messages, "tools": tools, "timeout_s": timeout_s})
        if self._pos >= len(self._script):
            # Loop: return last response again
            return self._script[-1] if self._script else {"content": "done"}
        resp = self._script[self._pos]
        self._pos += 1
        return resp
