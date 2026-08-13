"""Tests for structured logging / observability."""

from __future__ import annotations

from marketlens.observability import _sanitize


class TestSanitize:
    """Secret filtering in structured logs."""

    def test_filters_sensitive_keys(self) -> None:
        """Sensitive keys are removed."""
        payload = {
            "api_key": "sk-secret",
            "authorization": "Bearer token",
            "password": "hunter2",
            "database_url": "postgresql://user:pass@host/db",
            "request_id": "req-123",
            "event": "request_started",
        }
        cleaned = _sanitize(payload)
        assert "api_key" not in cleaned
        assert "authorization" not in cleaned
        assert "password" not in cleaned
        assert "database_url" not in cleaned
        # Safe fields preserved
        assert cleaned["request_id"] == "req-123"
        assert cleaned["event"] == "request_started"

    def test_case_insensitive(self) -> None:
        """Key matching is case-insensitive."""
        cleaned = _sanitize({"API_KEY": "sk-x", "Request_ID": "r"})
        assert "API_KEY" not in cleaned
        assert "Request_ID" in cleaned  # request_id is not sensitive

    def test_no_sensitive_keys_passthrough(self) -> None:
        """No sensitive keys → all fields preserved."""
        payload = {"request_id": "r1", "agent_run_id": 5, "latency_ms": 12.3}
        assert _sanitize(payload) == payload
