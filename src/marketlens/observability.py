"""Structured logging for MarketLens.

Emits JSON-line structured events with stable fields. Never logs API
keys, database passwords/URLs, Authorization headers, full user input,
or unbounded LLM responses.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("marketlens.observability")

# Fields that must never be logged.
_SENSITIVE_KEYS = {
    "api_key", "authorization", "password", "database_url", "token",
    "access_token", "secret",
}


def _sanitize(extra: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive keys from a log payload (best-effort)."""
    return {
        k: v for k, v in extra.items()
        if k.lower() not in _SENSITIVE_KEYS
    }


def log_event(event: str, **extra: Any) -> None:
    """Emit a structured log event (JSON line).

    Args:
        event: Event name (e.g. request_started, agent_completed).
        **extra: Additional structured fields (sanitized).
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "event": event,
    }
    payload.update(_sanitize(extra))
    logger.info(json.dumps(payload, default=str))
