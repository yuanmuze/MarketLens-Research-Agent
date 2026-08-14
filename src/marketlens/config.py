"""Validated MarketLens runtime configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

CatalogBackendName = Literal["json", "postgres"]
SemanticBackendName = Literal["memory", "pgvector"]

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
)


def _choice(
    variable: str,
    default: str,
    allowed: tuple[str, ...],
) -> str:
    """Read and validate a normalized environment choice."""
    value = os.environ.get(variable, default).strip().lower()
    if value not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"{variable} must be one of: {choices}")
    return value


def _boolean(variable: str, default: bool = False) -> bool:
    """Read a strict boolean environment variable."""
    raw = os.environ.get(variable)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{variable} must be a boolean value")


def _cors_origins(raw: str | None) -> tuple[str, ...]:
    """Parse and validate a comma-separated CORS origin allowlist."""
    if raw is None or not raw.strip():
        return DEFAULT_CORS_ORIGINS
    origins = tuple(dict.fromkeys(item.strip().rstrip("/") for item in raw.split(",") if item.strip()))
    if not origins:
        raise ValueError("MARKETLENS_CORS_ORIGINS must contain at least one origin")
    for origin in origins:
        if origin == "*":
            continue
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("MARKETLENS_CORS_ORIGINS contains an invalid origin")
    return origins


@dataclass(frozen=True)
class MarketLensSettings:
    """Runtime settings shared by API startup and retrieval injection."""

    catalog_backend: CatalogBackendName = "json"
    semantic_backend: SemanticBackendName = "memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    use_fake_embeddings: bool = False
    use_fake_llm: bool = True
    catalog_path: Path | None = None
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    cors_allow_credentials: bool = False

    @classmethod
    def from_env(cls) -> MarketLensSettings:
        """Build validated settings without exposing sensitive values."""
        catalog = cast(
            CatalogBackendName,
            _choice("MARKETLENS_CATALOG_BACKEND", "json", ("json", "postgres")),
        )
        semantic = cast(
            SemanticBackendName,
            _choice(
                "MARKETLENS_SEMANTIC_BACKEND",
                "memory",
                ("memory", "pgvector"),
            ),
        )
        model = os.environ.get(
            "MARKETLENS_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        ).strip()
        if not model:
            raise ValueError("MARKETLENS_EMBEDDING_MODEL must not be empty")
        catalog_path_raw = os.environ.get("MARKETLENS_CATALOG_PATH", "").strip()
        cors_allow_credentials = _boolean("MARKETLENS_CORS_ALLOW_CREDENTIALS")
        cors_origins = _cors_origins(os.environ.get("MARKETLENS_CORS_ORIGINS"))
        if cors_allow_credentials and "*" in cors_origins:
            raise ValueError(
                "MARKETLENS_CORS_ORIGINS cannot contain '*' when credentials are enabled"
            )
        return cls(
            catalog_backend=catalog,
            semantic_backend=semantic,
            embedding_model=model,
            use_fake_embeddings=_boolean("MARKETLENS_USE_FAKE_EMBEDDINGS"),
            use_fake_llm=_boolean("MARKETLENS_USE_FAKE_LLM", default=True),
            catalog_path=Path(catalog_path_raw) if catalog_path_raw else None,
            cors_origins=cors_origins,
            cors_allow_credentials=cors_allow_credentials,
        )
