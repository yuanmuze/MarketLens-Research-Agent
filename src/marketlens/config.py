"""Validated MarketLens runtime configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

CatalogBackendName = Literal["json", "postgres"]
SemanticBackendName = Literal["memory", "pgvector"]


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


@dataclass(frozen=True)
class MarketLensSettings:
    """Runtime settings shared by API startup and retrieval injection."""

    catalog_backend: CatalogBackendName = "json"
    semantic_backend: SemanticBackendName = "memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    use_fake_embeddings: bool = False
    use_fake_llm: bool = False
    catalog_path: Path | None = None

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
        return cls(
            catalog_backend=catalog,
            semantic_backend=semantic,
            embedding_model=model,
            use_fake_embeddings=_boolean("MARKETLENS_USE_FAKE_EMBEDDINGS"),
            use_fake_llm=_boolean("MARKETLENS_USE_FAKE_LLM"),
            catalog_path=Path(catalog_path_raw) if catalog_path_raw else None,
        )
