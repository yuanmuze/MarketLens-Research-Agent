"""Shared contract for semantic retrieval backends."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class SemanticBackendUnavailableError(RuntimeError):
    """Raised when a configured semantic backend cannot serve queries."""


@dataclass(frozen=True)
class SemanticBackendStatus:
    """Backend readiness information safe to expose in health responses."""

    backend: str
    model: str
    dimension: int
    ready: bool
    indexed_count: int
    expected_count: int
    detail: str = ""


@runtime_checkable
class SemanticRetriever(Protocol):
    """Minimal interface used by the active retrieval service."""

    @property
    def backend_name(self) -> str:
        """Return the storage/search backend name."""
        ...

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        ...

    @property
    def dim(self) -> int:
        """Return the query embedding dimension."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 20,
        candidate_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ranked product IDs and semantic similarity scores."""
        ...

    def status(
        self,
        expected_product_ids: Collection[str] | None = None,
    ) -> SemanticBackendStatus:
        """Return readiness for the configured catalog and index."""
        ...
