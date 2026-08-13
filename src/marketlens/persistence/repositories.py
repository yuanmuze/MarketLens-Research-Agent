"""Repository layer — isolates SQL from routes and orchestrator.

ProductRepository and AgentRunRepository wrap all database access.
API routes and agent orchestration call these repositories instead
of writing raw SQL or ORM queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from marketlens.models import Product
from marketlens.persistence.converters import product_to_record, record_to_product
from marketlens.persistence.models import (
    AgentRunRecord,
    AgentToolCallRecord,
    ProductEmbeddingRecord,
    ProductRecord,
)


def _utcnow() -> datetime:
    """UTC now, naive."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ProductRepository:
    """CRUD + upsert for products."""

    def __init__(self, session: Session) -> None:
        """Initialize with a SQLAlchemy session."""
        self._session = session

    def get_by_id(self, product_id: str) -> Product | None:
        """Fetch a product by ID, converted to Pydantic."""
        record = self._session.get(ProductRecord, product_id)
        return record_to_product(record) if record else None

    def count(self) -> int:
        """Count products."""
        return self._session.query(ProductRecord).count()

    def list_products(self, offset: int = 0, limit: int = 100) -> list[Product]:
        """List products with pagination."""
        stmt = select(ProductRecord).order_by(ProductRecord.product_id).offset(offset).limit(limit)
        records = self._session.scalars(stmt).all()
        return [record_to_product(r) for r in records]

    def upsert_many(self, products: list[Product]) -> dict[str, int]:
        """Upsert products by product_id.

        Returns counts: inserted, updated, unchanged, failed.
        """
        inserted = 0
        updated = 0
        unchanged = 0
        failed = 0

        for product in products:
            try:
                record = product_to_record(product)
                existing = self._session.get(ProductRecord, product.product_id)
                if existing is None:
                    self._session.add(record)
                    inserted += 1
                else:
                    # Compare core fields to detect actual change
                    changed = (
                        existing.title != record.title
                        or existing.brand != record.brand
                        or existing.price != record.price
                        or existing.rating != record.rating
                    )
                    if changed:
                        existing.title = record.title
                        existing.description = record.description
                        existing.brand = record.brand
                        existing.category = record.category
                        existing.price = record.price
                        existing.rating = record.rating
                        existing.review_count = record.review_count
                        existing.extra = record.extra
                        existing.updated_at = _utcnow()
                        updated += 1
                    else:
                        unchanged += 1
            except Exception:
                failed += 1
                self._session.rollback()

        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
            "failed": failed,
        }


class AgentRunRepository:
    """Lifecycle management for agent runs."""

    def __init__(self, session: Session) -> None:
        """Initialize with a SQLAlchemy session."""
        self._session = session

    def create_running(
        self,
        request_id: str,
        user_query: str,
        mode_requested: str,
        constraints: dict[str, Any] | None = None,
    ) -> AgentRunRecord:
        """Create an agent run in 'running' status."""
        record = AgentRunRecord(
            request_id=request_id,
            user_query=user_query,
            status="running",
            mode_requested=mode_requested,
            mode_used=None,
            degraded=False,
            constraints=constraints,
            started_at=_utcnow(),
        )
        self._session.add(record)
        self._session.flush()  # Assign id
        return record

    def add_tool_calls(
        self,
        agent_run_id: int,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Insert tool call records for an agent run."""
        for tc in tool_calls:
            record = AgentToolCallRecord(
                agent_run_id=agent_run_id,
                step_number=tc.get("step_number", 0),
                tool_name=tc.get("tool_name", "unknown"),
                arguments=tc.get("arguments"),
                result_product_ids=tc.get("result_product_ids"),
                success=tc.get("success", True),
                error_type=tc.get("error_type"),
                latency_ms=tc.get("latency_ms"),
            )
            self._session.add(record)

    def mark_completed(
        self,
        agent_run_id: int,
        status: str,
        mode_used: str,
        degraded: bool,
        response: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Mark an agent run completed with final response."""
        record = self._session.get(AgentRunRecord, agent_run_id)
        if record is None:
            return
        record.status = status
        record.mode_used = mode_used
        record.degraded = degraded
        record.response = response
        record.latency_ms = latency_ms
        record.completed_at = _utcnow()

    def mark_failed(
        self,
        agent_run_id: int,
        error_type: str,
        error_message: str,
        latency_ms: float | None = None,
    ) -> None:
        """Mark an agent run failed with a sanitized error message."""
        record = self._session.get(AgentRunRecord, agent_run_id)
        if record is None:
            return
        record.status = "failed"
        record.error_type = error_type
        record.error_message = error_message  # Already sanitized by caller
        record.latency_ms = latency_ms
        record.completed_at = _utcnow()

    def get_by_request_id(self, request_id: str) -> AgentRunRecord | None:
        """Fetch an agent run by request_id."""
        stmt = select(AgentRunRecord).where(AgentRunRecord.request_id == request_id)
        return self._session.scalars(stmt).first()

    def get_tool_calls(self, agent_run_id: int) -> list[AgentToolCallRecord]:
        """Fetch tool calls for an agent run, ordered by step."""
        stmt = (
            select(AgentToolCallRecord)
            .where(AgentToolCallRecord.agent_run_id == agent_run_id)
            .order_by(AgentToolCallRecord.step_number, AgentToolCallRecord.id)
        )
        return list(self._session.scalars(stmt).all())


class ProductEmbeddingRepository:
    """pgvector embedding storage + cosine search."""

    def __init__(self, session: Session) -> None:
        """Initialize with a SQLAlchemy session."""
        self._session = session

    def upsert_many(
        self,
        product_ids: list[str],
        embeddings: list[list[float]],
        model_name: str,
        dim: int,
    ) -> dict[str, int]:
        """Batch-upsert embeddings (idempotent per product_id + model_name).

        Raises on dimension mismatch. Returns inserted/updated/unchanged counts.
        """
        if len(product_ids) != len(embeddings):
            raise ValueError(
                f"product_ids ({len(product_ids)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        for emb in embeddings:
            if len(emb) != dim:
                raise ValueError(
                    f"Embedding dimension {len(emb)} != expected {dim}"
                )

        inserted = 0
        updated = 0
        unchanged = 0
        for pid, emb in zip(product_ids, embeddings):
            existing = self._session.query(ProductEmbeddingRecord).filter_by(
                product_id=pid, model_name=model_name
            ).first()
            if existing is None:
                self._session.add(ProductEmbeddingRecord(
                    product_id=pid,
                    model_name=model_name,
                    dim=dim,
                    embedding=emb,
                ))
                inserted += 1
            else:
                if existing.embedding != emb or existing.dim != dim:
                    existing.embedding = emb
                    existing.dim = dim
                    updated += 1
                else:
                    unchanged += 1

        return {"inserted": inserted, "updated": updated, "unchanged": unchanged}

    def count(self, model_name: str | None = None) -> int:
        """Count stored embeddings, optionally filtered by model."""
        q = self._session.query(ProductEmbeddingRecord)
        if model_name:
            q = q.filter(ProductEmbeddingRecord.model_name == model_name)
        return q.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        model_name: str | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine similarity top-k search (1 - cosine distance)."""
        q = self._session.query(
            ProductEmbeddingRecord.product_id,
            (1.0 - ProductEmbeddingRecord.embedding.cosine_distance(query_embedding)).label("similarity"),
        )
        if model_name:
            q = q.filter(ProductEmbeddingRecord.model_name == model_name)
        q = q.order_by(
            ProductEmbeddingRecord.embedding.cosine_distance(query_embedding)
        ).limit(top_k)

        return [(pid, float(sim)) for pid, sim in q.all()]
