"""Unit tests for persistence layer (SQLite in-memory, no PostgreSQL needed)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from marketlens.models import Product, ProductCategory
from marketlens.persistence.converters import (
    product_to_record,
    record_to_product,
)
from marketlens.persistence.models import (
    AgentToolCallRecord,
    Base,
)
from marketlens.persistence.repositories import AgentRunRepository, ProductRepository


@pytest.fixture
def session():
    """Create an in-memory SQLite session with schema and FK enforcement."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    # Enable foreign key enforcement so ON DELETE CASCADE works in SQLite
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture
def sample_product() -> Product:
    """A sample product."""
    return Product(
        product_id="P001",
        title="Test Wireless Headphones",
        brand="TestBrand",
        category=ProductCategory.ELECTRONICS,
        price=99.99,
        rating=4.5,
        review_count=100,
        attributes={"color": "Black"},
        description="Great headphones",
        images=["http://img/1.jpg"],
        url="http://example.com/P001",
    )


class TestConverters:
    """ORM <-> Pydantic conversion."""

    def test_product_to_record(self, sample_product: Product) -> None:
        """Product → ProductRecord conversion."""
        record = product_to_record(sample_product)
        assert record.product_id == "P001"
        assert record.title == "Test Wireless Headphones"
        assert record.brand == "TestBrand"
        assert str(record.price) == "99.99"

    def test_record_to_product(self, sample_product: Product) -> None:
        """ProductRecord → Product conversion roundtrip."""
        record = product_to_record(sample_product)
        product = record_to_product(record)
        assert product.product_id == sample_product.product_id
        assert product.title == sample_product.title
        assert product.price == sample_product.price
        assert product.rating == sample_product.rating
        assert product.attributes == sample_product.attributes
        assert product.images == sample_product.images
        assert product.url == sample_product.url


class TestProductRepository:
    """ProductRepository CRUD + upsert."""

    def test_upsert_insert(self, session, sample_product: Product) -> None:
        """First upsert inserts."""
        repo = ProductRepository(session)
        result = repo.upsert_many([sample_product])
        session.commit()
        assert result["inserted"] == 1
        assert repo.count() == 1

    def test_upsert_idempotent(self, session, sample_product: Product) -> None:
        """Re-upserting same product does not duplicate."""
        repo = ProductRepository(session)
        repo.upsert_many([sample_product])
        session.commit()
        # Second import of identical data
        result = repo.upsert_many([sample_product])
        session.commit()
        assert result["unchanged"] == 1
        assert result["inserted"] == 0
        assert repo.count() == 1  # Not doubled

    def test_upsert_update(self, session, sample_product: Product) -> None:
        """Updating an existing product's fields."""
        repo = ProductRepository(session)
        repo.upsert_many([sample_product])
        session.commit()

        updated = sample_product.model_copy()
        updated.price = 79.99
        result = repo.upsert_many([updated])
        session.commit()
        assert result["updated"] == 1
        fetched = repo.get_by_id("P001")
        assert fetched is not None
        assert fetched.price == 79.99

    def test_get_by_id_missing(self, session) -> None:
        """Missing product returns None."""
        repo = ProductRepository(session)
        assert repo.get_by_id("NONEXISTENT") is None

    def test_list_products(self, session, sample_product: Product) -> None:
        """List returns products."""
        repo = ProductRepository(session)
        repo.upsert_many([sample_product])
        session.commit()
        products = repo.list_products()
        assert len(products) == 1
        assert products[0].product_id == "P001"


class TestAgentRunRepository:
    """AgentRun + ToolCall lifecycle."""

    def test_full_lifecycle(self, session) -> None:
        """create_running → add_tool_calls → mark_completed."""
        repo = AgentRunRepository(session)
        record = repo.create_running("req-001", "best headphones", "balanced")
        session.commit()

        repo.add_tool_calls(record.id, [
            {"step_number": 1, "tool_name": "search_catalog", "arguments": {"query": "headphones"}, "result_product_ids": ["P001"], "success": True},
        ])
        session.commit()

        repo.mark_completed(
            record.id, "completed", "hybrid", False,
            {"answer": "Found headphones"}, 123.45,
        )
        session.commit()

        fetched = repo.get_by_request_id("req-001")
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.mode_used == "hybrid"
        assert fetched.response == {"answer": "Found headphones"}
        assert fetched.latency_ms == 123.45

        tool_calls = repo.get_tool_calls(record.id)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "search_catalog"
        assert tool_calls[0].result_product_ids == ["P001"]

    def test_failed_run(self, session) -> None:
        """mark_failed records sanitized error."""
        repo = AgentRunRepository(session)
        record = repo.create_running("req-002", "test", "balanced")
        session.commit()
        repo.mark_failed(record.id, "LLMConnectionError", "Connection refused")
        session.commit()
        fetched = repo.get_by_request_id("req-002")
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_type == "LLMConnectionError"
        assert fetched.error_message == "Connection refused"

    def test_request_id_unique(self, session) -> None:
        """Duplicate request_id raises IntegrityError."""
        repo = AgentRunRepository(session)
        repo.create_running("req-003", "a", "balanced")
        session.commit()
        with pytest.raises(Exception):
            repo.create_running("req-003", "b", "balanced")
            session.commit()
        session.rollback()

    def test_tool_call_fk_cascade(self, session) -> None:
        """Deleting agent run cascades to tool calls."""
        repo = AgentRunRepository(session)
        record = repo.create_running("req-004", "test", "balanced")
        session.commit()
        repo.add_tool_calls(record.id, [
            {"step_number": 1, "tool_name": "search_catalog", "arguments": {}, "result_product_ids": [], "success": True},
        ])
        session.commit()
        # Delete the run → tool calls cascade-deleted
        session.delete(record)
        session.commit()
        assert repo.get_by_request_id("req-004") is None
        # Tool calls gone
        from sqlalchemy import func, select
        count = session.scalar(select(func.count()).select_from(AgentToolCallRecord))
        assert count == 0

    def test_transaction_rollback(self, session) -> None:
        """Failure mid-write leaves no partial data."""
        repo = AgentRunRepository(session)
        record = repo.create_running("req-005", "test", "balanced")
        session.commit()
        # Simulate a failing write: add tool calls then rollback
        try:
            repo.add_tool_calls(record.id, [
                {"step_number": 1, "tool_name": "bad_tool", "arguments": {}, "result_product_ids": [], "success": False, "error_type": "ToolError"},
            ])
            raise RuntimeError("simulated failure")
        except RuntimeError:
            session.rollback()
        # Tool calls should not be persisted
        assert repo.get_tool_calls(record.id) == []
