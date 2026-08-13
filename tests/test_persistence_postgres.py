"""PostgreSQL integration tests (skip if no test database configured).

Requires MARKETLENS_TEST_DATABASE_URL pointing to a dedicated test DB.
Run with: uv run pytest -m postgres
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from marketlens.models import Product, ProductCategory
from marketlens.persistence.models import Base
from marketlens.persistence.repositories import AgentRunRepository, ProductRepository

TEST_DB_URL = os.environ.get("MARKETLENS_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres


def _require_test_db() -> str:
    """Return test DB URL or skip."""
    if not TEST_DB_URL:
        pytest.skip("MARKETLENS_TEST_DATABASE_URL not set — skipping PostgreSQL integration test")
    return TEST_DB_URL


@pytest.fixture
def session():
    """Create a session against the dedicated test DB, using a temp schema."""
    url = _require_test_db()
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    # Start clean
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_tool_calls"))
        conn.execute(text("DELETE FROM agent_runs"))
        conn.execute(text("DELETE FROM products"))
    yield s
    s.close()
    # Clean up after
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM agent_tool_calls"))
        conn.execute(text("DELETE FROM agent_runs"))
        conn.execute(text("DELETE FROM products"))
    engine.dispose()


class TestPostgresProductRepository:
    """Product upsert + idempotency against PostgreSQL."""

    def test_upsert_and_idempotent(self, session) -> None:
        """Insert twice, count stays 1."""
        repo = ProductRepository(session)
        product = Product(
            product_id="PG001", title="PG Product", brand="Brand",
            category=ProductCategory.ELECTRONICS, price=50.0, rating=4.0,
        )
        repo.upsert_many([product])
        session.commit()
        assert repo.count() == 1

        repo.upsert_many([product])
        session.commit()
        assert repo.count() == 1  # Idempotent

    def test_orm_to_pydantic(self, session) -> None:
        """ORM record converts back to Pydantic Product."""
        repo = ProductRepository(session)
        product = Product(
            product_id="PG002", title="Convert Me", brand="X",
            category=ProductCategory.ELECTRONICS, price=10.0, rating=3.5,
            attributes={"color": "red"},
        )
        repo.upsert_many([product])
        session.commit()
        fetched = repo.get_by_id("PG002")
        assert fetched is not None
        assert fetched.title == "Convert Me"
        assert fetched.attributes == {"color": "red"}
        assert fetched.price == 10.0


class TestPostgresAgentRun:
    """Agent run + tool call relationship on PostgreSQL."""

    def test_full_run_and_tool_calls(self, session) -> None:
        """Create run, add tool calls, mark completed, query by request_id."""
        repo = AgentRunRepository(session)
        record = repo.create_running("pg-req-001", "best product", "balanced")
        session.commit()

        repo.add_tool_calls(record.id, [
            {"step_number": 1, "tool_name": "search_catalog", "arguments": {"query": "test"}, "result_product_ids": ["PG001"], "success": True},
            {"step_number": 2, "tool_name": "compare_products", "arguments": {"product_ids": ["PG001", "PG002"]}, "result_product_ids": ["PG001", "PG002"], "success": True},
        ])
        session.commit()

        repo.mark_completed(record.id, "completed", "hybrid", False, {"answer": "ok"}, 100.0)
        session.commit()

        fetched = repo.get_by_request_id("pg-req-001")
        assert fetched is not None
        assert fetched.status == "completed"
        calls = repo.get_tool_calls(record.id)
        assert len(calls) == 2
        assert calls[0].tool_name == "search_catalog"
        assert calls[1].tool_name == "compare_products"

    def test_failed_run_recorded(self, session) -> None:
        """Failed run is queryable."""
        repo = AgentRunRepository(session)
        record = repo.create_running("pg-req-002", "test", "quality")
        session.commit()
        repo.mark_failed(record.id, "TimeoutError", "LLM timed out", 5000.0)
        session.commit()
        fetched = repo.get_by_request_id("pg-req-002")
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_type == "TimeoutError"
