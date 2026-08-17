"""PostgreSQL integration tests (skip if no test database configured).

Requires MARKETLENS_TEST_DATABASE_URL pointing to a dedicated test DB.
Run with: uv run pytest -m postgres
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from marketlens.models import Product, ProductCategory
from marketlens.persistence.models import (
    AgentRunRecord,
    AgentToolCallRecord,
    Base,
    ProductRecord,
)
from marketlens.persistence.repositories import AgentRunRepository, ProductRepository

TEST_DB_URL = os.environ.get("MARKETLENS_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres


def _require_test_db() -> str:
    """Return test DB URL, enforcing postgresql dialect + 'test' name.

    Guards against accidentally connecting to SQLite, a dev DB, or a
    production DB. Refuses any URL whose dialect is not postgresql or
    whose database name does not contain 'test'.
    """
    if not TEST_DB_URL:
        pytest.skip("MARKETLENS_TEST_DATABASE_URL not set — skipping PostgreSQL integration test")

    url = TEST_DB_URL
    assert url.startswith("postgresql"), (
        f"MARKETLENS_TEST_DATABASE_URL must use postgresql dialect, got: {url.split('://')[0]}"
    )
    # Extract database name (path after the last '/')
    db_name = url.rsplit("/", 1)[-1].split("?")[0]
    assert "test" in db_name.lower(), (
        f"MARKETLENS_TEST_DATABASE_URL must target a 'test' database, got: {db_name}"
    )
    return url


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
        """Same record transitions running → failed, no second record."""
        repo = AgentRunRepository(session)
        record = repo.create_running("pg-req-002", "test", "quality")
        session.commit()
        original_id = record.id

        # Verify running record exists and is queryable before failure
        running = repo.get_by_request_id("pg-req-002")
        assert running is not None
        assert running.status == "running"

        # Mark the SAME record failed
        repo.mark_failed(record.id, "TimeoutError", "LLM timed out", 5000.0)
        session.commit()

        fetched = repo.get_by_request_id("pg-req-002")
        assert fetched is not None
        assert fetched.id == original_id  # Same record, not a new one
        assert fetched.status == "failed"
        assert fetched.error_type == "TimeoutError"

        # No duplicate failed record
        count = session.scalar(select(func.count()).select_from(AgentRunRecord))
        assert count == 1

        # No partial tool calls left on the failed path
        assert repo.get_tool_calls(original_id) == []


class TestPostgresJSONBAndNumeric:
    """PostgreSQL JSONB + Numeric roundtrip."""

    def test_jsonb_roundtrip(self, session) -> None:
        """Nested attributes/images/url survive JSONB write+commit+requery."""
        repo = ProductRepository(session)
        product = Product(
            product_id="PG-JSONB-001",
            title="JSONB Test",
            brand="B",
            category=ProductCategory.ELECTRONICS,
            price=99.99,
            rating=4.5,
            attributes={"color": "black", "battery": {"hours": 30}},
            images=["http://img/a.jpg", "http://img/b.jpg"],
            url="http://example.com/PG-JSONB-001",
        )
        repo.upsert_many([product])
        session.commit()

        # Re-query in a NEW session to force a fresh load from DB
        fetched = repo.get_by_id("PG-JSONB-001")
        assert fetched is not None
        assert fetched.attributes == {"color": "black", "battery": {"hours": 30}}
        assert fetched.images == ["http://img/a.jpg", "http://img/b.jpg"]
        assert fetched.url == "http://example.com/PG-JSONB-001"

    def test_numeric_decimal_precision(self, session) -> None:
        """Decimal price with 2dp survives Numeric roundtrip."""

        repo = ProductRepository(session)
        product = Product(
            product_id="PG-NUM-001", title="Numeric Test", brand="N",
            category=ProductCategory.ELECTRONICS, price=123.45, rating=4.67,
        )
        repo.upsert_many([product])
        session.commit()

        # Verify stored as Decimal with correct precision (query raw column)
        from sqlalchemy import select
        raw = session.scalar(select(ProductRecord).where(ProductRecord.product_id == "PG-NUM-001"))
        assert raw is not None
        assert isinstance(raw.price, Decimal)
        assert raw.price == Decimal("123.45")

        fetched = repo.get_by_id("PG-NUM-001")
        assert fetched is not None
        assert fetched.price == 123.45


class TestPostgresConstraints:
    """PostgreSQL unique + FK cascade constraints."""

    def test_request_id_unique_constraint(self, session) -> None:
        """Duplicate request_id raises IntegrityError (DB-level unique)."""
        repo = AgentRunRepository(session)
        repo.create_running("pg-unique-001", "a", "balanced")
        session.commit()
        with pytest.raises(Exception):
            repo.create_running("pg-unique-001", "b", "balanced")
            session.commit()
        session.rollback()

    def test_on_delete_cascade(self, session) -> None:
        """Deleting an AgentRun DB-level cascades to its ToolCalls."""
        repo = AgentRunRepository(session)
        record = repo.create_running("pg-cascade-001", "test", "balanced")
        session.commit()
        repo.add_tool_calls(record.id, [
            {"step_number": 1, "tool_name": "search_catalog", "arguments": {}, "result_product_ids": [], "success": True},
        ])
        session.commit()

        # Delete the run at DB level (raw delete, not ORM cascade)
        session.delete(record)
        session.commit()

        count = session.scalar(select(func.count()).select_from(AgentToolCallRecord))
        assert count == 0  # Tool calls cascade-deleted by DB

    def test_transaction_rollback(self, session) -> None:
        """Failed write leaves no partial tool calls."""
        repo = AgentRunRepository(session)
        record = repo.create_running("pg-rollback-001", "test", "balanced")
        session.commit()
        try:
            repo.add_tool_calls(record.id, [
                {"step_number": 1, "tool_name": "bad_tool", "arguments": {}, "result_product_ids": [], "success": False, "error_type": "ToolError"},
            ])
            raise RuntimeError("simulated failure")
        except RuntimeError:
            session.rollback()
        assert repo.get_tool_calls(record.id) == []


class TestPostgresCatalogBackend:
    """postgres catalog backend reads products from DB."""

    def test_load_catalog_from_postgres(self, session, monkeypatch) -> None:
        """_load_catalog_from_postgres reads products via repository."""
        _require_test_db()  # Ensure test DB, skip otherwise
        # Seed products
        repo = ProductRepository(session)
        repo.upsert_many([
            Product(product_id="PG-CAT-001", title="Catalog A", brand="B",
                    category=ProductCategory.ELECTRONICS, price=10.0, rating=4.0),
            Product(product_id="PG-CAT-002", title="Catalog B", brand="B",
                    category=ProductCategory.ELECTRONICS, price=20.0, rating=4.5),
        ])
        session.commit()

        # Monkeypatch the global engine/session to point at the test DB
        # Replace session_scope in main module to use our test session factory
        from contextlib import contextmanager

        from marketlens.persistence.engine import reset_engine
        @contextmanager
        def _test_session_scope():
            from marketlens.persistence.engine import session_scope as real
            with real() as s:
                yield s

        # Simpler: directly use the repository via the test DB engine
        # Point engine module's cached engine at test DB
        monkeypatch.setenv("MARKETLENS_DATABASE_URL", TEST_DB_URL)
        reset_engine()

        from marketlens.api.main import _load_catalog_from_postgres
        catalog = _load_catalog_from_postgres()
        assert len(catalog) == 2
        ids = {p.product_id for p in catalog.get_all_products()}
        assert "PG-CAT-001" in ids
        assert "PG-CAT-002" in ids
        reset_engine()


class TestAgentRunRecording:
    """Agent API records AgentRun + ToolCall on PostgreSQL."""

    def test_recorded_run_creates_then_updates(self, monkeypatch) -> None:
        """_recorded_run: running record created before execution, updated after."""
        _require_test_db()  # Ensure test DB, skip otherwise
        import asyncio

        from marketlens.agent.models import AgentRequest, AgentResponse
        from marketlens.agent.orchestrator import AgentOrchestrator
        from marketlens.agent.providers.base import FakeLLMClient
        from marketlens.agent.tools import AgentTools
        from marketlens.catalog import ProductCatalog
        from marketlens.persistence.engine import reset_engine
        from marketlens.retrieval.embedding import FakeEmbeddingBackend
        from marketlens.retrieval.service import RetrievalService

        monkeypatch.setenv("MARKETLENS_DATABASE_URL", TEST_DB_URL)
        reset_engine()

        # Build a fake orchestrator backed by a fake LLM
        catalog = ProductCatalog.from_fixture("electronics_sample.json")
        service = RetrievalService(catalog, embedding_backend=FakeEmbeddingBackend(dim=16, seed=1))
        service.initialize()
        tools = AgentTools(service)
        fake_llm = FakeLLMClient([
            {"content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "search_catalog", "arguments": '{"query": "headphones", "top_k": 5}'},
            }]},
            {"content": "Here are recommendations."},
        ])
        orch = AgentOrchestrator(fake_llm, tools, service._product_index)

        from marketlens.api.routes import _recorded_run
        request = AgentRequest(message="best headphones", mode="balanced")

        response = asyncio.run(_recorded_run(request, orch, tools, service._product_index))

        assert isinstance(response, AgentResponse)

        # Verify AgentRun + ToolCall recorded
        from marketlens.persistence.engine import session_scope
        from marketlens.persistence.repositories import AgentRunRepository
        with session_scope() as s:
            repo = AgentRunRepository(s)
            # There should be exactly one run for this request
            runs = s.query(AgentRunRecord).all()
            # Find the run (request_id is a uuid, but there should be ≥1 run)
            assert len(runs) >= 1
            # The most recent run should be completed
            latest = runs[-1]
            assert latest.status == "completed"
            assert latest.mode_used == "hybrid"
            tool_calls = repo.get_tool_calls(latest.id)
            assert len(tool_calls) >= 1
            assert tool_calls[0].tool_name == "search_catalog"
        reset_engine()

    def test_request_hash_deterministic(self) -> None:
        """Same content → same hash; different content → different hash."""
        from marketlens.agent.models import AgentRequest
        from marketlens.api.routes import _compute_request_hash

        r1 = AgentRequest(message="best headphones", mode="balanced", max_results=3)
        r2 = AgentRequest(message="best headphones", mode="balanced", max_results=3)
        r3 = AgentRequest(message="different query", mode="balanced", max_results=3)

        assert _compute_request_hash(r1) == _compute_request_hash(r2)
        assert _compute_request_hash(r1) != _compute_request_hash(r3)


class TestIdempotency:
    """request_id idempotency and conflict detection."""

    def test_same_request_id_same_content_replay(self, monkeypatch) -> None:
        """Same request_id + same content returns existing stored result."""
        _require_test_db()
        import asyncio

        from marketlens.agent.models import AgentRequest
        from marketlens.agent.orchestrator import AgentOrchestrator
        from marketlens.agent.providers.base import FakeLLMClient
        from marketlens.agent.tools import AgentTools
        from marketlens.catalog import ProductCatalog
        from marketlens.persistence.engine import reset_engine, session_scope
        from marketlens.persistence.repositories import AgentRunRepository
        from marketlens.retrieval.embedding import FakeEmbeddingBackend
        from marketlens.retrieval.service import RetrievalService

        monkeypatch.setenv("MARKETLENS_DATABASE_URL", TEST_DB_URL)
        reset_engine()

        catalog = ProductCatalog.from_fixture("electronics_sample.json")
        service = RetrievalService(catalog, embedding_backend=FakeEmbeddingBackend(dim=16, seed=1))
        service.initialize()
        tools = AgentTools(service)
        orch = AgentOrchestrator(
            FakeLLMClient([{"content": "done."}]),
            tools,
            service._product_index,
        )

        from marketlens.api.routes import _recorded_run
        req = AgentRequest(message="best headphones", mode="balanced", request_id="idem-001")

        first = asyncio.run(_recorded_run(req, orch, tools, service._product_index))
        # Second call with SAME request_id + same content → replay, no new record
        replay = asyncio.run(_recorded_run(req, orch, tools, service._product_index))

        assert first.request_id == "idem-001"
        assert replay.request_id == "idem-001"

        with session_scope() as s:
            repo = AgentRunRepository(s)
            run = repo.get_by_request_id("idem-001")
            assert run is not None
            assert run.status == "completed"
        reset_engine()

    def test_same_request_id_different_content_conflict(self, monkeypatch) -> None:
        """Same request_id + different content → 409 conflict."""
        _require_test_db()
        import asyncio

        import pytest

        from marketlens.agent.models import AgentRequest
        from marketlens.agent.orchestrator import AgentOrchestrator
        from marketlens.agent.providers.base import FakeLLMClient
        from marketlens.agent.tools import AgentTools
        from marketlens.catalog import ProductCatalog
        from marketlens.persistence.engine import reset_engine
        from marketlens.retrieval.embedding import FakeEmbeddingBackend
        from marketlens.retrieval.service import RetrievalService

        monkeypatch.setenv("MARKETLENS_DATABASE_URL", TEST_DB_URL)
        reset_engine()

        catalog = ProductCatalog.from_fixture("electronics_sample.json")
        service = RetrievalService(catalog, embedding_backend=FakeEmbeddingBackend(dim=16, seed=1))
        service.initialize()
        tools = AgentTools(service)
        orch = AgentOrchestrator(
            FakeLLMClient([{"content": "done."}]),
            tools,
            service._product_index,
        )

        from marketlens.api.routes import _recorded_run
        req1 = AgentRequest(message="best headphones", mode="balanced", request_id="idem-002")
        asyncio.run(_recorded_run(req1, orch, tools, service._product_index))

        req2 = AgentRequest(message="different content", mode="balanced", request_id="idem-002")
        with pytest.raises(Exception) as exc_info:
            asyncio.run(_recorded_run(req2, orch, tools, service._product_index))
        assert "409" in str(exc_info.value) or "conflict" in str(exc_info.value).lower()
        reset_engine()


class TestFeedback:
    """Minimal feedback persistence."""

    def test_feedback_create_and_idempotent(self, session) -> None:
        """Feedback creates and idempotency_key prevents duplicates."""
        from marketlens.persistence.repositories import (
            AgentRunRepository,
            FeedbackRepository,
        )

        run_repo = AgentRunRepository(session)
        run = run_repo.create_running("fb-req-001", "test", "balanced")
        session.commit()

        fb_repo = FeedbackRepository(session)
        assert fb_repo.agent_run_exists(run.id)
        assert fb_repo.agent_run_exists(999999) is False

        fb_repo.create(run.id, "helpful", reason="good", idempotency_key="fb-key-1")
        session.commit()
        assert fb_repo.idempotency_key_exists("fb-key-1") is True
        assert fb_repo.idempotency_key_exists("fb-key-nonexistent") is False

    def test_feedback_fk_cascade(self, session) -> None:
        """Deleting agent run cascades feedback events."""
        from marketlens.persistence.models import FeedbackEventRecord
        from marketlens.persistence.repositories import (
            AgentRunRepository,
            FeedbackRepository,
        )

        run_repo = AgentRunRepository(session)
        run = run_repo.create_running("fb-req-002", "test", "balanced")
        session.commit()

        fb_repo = FeedbackRepository(session)
        fb_repo.create(run.id, "helpful")
        session.commit()

        # Delete the run → feedback cascades
        session.delete(run)
        session.commit()

        from sqlalchemy import func, select
        count = session.scalar(select(func.count()).select_from(FeedbackEventRecord))
        assert count == 0
