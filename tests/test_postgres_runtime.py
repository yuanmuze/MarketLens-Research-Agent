"""Real PostgreSQL/pgvector end-to-end tests for the active service path."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from marketlens.agent.models import AgentRequest
from marketlens.agent.orchestrator import AgentOrchestrator
from marketlens.agent.providers.base import FakeLLMClient
from marketlens.agent.tools import AgentTools
from marketlens.api.routes import _recorded_run
from marketlens.catalog import ProductCatalog
from marketlens.models import Product
from marketlens.persistence.engine import reset_engine, session_scope
from marketlens.persistence.repositories import (
    AgentRunRepository,
    ProductEmbeddingRepository,
    ProductRepository,
)
from marketlens.retrieval.embedding import SentenceTransformersBackend
from marketlens.retrieval.reranker import CrossEncoderReranker
from marketlens.retrieval.service import RetrievalService, _build_search_text

TEST_DB_URL = os.environ.get("MARKETLENS_TEST_DATABASE_URL", "")
MODEL_NAME = "all-MiniLM-L6-v2"

pytestmark = pytest.mark.postgres


def _require_test_db() -> str:
    if not TEST_DB_URL:
        pytest.skip("MARKETLENS_TEST_DATABASE_URL is not set")
    assert TEST_DB_URL.startswith("postgresql")
    database = TEST_DB_URL.rsplit("/", 1)[-1].split("?", 1)[0]
    assert "test" in database.lower(), f"refusing non-test database: {database}"
    return TEST_DB_URL


@pytest.fixture(scope="module")
def real_backend() -> SentenceTransformersBackend:
    backend = SentenceTransformersBackend(MODEL_NAME, batch_size=8, normalize=True)
    assert backend.dim == 384
    return backend


@pytest.fixture
def pg_catalog(
    real_backend: SentenceTransformersBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[ProductCatalog, sessionmaker[Session]]]:
    url = _require_test_db()
    monkeypatch.setenv("MARKETLENS_DATABASE_URL", url)
    reset_engine()
    engine = create_engine(url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM feedback_events"))
        connection.execute(text("DELETE FROM agent_tool_calls"))
        connection.execute(text("DELETE FROM agent_runs"))
        connection.execute(text("DELETE FROM product_embeddings"))
        connection.execute(text("DELETE FROM products"))

    products = [
        Product(
            product_id="P8-001",
            title="Wireless noise cancelling headphones",
            brand="Sony",
            price=299.0,
            rating=4.7,
            review_count=100,
        ),
        Product(
            product_id="P8-002",
            title="Budget Bluetooth earbuds",
            brand="Anker",
            price=49.0,
            rating=4.2,
            review_count=200,
        ),
        Product(
            product_id="P8-003",
            title="Studio monitor speakers",
            brand="Yamaha",
            price=199.0,
            rating=4.5,
            review_count=50,
        ),
    ]
    catalog = ProductCatalog(products)
    product_dicts = [product.model_dump() for product in products]
    vectors = real_backend.encode([_build_search_text(item) for item in product_dicts])
    with factory() as session:
        ProductRepository(session).upsert_many(products)
        session.flush()
        ProductEmbeddingRepository(session).upsert_many(
            [product.product_id for product in products],
            vectors.tolist(),
            MODEL_NAME,
            384,
        )
        session.commit()

    yield catalog, factory

    reset_engine()
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM feedback_events"))
        connection.execute(text("DELETE FROM agent_tool_calls"))
        connection.execute(text("DELETE FROM agent_runs"))
        connection.execute(text("DELETE FROM product_embeddings"))
        connection.execute(text("DELETE FROM products"))
    engine.dispose()


def _pg_service(
    catalog: ProductCatalog,
    factory: sessionmaker[Session],
    backend: SentenceTransformersBackend,
    *,
    quality: bool = False,
) -> RetrievalService:
    return RetrievalService(
        catalog,
        embedding_backend=backend,
        semantic_backend="pgvector",
        session_factory=factory,
        embedding_model_name=MODEL_NAME,
        reranker=CrossEncoderReranker() if quality else None,
    ).initialize()


def test_service_pgvector_semantic_and_hybrid_are_real_sql_paths(
    pg_catalog: tuple[ProductCatalog, sessionmaker[Session]],
    real_backend: SentenceTransformersBackend,
) -> None:
    catalog, factory = pg_catalog
    service = _pg_service(catalog, factory, real_backend)

    semantic = service.search("noise cancelling headphones", strategy="embedding", top_k=2)
    hybrid = service.search("noise cancelling headphones", strategy="hybrid", top_k=2)

    assert semantic.semantic_backend == "pgvector"
    assert semantic.results[0].product_id == "P8-001"
    assert hybrid.results[0].product_id == "P8-001"
    assert hybrid.results[0].bm25_score is not None
    assert hybrid.results[0].embedding_score is not None
    assert service._memory_embedding is None


def test_memory_and_pgvector_fixed_fixture_topk_match(
    pg_catalog: tuple[ProductCatalog, sessionmaker[Session]],
    real_backend: SentenceTransformersBackend,
) -> None:
    catalog, factory = pg_catalog
    memory = RetrievalService(catalog, embedding_backend=real_backend).initialize()
    pgvector = _pg_service(catalog, factory, real_backend)

    memory_ids = [
        item.product_id
        for item in memory.search("budget wireless earbuds", strategy="embedding", top_k=3).results
    ]
    pgvector_ids = [
        item.product_id
        for item in pgvector.search("budget wireless earbuds", strategy="embedding", top_k=3).results
    ]

    assert pgvector_ids == memory_ids


def test_pgvector_hybrid_quality_uses_real_cross_encoder(
    pg_catalog: tuple[ProductCatalog, sessionmaker[Session]],
    real_backend: SentenceTransformersBackend,
) -> None:
    catalog, factory = pg_catalog
    service = _pg_service(catalog, factory, real_backend, quality=True)

    output = service.search(
        "wireless headphones",
        strategy="rerank",
        top_k=2,
        candidate_k=3,
    )

    assert output.results
    assert all(item.reranker_score is not None for item in output.results)
    assert service.status()["reranker_backend"].startswith("CrossEncoder(")


def test_fake_llm_agent_uses_pgvector_hybrid_and_persists_tool_call(
    pg_catalog: tuple[ProductCatalog, sessionmaker[Session]],
    real_backend: SentenceTransformersBackend,
) -> None:
    catalog, factory = pg_catalog
    service = _pg_service(catalog, factory, real_backend)
    tools = AgentTools(service)
    llm = FakeLLMClient([
        {
            "content": None,
            "tool_calls": [{
                "id": "postgres-tool-1",
                "type": "function",
                "function": {
                    "name": "search_catalog",
                    "arguments": '{"query":"noise cancelling headphones","mode":"balanced","top_k":2}',
                },
            }],
        },
        {"content": "The retrieved catalog evidence supports these options."},
    ])
    product_index = {
        product.product_id: product.model_dump()
        for product in catalog.get_all_products()
    }
    orchestrator = AgentOrchestrator(llm, tools, product_index)
    request = AgentRequest(
        request_id="postgres-pgvector-agent",
        message="Find noise cancelling headphones",
        mode="balanced",
        max_results=2,
    )

    response = asyncio.run(_recorded_run(request, orchestrator, tools, product_index))

    assert response.mode_used == "hybrid"
    with session_scope() as session:
        repo = AgentRunRepository(session)
        run = repo.get_by_request_id("postgres-pgvector-agent")
        assert run is not None
        calls = repo.get_tool_calls(run.id)
        assert run.status in {"completed", "no_results"}
        assert len(calls) == 1
        assert calls[0].tool_name == "search_catalog"
        assert calls[0].success is True
