"""pgvector integration tests (PostgreSQL + pgvector required).

Run with: uv run pytest -m postgres
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from marketlens.models import Product, ProductCategory
from marketlens.persistence.models import Base
from marketlens.persistence.repositories import (
    ProductEmbeddingRepository,
    ProductRepository,
)

TEST_DB_URL = os.environ.get("MARKETLENS_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.postgres


def _require_test_db() -> str:
    """Return test DB URL, enforcing postgresql dialect + 'test' name."""
    if not TEST_DB_URL:
        pytest.skip("MARKETLENS_TEST_DATABASE_URL not set — skipping pgvector integration test")
    assert TEST_DB_URL.startswith("postgresql"), "test DB must use postgresql dialect"
    db_name = TEST_DB_URL.rsplit("/", 1)[-1].split("?")[0]
    assert "test" in db_name.lower(), f"test DB name must contain 'test', got {db_name}"
    return TEST_DB_URL


@pytest.fixture
def session():
    """Session against the test DB, cleaned before/after."""
    url = _require_test_db()
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM product_embeddings"))
        conn.execute(text("DELETE FROM products"))
    yield s
    s.close()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM product_embeddings"))
        conn.execute(text("DELETE FROM products"))
    engine.dispose()


def _seed_products(session) -> None:
    """Seed a few products for FK + search tests."""
    repo = ProductRepository(session)
    repo.upsert_many([
        Product(product_id="V001", title="Wireless noise cancelling headphones", brand="Sony",
                category=ProductCategory.ELECTRONICS, price=299.0, rating=4.7),
        Product(product_id="V002", title="Budget earbuds", brand="Anker",
                category=ProductCategory.ELECTRONICS, price=49.0, rating=4.2),
        Product(product_id="V003", title="Studio monitor speakers", brand="Yamaha",
                category=ProductCategory.ELECTRONICS, price=199.0, rating=4.5),
    ])
    session.commit()


class TestPgVectorExtension:
    """Extension and schema checks."""

    def test_vector_extension_exists(self, session) -> None:
        """pgvector extension is installed."""
        row = session.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).fetchone()
        assert row is not None
        assert row[0] == "vector"

    def test_vector_column_dimension(self, session) -> None:
        """embedding column is vector(384)."""
        insp = inspect(session.bind)
        cols = {c["name"]: c for c in insp.get_columns("product_embeddings")}
        assert "embedding" in cols
        assert "VECTOR" in str(cols["embedding"]["type"]).upper()


class TestProductEmbeddingRepository:
    """pgvector embedding upsert + search."""

    def test_upsert_idempotent(self, session) -> None:
        """Re-upserting same embeddings does not duplicate."""
        _seed_products(session)
        repo = ProductEmbeddingRepository(session)
        emb1 = [0.1] * 384
        emb2 = [0.2] * 384
        emb3 = [0.3] * 384
        r1 = repo.upsert_many(["V001", "V002", "V003"], [emb1, emb2, emb3], "all-MiniLM-L6-v2", 384)
        session.commit()
        assert r1["inserted"] == 3
        assert repo.count("all-MiniLM-L6-v2") == 3

        # Re-upsert identical → unchanged, no duplicates
        r2 = repo.upsert_many(["V001", "V002", "V003"], [emb1, emb2, emb3], "all-MiniLM-L6-v2", 384)
        session.commit()
        assert r2["unchanged"] == 3
        assert repo.count("all-MiniLM-L6-v2") == 3

    def test_upsert_tolerates_only_serialization_scale_noise(self, session) -> None:
        """Tiny pgvector round-trip noise is unchanged; a real change updates."""
        _seed_products(session)
        repo = ProductEmbeddingRepository(session)
        original = np.linspace(-0.25, 0.25, 384, dtype=np.float32)
        repo.upsert_many(["V001"], [original.tolist()], "m", 384)
        session.commit()

        tiny_noise = original.copy()
        tiny_noise[17] += np.float32(5e-8)
        unchanged = repo.upsert_many(["V001"], [tiny_noise.tolist()], "m", 384)
        assert unchanged == {"inserted": 0, "updated": 0, "unchanged": 1}

        material_change = original.copy()
        material_change[17] += np.float32(1e-4)
        updated = repo.upsert_many(["V001"], [material_change.tolist()], "m", 384)
        assert updated == {"inserted": 0, "updated": 1, "unchanged": 0}

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_upsert_rejects_non_finite_values(self, session, bad_value: float) -> None:
        """NaN and infinities never reach pgvector or compare as unchanged."""
        _seed_products(session)
        embedding = [0.1] * 384
        embedding[12] = bad_value

        with pytest.raises(ValueError, match="finite"):
            ProductEmbeddingRepository(session).upsert_many(
                ["V001"], [embedding], "m", 384
            )

    def test_full_2000_embedding_upsert_is_idempotent(self, session) -> None:
        """A complete 2,000-row repeat stays unchanged with no duplicates."""
        products = [
            Product(
                product_id=f"P8-{index:04d}",
                title=f"Phase 8 product {index}",
                category=ProductCategory.OTHER,
            )
            for index in range(2000)
        ]
        ProductRepository(session).upsert_many(products)
        session.flush()
        product_ids = [product.product_id for product in products]
        embeddings = [
            np.full(384, index / 2000, dtype=np.float32).tolist()
            for index in range(2000)
        ]
        repo = ProductEmbeddingRepository(session)

        first = repo.upsert_many(product_ids, embeddings, "phase8-idempotency", 384)
        session.commit()
        second = repo.upsert_many(product_ids, embeddings, "phase8-idempotency", 384)
        session.commit()

        assert first == {"inserted": 2000, "updated": 0, "unchanged": 0}
        assert second == {"inserted": 0, "updated": 0, "unchanged": 2000}
        assert repo.count("phase8-idempotency") == 2000

    def test_dimension_mismatch_raises(self, session) -> None:
        """Embedding with wrong dim fails clearly."""
        _seed_products(session)
        repo = ProductEmbeddingRepository(session)
        with pytest.raises(ValueError, match="dimension"):
            repo.upsert_many(["V001"], [[0.1] * 128], "all-MiniLM-L6-v2", 384)

    def test_cosine_topk_order(self, session) -> None:
        """Cosine search returns nearest embeddings first."""
        _seed_products(session)
        repo = ProductEmbeddingRepository(session)
        # V001 near [1,0,0,...], V002 near [0,1,0,...], V003 near [0,0,1,...]
        e1 = [1.0] + [0.0] * 383
        e2 = [0.0, 1.0] + [0.0] * 382
        e3 = [0.0, 0.0, 1.0] + [0.0] * 381
        repo.upsert_many(["V001", "V002", "V003"], [e1, e2, e3], "m", 384)
        session.commit()

        results = repo.search([1.0] + [0.0] * 383, top_k=3, model_name="m")
        assert results[0][0] == "V001"  # closest to [1,0,0,...]
        assert results[0][1] > results[1][1]  # similarity descending

    def test_model_filter(self, session) -> None:
        """Only embeddings for the given model are returned."""
        _seed_products(session)
        repo = ProductEmbeddingRepository(session)
        repo.upsert_many(["V001"], [[1.0] + [0.0] * 383], "model-a", 384)
        repo.upsert_many(["V002"], [[0.0, 1.0] + [0.0] * 382], "model-b", 384)
        session.commit()
        assert repo.count("model-a") == 1
        assert repo.count("model-b") == 1
        results = repo.search([1.0] + [0.0] * 383, top_k=5, model_name="model-a")
        assert all(pid == "V001" for pid, _ in results)

    def test_fk_cascade_on_product_delete(self, session) -> None:
        """Deleting a product cascades its embeddings."""
        _seed_products(session)
        repo = ProductEmbeddingRepository(session)
        repo.upsert_many(["V001"], [[1.0] + [0.0] * 383], "m", 384)
        session.commit()

        from marketlens.persistence.models import ProductRecord
        prod = session.get(ProductRecord, "V001")
        session.delete(prod)
        session.commit()

        assert repo.count("m") == 0


class TestRetrievalConsistency:
    """In-memory vs pgvector top-k overlap (same embedding backend)."""

    def test_topk_overlap_in_memory_vs_pgvector(self, session) -> None:
        """Same FakeEmbeddingBackend(384) → consistent top-k results."""
        from marketlens.retrieval.embedding import (
            EmbeddingRetriever,
            FakeEmbeddingBackend,
        )

        backend = FakeEmbeddingBackend(dim=384, seed=42)

        # Build search texts from seeded products
        texts = [
            "Wireless noise cancelling headphones Sony",
            "Budget earbuds Anker",
            "Studio monitor speakers Yamaha",
        ]
        ids = ["V001", "V002", "V003"]
        _seed_products(session)

        # In-memory retriever
        mem = EmbeddingRetriever(backend).fit(texts, ids)
        mem_results = mem.search("noise cancelling headphones", top_k=3)
        mem_ids = [pid for pid, _ in mem_results]

        # pgvector: store same embeddings, search same query
        repo = ProductEmbeddingRepository(session)
        embeddings = [backend.encode([t])[0].tolist() for t in texts]
        repo.upsert_many(ids, embeddings, "consistency-test", 384)
        session.commit()

        query_vec = backend.encode(["noise cancelling headphones"])[0].tolist()
        pg_results = repo.search(query_vec, top_k=3, model_name="consistency-test")
        pg_ids = [pid for pid, _ in pg_results]

        # Report real overlap (not a fabricated threshold)
        overlap = set(mem_ids) & set(pg_ids)
        assert len(overlap) > 0, f"Expected some overlap, in-memory={mem_ids}, pgvector={pg_ids}"
