"""Phase 8 API configuration, dependency injection, and lifespan tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import marketlens.api.main as main_module
import marketlens.api.routes as routes
from marketlens.agent.providers.base import FakeLLMClient
from marketlens.api.main import app
from marketlens.catalog import ProductCatalog
from marketlens.config import MarketLensSettings


@pytest.fixture(autouse=True)
def restore_route_globals() -> Iterator[None]:
    saved = (
        routes._catalog,
        routes._service,
        routes._settings,
        routes._startup_error,
    )
    yield
    (
        routes._catalog,
        routes._service,
        routes._settings,
        routes._startup_error,
    ) = saved


def test_settings_parse_supported_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKETLENS_CATALOG_BACKEND", "postgres")
    monkeypatch.setenv("MARKETLENS_SEMANTIC_BACKEND", "pgvector")
    monkeypatch.setenv("MARKETLENS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setenv("MARKETLENS_USE_FAKE_EMBEDDINGS", "false")
    monkeypatch.setenv("MARKETLENS_USE_FAKE_LLM", "true")
    monkeypatch.setenv("MARKETLENS_CATALOG_PATH", "/data/processed/products.json")

    settings = MarketLensSettings.from_env()

    assert settings.catalog_backend == "postgres"
    assert settings.semantic_backend == "pgvector"
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.use_fake_embeddings is False
    assert settings.use_fake_llm is True
    assert settings.catalog_path == Path("/data/processed/products.json")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("MARKETLENS_CATALOG_BACKEND", "unknown"),
        ("MARKETLENS_SEMANTIC_BACKEND", "fallback"),
        ("MARKETLENS_USE_FAKE_EMBEDDINGS", "sometimes"),
        ("MARKETLENS_USE_FAKE_LLM", "sometimes"),
    ],
)
def test_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        MarketLensSettings.from_env()


def test_pgvector_session_factory_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def session_factory():
        return None

    class CapturingService:
        def __init__(self, _catalog, **kwargs):
            captured.update(kwargs)

        def initialize(self):
            return self

    monkeypatch.setattr(routes, "RetrievalService", CapturingService)
    settings = MarketLensSettings(
        catalog_backend="postgres",
        semantic_backend="pgvector",
    )

    initialized = routes.init_catalog(
        ProductCatalog(),
        settings=settings,
        session_factory=session_factory,
    )

    assert initialized is True
    assert captured["semantic_backend"] == "pgvector"
    assert captured["session_factory"] is session_factory
    assert captured["embedding_model_name"] == "all-MiniLM-L6-v2"
    assert captured["use_fake_embeddings"] is False


def test_fake_llm_is_explicit_and_deterministic() -> None:
    routes._settings = MarketLensSettings(use_fake_llm=True)

    client = routes._build_llm_client()
    first = client.send([], [])
    second = client.send([], [])

    assert isinstance(client, FakeLLMClient)
    assert client.model_name == "phase8-deterministic-fake"
    assert first["tool_calls"][0]["function"]["name"] == "search_catalog"
    assert second["content"].startswith("Here are evidence-backed")


def test_pgvector_rejects_fake_configuration() -> None:
    settings = MarketLensSettings(
        semantic_backend="pgvector",
        use_fake_embeddings=True,
    )
    assert routes.init_catalog(ProductCatalog(), settings=settings) is False
    assert routes._service is None
    assert "fake embeddings are not allowed" in routes._startup_error


def test_request_id_propagates_to_search_response() -> None:
    settings = MarketLensSettings(use_fake_embeddings=True)
    catalog = ProductCatalog.from_fixture("electronics_sample.json")
    assert routes.init_catalog(catalog, settings=settings)
    client = TestClient(app)

    response = client.get(
        "/search?q=headphones&top_k=1",
        headers={"X-Request-ID": "phase8-request-id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "phase8-request-id"
    assert response.json()["request_id"] == "phase8-request-id"


def test_unavailable_service_returns_structured_503() -> None:
    routes.mark_startup_unavailable("retrieval unavailable")
    client = TestClient(app)

    response = client.get(
        "/search?q=headphones",
        headers={"X-Request-ID": "phase8-unavailable"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "code": "service_unavailable",
        "message": "retrieval unavailable",
        "request_id": "phase8-unavailable",
    }


def test_ready_reports_actual_memory_backend() -> None:
    settings = MarketLensSettings(use_fake_embeddings=True)
    catalog = ProductCatalog.from_fixture("electronics_sample.json")
    assert routes.init_catalog(catalog, settings=settings)

    response = TestClient(app).get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["catalog_backend"] == "json"
    assert body["semantic_backend"] == "memory"
    assert body["semantic_index_ready"] is True
    assert body["semantic_indexed_count"] == 20


def test_postgres_catalog_lifespan_yields_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    settings = MarketLensSettings(
        catalog_backend="postgres",
        semantic_backend="memory",
        use_fake_embeddings=True,
        catalog_path=Path("/data/processed/products.json"),
    )
    monkeypatch.setattr(
        main_module.MarketLensSettings,
        "from_env",
        classmethod(lambda cls: settings),
    )
    monkeypatch.setattr(main_module, "init_db", lambda: calls.append("db"))
    monkeypatch.setattr(
        main_module,
        "_load_catalog_from_postgres",
        lambda: ProductCatalog.from_fixture("electronics_sample.json"),
    )

    captured: dict = {}

    def fake_init_catalog(catalog, **kwargs):
        calls.append(f"catalog:{len(catalog)}")
        captured.update(kwargs)
        return True

    monkeypatch.setattr(main_module, "init_catalog", fake_init_catalog)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200

    assert calls == ["db", "catalog:20"]
    assert captured["data_path"] == Path("/data/processed/products.json")


def test_liveness_survives_startup_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "init_db", lambda: (_ for _ in ()).throw(OSError()))

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get(
            "/health/ready",
            headers={"X-Request-ID": "phase8-not-ready"},
        )

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["code"] == "service_unavailable"
    assert ready.json()["request_id"] == "phase8-not-ready"
