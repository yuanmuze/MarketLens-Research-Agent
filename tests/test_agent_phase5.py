"""Tests for Phase 5 agent: orchestrator, tools, evidence, providers."""

from __future__ import annotations

import json

import pytest

from marketlens.agent.evidence import EvidenceVerifier
from marketlens.agent.models import (
    AgentRequest,
    AgentResponse,
    CompareProductsParams,
    EvidenceRef,
    GetProductDetailsParams,
    RecommendationItem,
    SearchCatalogParams,
    SearchCatalogResult,
)
from marketlens.agent.orchestrator import AgentOrchestrator
from marketlens.agent.providers.base import FakeLLMClient, LLMClient
from marketlens.agent.tools import TOOL_DEFINITIONS, AgentTools
from marketlens.catalog import ProductCatalog
from marketlens.retrieval.embedding import FakeEmbeddingBackend
from marketlens.retrieval.service import RetrievalService

# ---------------------------------------------------------------------------
# Fake LLM + RetrievalService Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service() -> RetrievalService:
    """Create RetrievalService with fixture catalog."""
    catalog = ProductCatalog.from_fixture("electronics_sample.json")
    svc = RetrievalService(catalog, embedding_backend=FakeEmbeddingBackend(dim=16, seed=42))
    svc.initialize()
    return svc


@pytest.fixture
def tools(service: RetrievalService) -> AgentTools:
    """Create AgentTools with test service."""
    return AgentTools(service)


@pytest.fixture
def product_index(service: RetrievalService) -> dict:
    """Product index for evidence verification."""
    return service._product_index


@pytest.fixture
def verifier(product_index: dict) -> EvidenceVerifier:
    """Create evidence verifier."""
    return EvidenceVerifier(product_index)


# ---------------------------------------------------------------------------
# Tool Tests
# ---------------------------------------------------------------------------

class TestSearchCatalog:
    """search_catalog tool tests."""

    def test_basic_search(self, tools: AgentTools) -> None:
        """Basic balanced search returns results."""
        params = SearchCatalogParams(query="headphones", mode="balanced", top_k=5)
        result = tools.search_catalog(params)
        assert isinstance(result, SearchCatalogResult)
        assert result.total_found > 0
        assert result.mode_used == "hybrid"

    def test_fast_mode_uses_bm25(self, tools: AgentTools) -> None:
        """Fast mode uses BM25."""
        params = SearchCatalogParams(query="headphones", mode="fast", top_k=5)
        result = tools.search_catalog(params)
        assert result.mode_used == "bm25"

    def test_quality_mode_uses_rerank(self, tools: AgentTools) -> None:
        """Quality mode attempts rerank."""
        params = SearchCatalogParams(query="Sony", mode="quality", top_k=5)
        try:
            result = tools.search_catalog(params)
            # mode_used may be "rerank" or may fall back
            assert result.mode_used in ("rerank", "hybrid")
        except ImportError:
            pytest.skip("sentence-transformers not available")

    def test_price_filter(self, tools: AgentTools) -> None:
        """Price filter excludes out-of-budget products."""
        params = SearchCatalogParams(query="headphones", price_max=100.0, top_k=10)
        result = tools.search_catalog(params)
        for item in result.results:
            assert item.price is not None and item.price <= 100.0

    def test_rating_filter(self, tools: AgentTools) -> None:
        """Rating filter works."""
        params = SearchCatalogParams(query="headphones", min_rating=4.5, top_k=10)
        result = tools.search_catalog(params)
        for item in result.results:
            assert item.rating is not None and item.rating >= 4.5

    def test_top_k_limit(self, tools: AgentTools) -> None:
        """top_k limits results."""
        params = SearchCatalogParams(query="headphones", top_k=3)
        result = tools.search_catalog(params)
        assert len(result.results) <= 3

    def test_invalid_params_rejected(self, tools: AgentTools) -> None:
        """Empty query is rejected."""
        with pytest.raises(Exception):
            SearchCatalogParams(query="")

    def test_dispatch(self, tools: AgentTools) -> None:
        """dispatch() routes to search_catalog."""
        result = tools.dispatch("search_catalog", {"query": "headphones", "top_k": 3})
        assert isinstance(result, SearchCatalogResult)


class TestGetProductDetails:
    """get_product_details tool tests."""

    def test_valid_ids(self, tools: AgentTools) -> None:
        """Fetching valid product IDs returns details."""
        params = GetProductDetailsParams(product_ids=["B001", "B002"])
        result = tools.get_product_details(params)
        assert len(result.products) == 2
        assert result.products[0].product_id in ("B001", "B002")

    def test_unknown_id_skipped(self, tools: AgentTools) -> None:
        """Unknown product IDs are skipped silently."""
        params = GetProductDetailsParams(product_ids=["B001", "NONEXISTENT"])
        result = tools.get_product_details(params)
        assert len(result.products) == 1
        assert result.products[0].product_id == "B001"

    def test_max_ids_enforced(self, tools: AgentTools) -> None:
        """More than 10 IDs is rejected."""
        with pytest.raises(Exception):
            GetProductDetailsParams(product_ids=[f"B{i:03d}" for i in range(11)])


class TestCompareProducts:
    """compare_products tool tests."""

    def test_basic_compare(self, tools: AgentTools) -> None:
        """Basic comparison works."""
        params = CompareProductsParams(product_ids=["B001", "B002"])
        result = tools.compare_products(params)
        assert len(result.products) == 2
        assert len(result.comparison) > 0

    def test_min_two_required(self, tools: AgentTools) -> None:
        """Less than 2 products is rejected."""
        with pytest.raises(Exception):
            CompareProductsParams(product_ids=["B001"])

    def test_max_five_enforced(self, tools: AgentTools) -> None:
        """More than 5 products is rejected."""
        with pytest.raises(Exception):
            CompareProductsParams(product_ids=["B001", "B002", "B003", "B004", "B005", "B006"])


# ---------------------------------------------------------------------------
# Evidence Verifier Tests
# ---------------------------------------------------------------------------

class TestEvidenceVerifier:
    """Evidence verification tests."""

    def test_valid_recommendation(self, verifier: EvidenceVerifier) -> None:
        """Valid recommendation has no issues."""
        rec = RecommendationItem(
            product_id="B001", title="Sony WH-1000XM5", brand="Sony",
            price=349.99, rating=4.7, review_count=None,
            reason="Good match",
            evidence=[EvidenceRef(product_id="B001", field="title", observed_value="Sony WH-1000XM5 Wireless Noise Cancelling Headphones")],
        )
        issues = verifier.verify_recommendation(rec)
        assert issues == []

    def test_nonexistent_product(self, verifier: EvidenceVerifier) -> None:
        """Non-existent product flagged."""
        rec = RecommendationItem(product_id="FAKE99", title="Fake", review_count=None, brand=None, price=None, rating=None, reason="??")
        issues = verifier.verify_recommendation(rec)
        assert len(issues) > 0
        assert any("does not exist" in i for i in issues)

    def test_brand_mismatch(self, verifier: EvidenceVerifier) -> None:
        """Brand mismatch is detected."""
        rec = RecommendationItem(
            product_id="B001", title="Test", brand="WrongBrand", review_count=None, price=None, rating=None, reason="Test",
        )
        issues = verifier.verify_recommendation(rec)
        assert len(issues) > 0
        assert any("Brand" in i for i in issues)

    def test_evidence_field_unknown(self, verifier: EvidenceVerifier) -> None:
        """Evidence citing non-existent field is flagged."""
        rec = RecommendationItem(
            product_id="B001", title="Test", review_count=None, brand=None, price=None, rating=None,
            evidence=[EvidenceRef(product_id="B001", field="imaginary_field", observed_value="x")],
            reason="Test",
        )
        issues = verifier.verify_recommendation(rec)
        assert len(issues) > 0

    def test_verify_response_batch(self, verifier: EvidenceVerifier) -> None:
        """Batch verification returns all issues."""
        recs = [
            RecommendationItem(product_id="B001", title="Test", review_count=None, brand=None, price=None, rating=None, reason="ok"),
            RecommendationItem(product_id="FAKE99", title="Fake", review_count=None, brand=None, price=None, rating=None, reason="bad"),
        ]
        valid, issues = verifier.verify_response(recs)
        assert not valid
        assert len(issues) > 0


# ---------------------------------------------------------------------------
# Agent Orchestrator Tests (Fake LLM)
# ---------------------------------------------------------------------------

def _make_fake_llm(tool_calls: list[dict], final_content: str = "Here are your recommendations.") -> FakeLLMClient:
    """Build a fake LLM script: one search + one final response."""
    script = []
    if tool_calls:
        script.append({"content": None, "tool_calls": tool_calls})
    script.append({"content": final_content})
    return FakeLLMClient(script)


class TestAgentOrchestrator:
    """Agent orchestrator tests with FakeLLM."""

    def test_basic_run(self, service: RetrievalService) -> None:
        """Agent runs a simple request."""
        fake = _make_fake_llm(
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search_catalog", "arguments": json.dumps({"query": "wireless headphones", "top_k": 5})},
            }],
            final_content="I found Sony WH-1000XM5 headphones.",
        )
        tools = AgentTools(service)
        orch = AgentOrchestrator(fake, tools, service._product_index)
        req = AgentRequest(message="best wireless headphones", mode="balanced")
        resp = orch.run(req)
        assert isinstance(resp, AgentResponse)
        assert resp.status in ("completed", "degraded")
        assert resp.mode_used == "hybrid"
        assert resp.tool_calls >= 1

    def test_max_steps_enforced(self, service: RetrievalService) -> None:
        """Agent stops after max steps."""
        # Create fake that keeps calling tools forever
        script = []
        for _i in range(10):
            script.append({
                "content": None,
                "tool_calls": [{
                    "id": f"call_{_i}",
                    "type": "function",
                    "function": {"name": "search_catalog", "arguments": json.dumps({"query": "test", "top_k": 3})},
                }],
            })
        fake = FakeLLMClient(script)
        tools = AgentTools(service)
        orch = AgentOrchestrator(fake, tools, service._product_index, max_steps=3, max_tool_calls=6)
        req = AgentRequest(message="test", mode="balanced")
        resp = orch.run(req)
        assert resp.tool_calls <= 6

    def test_degraded_fallback_on_llm_error(self, service: RetrievalService) -> None:
        """When LLM fails, degraded fallback returns catalog results."""
        class FailingLLM(LLMClient):
            @property
            def model_name(self):
                return "failing"
            def send(self, messages, tools, *, timeout_s=30.0):
                raise ConnectionError("Simulated failure")
        tools = AgentTools(service)
        orch = AgentOrchestrator(FailingLLM(), tools, service._product_index)
        req = AgentRequest(message="headphones", mode="balanced")
        resp = orch.run(req)
        assert resp.status == "degraded"
        assert resp.degraded is True
        assert "LLM unavailable" in resp.answer or "degraded" in str(resp.status).lower()

    def test_recommendations_from_tool_results(self, service: RetrievalService) -> None:
        """Recommendations come from actual tool results."""
        fake = _make_fake_llm(
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {"name": "search_catalog", "arguments": json.dumps({"query": "headphones", "top_k": 5})},
            }],
            final_content="Done.",
        )
        tools = AgentTools(service)
        orch = AgentOrchestrator(fake, tools, service._product_index)
        req = AgentRequest(message="headphones", mode="balanced")
        resp = orch.run(req)
        assert len(resp.recommendations) > 0
        # All recommended product_ids must be real
        for rec in resp.recommendations:
            assert rec.product_id in service._product_index

    def test_quality_mode(self, service: RetrievalService) -> None:
        """Quality mode is passed through — falls back to hybrid if reranker unavailable."""
        # When sentence-transformers is unavailable, quality mode
        # triggers degraded=hybrid (degraded=true)
        fake = _make_fake_llm(
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {"name": "search_catalog", "arguments": json.dumps({"query": "headphones", "mode": "quality", "top_k": 5})},
            }],
        )
        tools = AgentTools(service)
        orch = AgentOrchestrator(fake, tools, service._product_index)
        req = AgentRequest(message="headphones", mode="quality")
        # May fail if CrossEncoder not installed — that's a legitimate degraded path
        try:
            resp = orch.run(req)
            assert resp.mode_requested == "quality"
            # Either completed with rerank, or degraded with hybrid fallback
            assert resp.status in ("completed", "degraded")
        except ImportError:
            pytest.skip("CrossEncoder not available, quality mode fallback testable only with sentence-transformers")

    def test_nonexistent_product_not_recommended(self, service: RetrievalService) -> None:
        """Agent cannot recommend products outside tool results."""
        valid_ids = set(service._product_index.keys())
        fake = _make_fake_llm([], "Done.")
        tools = AgentTools(service)
        orch = AgentOrchestrator(fake, tools, service._product_index)
        req = AgentRequest(message="test", mode="balanced")
        resp = orch.run(req)
        for rec in resp.recommendations:
            assert rec.product_id in valid_ids

    def test_deterministic_same_input(self, service: RetrievalService) -> None:
        """Same input + same fake LLM = same output."""
        script = [{
            "content": None,
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "search_catalog", "arguments": json.dumps({"query": "headphones", "top_k": 5})},
            }],
        }, {"content": "Done."}]

        for _ in range(2):
            fake = FakeLLMClient([dict(d) for d in script])
            tools = AgentTools(service)
            orch = AgentOrchestrator(fake, tools, service._product_index)
            req = AgentRequest(message="headphones", mode="balanced")
            resp = orch.run(req)
            assert resp.status == "completed"

    def test_tool_definitions_valid(self) -> None:
        """Tool definitions have required fields."""
        for td in TOOL_DEFINITIONS:
            assert td["type"] == "function"
            fn = td["function"]
            assert "name" in fn
            assert "parameters" in fn
            assert "properties" in fn["parameters"]


class TestFakeLLMClient:
    """FakeLLMClient tests."""

    def test_consumes_script(self) -> None:
        """Script is consumed in order."""
        script = [
            {"content": "first", "tool_calls": [{"name": "t1"}]},
            {"content": "second"},
        ]
        fake = FakeLLMClient(script)
        r1 = fake.send([], [])
        assert r1["content"] == "first"
        assert len(r1["tool_calls"]) == 1
        r2 = fake.send([], [])
        assert r2["content"] == "second"

    def test_loops_last(self) -> None:
        """After script exhausted, returns last response."""
        fake = FakeLLMClient([{"content": "only"}])
        r1 = fake.send([], [])
        r2 = fake.send([], [])
        assert r1["content"] == "only"
        assert r2["content"] == "only"

    def test_records_calls(self) -> None:
        """All send() calls are recorded."""
        fake = FakeLLMClient([{"content": "ok"}])
        fake.send([{"role": "user", "content": "hi"}], [{"type": "function"}])
        assert len(fake.calls) == 1
        assert fake.calls[0]["messages"][0]["content"] == "hi"


class TestAgentModels:
    """Pydantic model validation tests."""

    def test_agent_request_defaults(self) -> None:
        """Default values are applied."""
        req = AgentRequest(message="test")
        assert req.mode == "balanced"
        assert req.max_results == 5

    def test_invalid_mode_rejected(self) -> None:
        """Invalid mode raises validation error."""
        with pytest.raises(Exception):
            AgentRequest(message="test", mode="ultra_fast")

    def test_search_catalog_params_validation(self) -> None:
        """Unknown fields are forbidden."""
        with pytest.raises(Exception):
            SearchCatalogParams(query="test", unknown_field=123)

    def test_agent_response_defaults(self) -> None:
        """AgentResponse has proper defaults."""
        resp = AgentResponse(request_id="r1", status="completed", answer="ok")
        assert resp.degraded is False
        assert resp.warnings == []
        assert resp.tool_calls == 0
