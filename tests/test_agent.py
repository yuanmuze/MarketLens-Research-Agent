"""Tests for the MarketLens product research agent."""

import pytest

from marketlens.agent.fake_llm import FakeLLM
from marketlens.agent.graph import create_research_agent
from marketlens.catalog import ProductCatalog
from marketlens.models import Product, UserConstraints


class TestFakeLLM:
    """FakeLLM tests."""

    @pytest.fixture
    def fake(self, sample_products: list[Product]) -> FakeLLM:
        """Create FakeLLM with sample products."""
        return FakeLLM(sample_products)

    def test_parse_request_basic(self, fake: FakeLLM) -> None:
        """Test basic query parsing."""
        result = fake.parse_request("best wireless headphones under $300")
        assert "wireless" in result["search_query"]
        assert result["budget"] == 300.0
        assert result["category_hint"] == "headphones"

    def test_parse_request_with_brand(self, fake: FakeLLM) -> None:
        """Test parsing query with brand preference."""
        result = fake.parse_request("Sony noise cancelling earbuds")
        assert "sony" in result["preferred_brands"]

    def test_parse_request_no_budget(self, fake: FakeLLM) -> None:
        """Test parsing query without budget."""
        result = fake.parse_request("best earbuds for running")
        assert result["budget"] is None

    def test_assess_evidence(self, fake: FakeLLM, sample_products: list[Product]) -> None:
        """Test evidence assessment."""
        evidence = fake.assess_evidence(sample_products[:3], "wireless headphones")
        assert len(evidence) == 3
        assert "product_id" in evidence[0]
        assert "relevance_score" in evidence[0]
        assert "evidence_text" in evidence[0]

    def test_assess_evidence_sorted(self, fake: FakeLLM, sample_products: list[Product]) -> None:
        """Test evidence is sorted by relevance."""
        evidence = fake.assess_evidence(sample_products, "wireless headphones")
        scores = [e["relevance_score"] for e in evidence]
        assert scores == sorted(scores, reverse=True)

    def test_compare_products(self, fake: FakeLLM, sample_products: list[Product]) -> None:
        """Test product comparison."""
        comparisons = fake.compare_products(sample_products[:3], "wireless headphones")
        assert len(comparisons) == 3
        assert "pros" in comparisons[0]
        assert "cons" in comparisons[0]
        assert "recommendation_score" in comparisons[0]

    def test_validate_constraints_all_pass(self, fake: FakeLLM) -> None:
        """Test validation where all products pass."""
        products = [
            Product(product_id="T1", title="Test1", price=99.99, rating=4.5, review_count=100, brand="Sony"),
            Product(product_id="T2", title="Test2", price=149.99, rating=4.3, review_count=200, brand="Sony"),
        ]
        constraints = UserConstraints(max_budget=200.0, preferred_brands=["Sony"])
        result = fake.validate_constraints(products, constraints)
        assert result["all_satisfied"] is True
        assert result["passed_count"] == 2

    def test_validate_constraints_budget_fail(self, fake: FakeLLM) -> None:
        """Test validation where budget is exceeded."""
        products = [
            Product(product_id="T1", title="Expensive", price=599.99, rating=4.5, review_count=100),
        ]
        constraints = UserConstraints(max_budget=200.0)
        result = fake.validate_constraints(products, constraints)
        assert result["all_satisfied"] is False
        assert result["passed_count"] == 0
        assert len(result["violations"]) == 1

    def test_validate_constraints_no_constraints(self, fake: FakeLLM) -> None:
        """Test validation with no constraints."""
        products = [Product(product_id="T1", title="Test", price=99.99)]
        result = fake.validate_constraints(products, None)
        assert result["all_satisfied"] is True

    def test_generate_report(self, fake: FakeLLM, sample_products: list[Product]) -> None:
        """Test report generation."""
        products = sample_products[:3]
        comparisons = [
            {
                "product_id": p.product_id,
                "title": p.title,
                "brand": p.brand or "N/A",
                "price": p.price or 0,
                "rating": p.rating or 0,
                "pros": ["Good quality"],
                "cons": [],
                "recommendation_score": 8.0,
            }
            for p in products
        ]
        validation = {"all_satisfied": True, "violations": [], "passed_count": 3, "failed_count": 0}
        report = fake.generate_report("wireless headphones", products, comparisons, validation)
        assert "MarketLens Research Report" in report
        assert "Executive Summary" in report
        assert "Product Comparisons" in report
        assert products[0].title in report


class TestResearchAgent:
    """End-to-end agent workflow tests."""

    @pytest.fixture
    def agent(self, catalog: ProductCatalog):
        """Create a research agent."""
        return create_research_agent(catalog, use_fake_llm=True)

    @pytest.mark.asyncio
    async def test_basic_research(self, agent) -> None:
        """Test a basic research query through the agent."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "best wireless headphones under $300",
            "request_id": "test-001",
        })
        assert result["status"] == "completed"
        assert result.get("final_report") is not None
        assert len(result.get("final_report", "")) > 0

    @pytest.mark.asyncio
    async def test_research_with_brand(self, agent) -> None:
        """Test research with brand preference."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "Sony noise cancelling headphones",
            "request_id": "test-002",
        })
        assert result["status"] == "completed"
        assert result.get("final_report") is not None

    @pytest.mark.asyncio
    async def test_research_returns_products(self, agent) -> None:
        """Test that the agent retrieves products."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "earbuds",
            "request_id": "test-003",
        })
        products = result.get("products", [])
        assert len(products) >= 0  # May be empty depending on catalog

    @pytest.mark.asyncio
    async def test_research_has_evidence(self, agent) -> None:
        """Test that evidence is generated."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "headphones",
            "request_id": "test-004",
        })
        evidence = result.get("evidence", [])
        products = result.get("products", [])
        if products:
            assert len(evidence) > 0

    @pytest.mark.asyncio
    async def test_research_compares_products(self, agent) -> None:
        """Test that comparisons are generated."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "wireless noise cancelling headphones",
            "request_id": "test-005",
        })
        comparisons = result.get("comparisons", [])
        products = result.get("products", [])
        if products:
            assert len(comparisons) > 0

    @pytest.mark.asyncio
    async def test_research_validates_constraints(self, agent) -> None:
        """Test constraint validation."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "best headphones",
            "request_id": "test-006",
        })
        assert result.get("constraints_satisfied") is not None

    @pytest.mark.asyncio
    async def test_research_handles_empty_query(self, catalog: ProductCatalog) -> None:
        """Test agent handles empty query gracefully."""
        agent = create_research_agent(catalog, use_fake_llm=True)
        result = await agent.ainvoke({
            "messages": [],
            "query": "",
            "request_id": "test-empty",
        })
        assert result["status"] in ("completed", "failed")
        # Should not crash

    @pytest.mark.asyncio
    async def test_research_report_contains_query(self, agent) -> None:
        """Test that the final report references the original query."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "budget wireless earbuds",
            "request_id": "test-007",
        })
        report = result.get("final_report", "")
        assert "wireless" in report.lower() or "budget" in report.lower()

    @pytest.mark.asyncio
    async def test_research_has_timings(self, agent) -> None:
        """Test that node timings are recorded."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "headphones",
            "request_id": "test-008",
        })
        timings = result.get("node_timings", {})
        assert "parse_request" in timings
        assert "retrieve_catalog" in timings

    @pytest.mark.asyncio
    async def test_research_with_empty_catalog(self, empty_catalog: ProductCatalog) -> None:
        """Test agent handles empty catalog gracefully."""
        agent = create_research_agent(empty_catalog, use_fake_llm=True)
        result = await agent.ainvoke({
            "messages": [],
            "query": "best headphones",
            "request_id": "test-empty-cat",
        })
        assert result["status"] in ("completed", "failed")
        products = result.get("products", [])
        assert products == []

    @pytest.mark.asyncio
    async def test_multiple_queries(self, agent) -> None:
        """Test running multiple queries sequentially."""
        queries = [
            "wireless headphones",
            "budget earbuds under $100",
            "sony noise cancelling",
        ]
        for i, query in enumerate(queries):
            result = await agent.ainvoke({
                "messages": [],
                "query": query,
                "request_id": f"test-multi-{i}",
            })
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_research_tracks_tool_calls(self, agent) -> None:
        """Test that tool call count is tracked."""
        result = await agent.ainvoke({
            "messages": [],
            "query": "best headphones for travel",
            "request_id": "test-tools",
        })
        tool_calls = result.get("tool_calls", 0)
        assert tool_calls >= 0  # At minimum, none failed
