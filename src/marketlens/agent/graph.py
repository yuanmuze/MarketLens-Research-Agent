"""LangGraph workflow for the MarketLens product research agent.

Implements a single-agent evidence-grounded workflow:
parse_request → retrieve_catalog → assess_evidence → (optional web) →
compare_products → validate_constraints → generate_report | handle_failure
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from marketlens.agent.fake_llm import FakeLLM
from marketlens.agent.state import AgentInputState, AgentState
from marketlens.agent.tools import (
    create_catalog_search_tool,
    create_web_search_tool,
)
from marketlens.catalog import ProductCatalog
from marketlens.models import (
    ComparisonItem,
    ProductEvidence,
    SearchQuery,
    UserConstraints,
)
from marketlens.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# Constants
MAX_TOOL_CALLS = 10
MAX_RETRIES = 3
TOOL_TIMEOUT_S = 30


def build_research_graph(
    catalog: ProductCatalog,
    use_fake_llm: bool = True,
) -> CompiledStateGraph[AgentState, None, AgentInputState, AgentState]:
    """Build the MarketLens research agent LangGraph.

    Args:
        catalog: The product catalog to search.
        use_fake_llm: If True, use FakeLLM (no API keys needed).
                     If False, use LangChain chat models (requires API keys).

    Returns:
        A compiled LangGraph StateGraph.
    """
    # Initialize components
    if len(catalog) > 0:
        retriever = HybridRetriever(catalog).fit()
    else:
        retriever = None  # Empty catalog, no retrieval possible
    if use_fake_llm:
        llm = FakeLLM(catalog.get_all_products())

    # Create tools (available for real LLM tool calling)
    if retriever is not None:
        _catalog_search = create_catalog_search_tool(catalog, retriever)
    web_search = create_web_search_tool()

    # --- Node implementations ---

    def parse_request_node(state: AgentState) -> dict[str, Any]:
        """Node 1: Parse the user request into structured search parameters."""
        start_time = time.monotonic()
        query = state.get("query", "")
        request_id = state.get("request_id", "unknown")

        logger.info("[%s] parse_request: %s", request_id, query[:100])

        try:
            if use_fake_llm:
                parsed = llm.parse_request(query)
            else:
                parsed = {"search_query": query, "constraints": UserConstraints()}

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["parse_request"] = elapsed

            return {
                "search_query": parsed.get("search_query", query),
                "constraints": parsed.get("constraints", UserConstraints()),
                "status": "running",
                "node_timings": timings,
                "tool_calls": state.get("tool_calls", 0),
            }
        except Exception as e:
            logger.error("[%s] parse_request error: %s", request_id, e)
            return {"error": str(e), "status": "failed"}

    def retrieve_catalog_node(state: AgentState) -> dict[str, Any]:
        """Node 2: Retrieve products from the catalog using hybrid search."""
        start_time = time.monotonic()
        request_id = state.get("request_id", "unknown")
        search_query = state.get("search_query", "")
        constraints = state.get("constraints", None)

        logger.info("[%s] retrieve_catalog: %s", request_id, search_query[:100])

        try:
            if retriever is not None:
                sq = SearchQuery(text=search_query, top_k=20, filters=constraints)
                search_results = retriever.search(sq)
            else:
                search_results = []

            products = [sr.product for sr in search_results]
            tool_calls = state.get("tool_calls", 0) + 1

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["retrieve_catalog"] = elapsed

            logger.info(
                "[%s] retrieve_catalog: found %d products (%.0fms)",
                request_id, len(products), elapsed * 1000,
            )

            return {
                "search_results": search_results,
                "products": products,
                "tool_calls": tool_calls,
                "node_timings": timings,
            }
        except Exception as e:
            logger.error("[%s] retrieve_catalog error: %s", request_id, e)
            return {
                "search_results": [],
                "products": [],
                "error": f"Catalog retrieval failed: {e}",
            }

    def assess_evidence_node(state: AgentState) -> dict[str, Any]:
        """Node 3: Assess evidence quality for retrieved products."""
        start_time = time.monotonic()
        request_id = state.get("request_id", "unknown")
        query = state.get("query", "")
        products = state.get("products", [])

        logger.info("[%s] assess_evidence: %d products", request_id, len(products))

        try:
            evidence_list: list[ProductEvidence] = []
            if use_fake_llm:
                assessments = llm.assess_evidence(products, query)
                for a in assessments:
                    ev = ProductEvidence(
                        product_id=a["product_id"],
                        source_type="catalog",
                        source_detail="MarketLens product catalog",
                        relevance_score=a.get("relevance_score", 0.0),
                        evidence_text=a.get("evidence_text", ""),
                    )
                    evidence_list.append(ev)
            else:
                # Without fake LLM, create basic evidence from catalog
                for product in products:
                    ev = ProductEvidence(
                        product_id=product.product_id,
                        source_type="catalog",
                        source_detail="MarketLens product catalog",
                        relevance_score=0.5,
                        evidence_text=(
                            f"Product: {product.title} | {product.brand} | "
                            f"${product.price:.2f} | {product.rating}/5"
                        ),
                    )
                    evidence_list.append(ev)

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["assess_evidence"] = elapsed

            return {
                "evidence": evidence_list,
                "node_timings": timings,
            }
        except Exception as e:
            logger.error("[%s] assess_evidence error: %s", request_id, e)
            return {"evidence": [], "error": str(e)}

    def optional_web_research_node(state: AgentState) -> dict[str, Any]:
        """Node 4: Optional web research to supplement catalog data."""
        start_time = time.monotonic()
        request_id = state.get("request_id", "unknown")
        query = state.get("query", "")

        logger.info("[%s] optional_web_research", request_id)

        try:
            web_result = web_search.invoke({"query": query})
            tool_calls = state.get("tool_calls", 0) + 1

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["web_research"] = elapsed

            return {
                "web_search_used": True,
                "web_search_results": [{"query": query, "result": web_result}],
                "tool_calls": tool_calls,
                "node_timings": timings,
            }
        except Exception as e:
            logger.warning("[%s] web_research error (non-fatal): %s", request_id, e)
            return {
                "web_search_used": False,
                "web_search_results": [],
            }

    def compare_products_node(state: AgentState) -> dict[str, Any]:
        """Node 5: Compare products and generate pros/cons."""
        start_time = time.monotonic()
        request_id = state.get("request_id", "unknown")
        query = state.get("query", "")
        products = state.get("products", [])
        evidence = state.get("evidence", [])

        logger.info("[%s] compare_products: %d products", request_id, len(products))

        try:
            evidence_map = {e.product_id: e for e in evidence}
            comparisons: list[ComparisonItem] = []

            if use_fake_llm:
                raw_comparisons = llm.compare_products(products, query)
                for rc in raw_comparisons:
                    pid = rc["product_id"]
                    prod = next((p for p in products if p.product_id == pid), None)
                    if prod is None:
                        continue
                    ev_item: ProductEvidence | None = evidence_map.get(pid)
                    comp = ComparisonItem(
                        product=prod,
                        evidence=[ev_item] if ev_item is not None else [],
                        pros=rc.get("pros", []),
                        cons=rc.get("cons", []),
                        recommendation_score=rc.get("recommendation_score", 5.0),
                    )
                    comparisons.append(comp)
            else:
                # Basic comparison without LLM
                for product in products[:10]:
                    ev_item2: ProductEvidence | None = evidence_map.get(product.product_id)
                    comp = ComparisonItem(
                        product=product,
                        evidence=[ev_item2] if ev_item2 is not None else [],
                        pros=[],
                        cons=[],
                        recommendation_score=float(product.rating or 5),
                    )
                    comparisons.append(comp)

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["compare_products"] = elapsed

            return {
                "comparisons": comparisons,
                "node_timings": timings,
            }
        except Exception as e:
            logger.error("[%s] compare_products error: %s", request_id, e)
            return {"comparisons": [], "error": str(e)}

    def validate_constraints_node(state: AgentState) -> dict[str, Any]:
        """Node 6: Deterministic constraint validation (plain Python, not LLM)."""
        start_time = time.monotonic()
        request_id = state.get("request_id", "unknown")
        products = state.get("products", [])
        constraints = state.get("constraints", None)

        logger.info("[%s] validate_constraints", request_id)

        try:
            if constraints:
                # Use catalog's deterministic filtering
                filtered = catalog.filter_by_constraints(
                    product_ids=[p.product_id for p in products],
                    max_budget=constraints.max_budget,
                    min_budget=constraints.min_budget,
                    brands=constraints.preferred_brands if constraints.preferred_brands else None,
                    excluded_brands=constraints.excluded_brands if constraints.excluded_brands else None,
                    min_rating=constraints.min_rating,
                    min_review_count=constraints.min_review_count,
                )
                products_passing = [
                    p for p in products if p.product_id in filtered
                ]
                constraints_satisfied = len(products_passing) == len(products)
                violations = [
                    {
                        "product_id": p.product_id,
                        "title": p.title,
                        "reason": "Does not satisfy one or more constraints",
                    }
                    for p in products if p.product_id not in filtered
                ]
            else:
                constraints_satisfied = True
                violations = []

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["validate_constraints"] = elapsed

            return {
                "constraints_satisfied": constraints_satisfied,
                "constraint_violations": violations,
                "node_timings": timings,
            }
        except Exception as e:
            logger.error("[%s] validate_constraints error: %s", request_id, e)
            return {"constraints_satisfied": True, "constraint_violations": []}

    def generate_report_node(state: AgentState) -> dict[str, Any]:
        """Node 7: Generate the final research report."""
        start_time = time.monotonic()
        request_id = state.get("request_id", "unknown")
        query = state.get("query", "")
        products = state.get("products", [])
        comparisons_raw = state.get("comparisons", [])
        validation = {
            "all_satisfied": state.get("constraints_satisfied", True),
            "violations": state.get("constraint_violations", []),
            "passed_count": len(products) - len(state.get("constraint_violations", [])),
            "failed_count": len(state.get("constraint_violations", [])),
        }

        logger.info("[%s] generate_report", request_id)

        try:
            if use_fake_llm:
                # Convert comparisons to dict format for the fake LLM
                comp_dicts = [
                    {
                        "product_id": c.product.product_id,
                        "title": c.product.title,
                        "brand": c.product.brand or "Unknown",
                        "price": c.product.price or 0,
                        "rating": c.product.rating or 0,
                        "pros": c.pros,
                        "cons": c.cons,
                        "recommendation_score": c.recommendation_score or 5.0,
                    }
                    for c in comparisons_raw
                ]
                report = llm.generate_report(query, products, comp_dicts, validation)
            else:
                # Simple report without LLM
                report = f"# Product Research Report\n\n**Query**: {query}\n\nNo LLM available. Found {len(products)} products."

            elapsed = time.monotonic() - start_time
            timings = dict(state.get("node_timings", {}))
            timings["generate_report"] = elapsed

            return {
                "final_report": report,
                "status": "completed",
                "completed_at": datetime.now(UTC),
                "node_timings": timings,
            }
        except Exception as e:
            logger.error("[%s] generate_report error: %s", request_id, e)
            return {
                "final_report": f"Error generating report: {e}",
                "status": "failed",
                "error": str(e),
            }

    def handle_failure_node(state: AgentState) -> dict[str, Any]:
        """Node 8: Handle failures gracefully."""
        request_id = state.get("request_id", "unknown")
        error = state.get("error", "Unknown error")
        logger.warning("[%s] handle_failure: %s", request_id, error)

        return {
            "status": "failed",
            "final_report": (
                f"# Research Failed\n\n"
                f"**Request**: {state.get('query', 'N/A')}\n\n"
                f"**Error**: {error}\n\n"
                f"Please try again with different search terms or constraints."
            ),
            "completed_at": datetime.now(UTC),
        }

    # --- Routing ---

    def route_after_parse(state: AgentState) -> Literal["retrieve_catalog", "handle_failure"]:
        """Route after parsing."""
        if state.get("error"):
            return "handle_failure"
        if state.get("tool_calls", 0) >= MAX_TOOL_CALLS:
            return "handle_failure"
        return "retrieve_catalog"

    def route_after_retrieve(state: AgentState) -> Literal["assess_evidence", "handle_failure"]:
        """Route after retrieval."""
        if state.get("error"):
            return "handle_failure"
        products = state.get("products", [])
        if not products:
            logger.info("No products found, generating empty report")
            # Still proceed — report will show no results
        return "assess_evidence"

    def route_after_assess(state: AgentState) -> Literal[
        "compare_products", "handle_failure"
    ]:
        """Route after evidence assessment."""
        if state.get("error"):
            return "handle_failure"
        return "compare_products"

    def route_after_compare(state: AgentState) -> Literal[
        "validate_constraints", "handle_failure"
    ]:
        """Route after comparison."""
        if state.get("error"):
            return "handle_failure"
        return "validate_constraints"

    def route_after_validate(state: AgentState) -> Literal[
        "generate_report", "handle_failure"
    ]:
        """Route after validation."""
        if state.get("error"):
            return "handle_failure"
        return "generate_report"

    # --- Build Graph ---

    builder = StateGraph(AgentState, input_schema=AgentInputState)
    builder.add_node("parse_request", parse_request_node)
    builder.add_node("retrieve_catalog", retrieve_catalog_node)
    builder.add_node("assess_evidence", assess_evidence_node)
    builder.add_node("optional_web_research", optional_web_research_node)
    builder.add_node("compare_products", compare_products_node)
    builder.add_node("validate_constraints", validate_constraints_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("handle_failure", handle_failure_node)

    # Edges
    builder.add_edge(START, "parse_request")
    builder.add_conditional_edges("parse_request", route_after_parse)
    builder.add_conditional_edges("retrieve_catalog", route_after_retrieve)
    builder.add_conditional_edges("assess_evidence", route_after_assess)
    builder.add_conditional_edges("compare_products", route_after_compare)
    builder.add_conditional_edges("validate_constraints", route_after_validate)
    builder.add_edge("generate_report", END)
    builder.add_edge("handle_failure", END)

    return builder.compile()


# Convenience factory
def create_research_agent(
    catalog: ProductCatalog,
    use_fake_llm: bool = True,
):
    """Create a compiled MarketLens research agent.

    Args:
        catalog: Product catalog with products.
        use_fake_llm: Use FakeLLM (no API keys) if True.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    return build_research_graph(catalog, use_fake_llm=use_fake_llm)


async def run_research(
    query: str,
    catalog: ProductCatalog,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Run a research query through the MarketLens agent.

    Args:
        query: Natural language research query.
        catalog: Product catalog.
        request_id: Optional request identifier.

    Returns:
        Agent output dict with final_report, products, comparisons, etc.
    """
    if request_id is None:
        import uuid
        request_id = f"req-{uuid.uuid4().hex[:12]}"

    agent = create_research_agent(catalog, use_fake_llm=True)

    input_state: AgentInputState = {
        "messages": [],
        "query": query,
        "request_id": request_id,
    }

    result = await agent.ainvoke(input_state)
    return result
