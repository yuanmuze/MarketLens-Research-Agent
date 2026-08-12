"""Typed agent tools — search, details, compare.

Reuses Phase 3 RetrievalService for actual search. All tool params
are validated by Pydantic models before execution.
"""

from __future__ import annotations

import logging
from typing import Any

from marketlens.agent.models import (
    CompareProductsParams,
    CompareProductsResult,
    GetProductDetailsParams,
    GetProductDetailsResult,
    ProductDetail,
    SearchCatalogParams,
    SearchCatalogResult,
    SearchResultItem,
)
from marketlens.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

# Mode → retrieval strategy mapping
MODE_STRATEGY: dict[str, str] = {
    "fast": "bm25",
    "balanced": "hybrid",
    "quality": "rerank",
}

# Tool definitions in OpenAI function-calling format
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search product catalog by keyword. Use for initial discovery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "balanced", "quality"],
                        "description": "fast=BM25, balanced=Hybrid, quality=Rerank",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                    "price_min": {"type": "number", "minimum": 0},
                    "price_max": {"type": "number", "minimum": 0},
                    "brands": {"type": "array", "items": {"type": "string"}},
                    "min_rating": {"type": "number", "minimum": 0, "maximum": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get full details for specific products by ID. Use after search to inspect candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10,
                        "description": "Product IDs to fetch",
                    },
                },
                "required": ["product_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Compare 2-5 products side-by-side by specified fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 5,
                        "description": "Product IDs to compare",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to compare: title, price, rating, brand, review_count",
                    },
                },
                "required": ["product_ids"],
            },
        },
    },
]


class AgentTools:
    """Encapsulates all agent tools with a shared RetrievalService."""

    def __init__(self, service: RetrievalService) -> None:
        """Initialize tools with a retrieval service.

        Args:
            service: Initialized RetrievalService.
        """
        self._service = service

    def search_catalog(self, params: SearchCatalogParams) -> SearchCatalogResult:
        """Execute a catalog search.

        Args:
            params: Validated search parameters.

        Returns:
            Search results.
        """
        strategy = MODE_STRATEGY.get(params.mode, "hybrid")
        output = self._service.search(
            query=params.query,
            strategy=strategy,
            top_k=params.top_k,
            candidate_k=50,
            min_price=params.price_min,
            max_price=params.price_max,
            brand=None,  # Multi-brand support via filter step
            min_rating=params.min_rating,
        )

        items = [
            SearchResultItem(
                rank=item.rank,
                product_id=item.product_id,
                title=item.title,
                brand=item.brand or None,
                price=item.price,
                rating=item.rating,
                review_count=item.review_count,
                score=item.final_score,
            )
            for item in output.results
        ]

        return SearchCatalogResult(
            query=params.query,
            mode_used=output.strategy,
            total_found=len(items),
            results=items,
        )

    def get_product_details(self, params: GetProductDetailsParams) -> GetProductDetailsResult:
        """Fetch full product details.

        Args:
            params: Product IDs to fetch.

        Returns:
            Product details for each valid ID.
        """
        products: list[ProductDetail] = []
        for pid in params.product_ids:
            prod = self._service._product_index.get(pid)
            if prod is None:
                logger.warning("get_product_details: unknown id %s", pid)
                continue
            products.append(ProductDetail(
                product_id=str(prod.get("product_id", pid)),
                title=str(prod.get("title", "")),
                brand=prod.get("brand") or None,
                price=prod.get("price") if prod.get("price") is not None else None,
                rating=prod.get("rating") if prod.get("rating") is not None else None,
                review_count=prod.get("review_count") if prod.get("review_count") is not None else None,
                description=str(prod.get("description") or ""),
                attributes=prod.get("attributes", {}),
                url=str(prod.get("url") or ""),
            ))
        return GetProductDetailsResult(products=products)

    def compare_products(self, params: CompareProductsParams) -> CompareProductsResult:
        """Compare products by requested fields.

        Args:
            params: Product IDs and fields to compare.

        Returns:
            Comparison with product details and field table.
        """
        detail_result = self.get_product_details(
            GetProductDetailsParams(product_ids=params.product_ids)
        )
        products = detail_result.products

        # Build comparison table
        comparison: list[dict[str, Any]] = []
        for field in params.fields:
            row: dict[str, Any] = {"field": field}
            for p in products:
                if field == "title":
                    row[p.product_id] = p.title
                elif field == "price":
                    row[p.product_id] = p.price if p.price is not None else "unknown"
                elif field == "rating":
                    row[p.product_id] = p.rating if p.rating is not None else "unknown"
                elif field == "brand":
                    row[p.product_id] = p.brand or "unknown"
                elif field == "review_count":
                    row[p.product_id] = p.review_count if p.review_count is not None else "unknown"
            comparison.append(row)

        return CompareProductsResult(products=products, comparison=comparison)

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call by name with validated arguments.

        Args:
            tool_name: One of search_catalog, get_product_details, compare_products.
            arguments: Raw arguments from LLM.

        Returns:
            Pydantic model result.

        Raises:
            ValueError: Unknown tool or invalid arguments.
        """
        if tool_name == "search_catalog":
            params = SearchCatalogParams(**arguments)
            return self.search_catalog(params)
        elif tool_name == "get_product_details":
            params = GetProductDetailsParams(**arguments)
            return self.get_product_details(params)
        elif tool_name == "compare_products":
            params = CompareProductsParams(**arguments)
            return self.compare_products(params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# Legacy Phase 2 compatibility (keep old graph.py working)
# ---------------------------------------------------------------------------

def create_catalog_search_tool(catalog: Any, retriever: Any = None) -> Any:
    """Legacy catalog search tool for old graph.py (Phase 2)."""
    from langchain_core.tools import tool
    from marketlens.models import SearchQuery

    @tool(description="Search the product catalog by keyword query.")
    def search_catalog(query: str, top_k: int = 10) -> str:
        if retriever is None:
            return "Catalog retriever not available."
        try:
            sq = SearchQuery(text=query, top_k=min(top_k, 50))
            results = retriever.search(sq)
        except Exception as e:
            return f"Error searching catalog: {e}"
        if not results:
            return "No products found."
        lines = [f"Found {len(results)} products for: {query}", ""]
        for r in results:
            p = r.product
            lines.append(
                f"- [{p.product_id}] {p.title} | {p.brand} | "
                f"${p.price:.2f} | {p.rating}/5 | Score: {r.score:.3f}"
            )
        return "\n".join(lines)

    return search_catalog


def create_web_search_tool() -> Any:
    """Legacy web search tool for old graph.py (Phase 2)."""
    from langchain_core.tools import tool

    @tool(description="Search the web for supplementary product information.")
    def web_search(query: str) -> str:
        return (
            "[Web search disabled] No TAVILY_API_KEY configured. "
            "Using catalog data only."
        )
    return web_search
