"""Agent tools for MarketLens product research.

Provides tools for catalog search, web search, and research completion
that the LangGraph agent can invoke.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from marketlens.catalog import ProductCatalog
from marketlens.models import SearchQuery
from marketlens.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


def create_catalog_search_tool(
    catalog: ProductCatalog,
    retriever: HybridRetriever | None = None,
) -> Any:
    """Create a LangChain tool for searching the product catalog.

    Args:
        catalog: The product catalog to search.
        retriever: Optional pre-built HybridRetriever. If None, one is created.

    Returns:
        A LangChain tool that accepts a query string and returns results.
    """
    if retriever is None:
        if len(catalog) > 0:
            retriever = HybridRetriever(catalog).fit()
        # If empty catalog, retriever stays None

    @tool(description="Search the product catalog by keyword query. Returns relevant products with scores and evidence.")
    def search_catalog(query: str, top_k: int = 10) -> str:
        """Search the product catalog.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return (default 10).

        Returns:
            Formatted search results as a string.
        """
        if retriever is None:
            return "Catalog is empty. No products available to search."

        search_query = SearchQuery(text=query, top_k=min(top_k, 50))
        try:
            results = retriever.search(search_query)
        except Exception as e:
            logger.error("Catalog search error: %s", e)
            return f"Error searching catalog: {e}"

        if not results:
            return "No products found matching your query."

        lines = [f"Found {len(results)} products for: {query}", ""]
        for r in results:
            p = r.product
            lines.append(
                f"- [{p.product_id}] {p.title} | {p.brand} | "
                f"${p.price:.2f} | {p.rating}/5 ({p.review_count} reviews) | "
                f"Score: {r.score:.3f} | Source: {r.source}"
            )
            if p.description:
                lines.append(f"  {p.description[:200]}")
            lines.append("")
        return "\n".join(lines)

    return search_catalog


def create_web_search_tool() -> Any:
    """Create an optional web search tool.

    Without API keys, this tool returns a message indicating web search
    is unavailable. In production, this would use Tavily or similar.

    Returns:
        A LangChain tool for web search.
    """

    @tool(description="Search the web for supplementary product information. Requires API keys to function.")
    def web_search(query: str) -> str:
        """Search the web for supplementary information.

        Args:
            query: The search query.

        Returns:
            Web search results or a disabled message.
        """
        import os

        tavily_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("TAVILY_SEARCH_API_KEY")
        if not tavily_key:
            return (
                "[Web search disabled] No TAVILY_API_KEY configured. "
                "Using catalog data only. To enable web search, set the TAVILY_API_KEY "
                "environment variable. Query would have been: " + query
            )

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=5)
            results = response.get("results", [])
            if not results:
                return f"No web results found for: {query}"

            lines = [f"Web search results for: {query}", ""]
            for i, r in enumerate(results[:5], 1):
                lines.append(f"{i}. {r.get('title', 'N/A')}")
                lines.append(f"   URL: {r.get('url', 'N/A')}")
                lines.append(f"   {r.get('content', '')[:300]}")
                lines.append("")
            return "\n".join(lines)
        except ImportError:
            return "[Web search disabled] tavily-python not available."
        except Exception as e:
            logger.error("Web search error: %s", e)
            return f"Web search error: {e}"

    return web_search


def create_research_complete_tool() -> Any:
    """Create a tool signaling research completion."""

    @tool(description="Call this when you have gathered enough information and are ready to generate the final report.")
    def research_complete(summary: str = "") -> str:
        """Signal research completion.

        Args:
            summary: Brief summary of research findings.

        Returns:
            Confirmation message.
        """
        return f"Research complete. Summary: {summary}"

    return research_complete
