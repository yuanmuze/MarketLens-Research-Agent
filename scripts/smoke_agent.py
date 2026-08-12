#!/usr/bin/env python3
"""Smoke-test the Phase 5 agent with a real LLM (requires API key).

Without MARKETLENS_AGENT_API_KEY, this script exits with a clear
message and does not attempt any network calls.

Usage:
  # Requires MARKETLENS_AGENT_API_KEY to be set
  uv run python scripts/smoke_agent.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run agent smoke test with real product data."""
    api_key = os.environ.get("MARKETLENS_AGENT_API_KEY", "")
    if not api_key:
        logger.info("MARKETLENS_AGENT_API_KEY not set — skipping real model smoke test.")
        logger.info("Set it and re-run to test with a live LLM.")
        sys.exit(0)

    from marketlens.agent.models import AgentRequest  # noqa: E402
    from marketlens.agent.orchestrator import AgentOrchestrator  # noqa: E402
    from marketlens.agent.tools import AgentTools  # noqa: E402
    from marketlens.api.routes import _build_llm_client  # noqa: E402
    from marketlens.catalog import ProductCatalog  # noqa: E402
    from marketlens.retrieval.service import RetrievalService  # noqa: E402

    catalog = ProductCatalog.from_fixture("electronics_sample.json")
    service = RetrievalService(catalog, use_fake_embeddings=True)
    service.initialize()
    tools = AgentTools(service)
    llm = _build_llm_client()

    logger.info("LLM: %s (configured)", llm.model_name)
    logger.info("Products: %d", service.product_count)

    orch = AgentOrchestrator(llm, tools, service._product_index)

    test_queries = [
        "I need wireless headphones under $300 with a rating of at least 4.5. Compare the best two.",
        "Find me a budget earbud under $100.",
        "Show me Sony noise cancelling headphones.",
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info("--- Query %d: %s ---", i, query)
        t0 = time.monotonic()
        try:
            resp = orch.run(AgentRequest(message=query, mode="balanced", max_results=3))
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("  Status: %s, Recommendations: %d, Time: %.0fms",
                        resp.status, len(resp.recommendations), elapsed)
            for rec in resp.recommendations[:3]:
                logger.info("    - %s | %s | $%s | %s",
                            rec.product_id, rec.title[:50], rec.price, rec.rating)
        except Exception as e:
            logger.error("  Failed: %s", e)

    logger.info("Smoke test complete.")


if __name__ == "__main__":
    main()
