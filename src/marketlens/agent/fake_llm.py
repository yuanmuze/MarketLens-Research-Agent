"""Fake LLM for offline end-to-end testing without API keys.

Provides a deterministic, rule-based fake LLM that can respond to
product research queries with structured outputs based on the catalog.
"""

from __future__ import annotations

import logging
from typing import Any

from marketlens.models import (
    Product,
    UserConstraints,
)

logger = logging.getLogger(__name__)


class FakeLLM:
    """Fake LLM that uses simple keyword matching for offline testing.

    This allows the full agent workflow to run without any API keys.
    For real usage, swap with a LangChain chat model.
    """

    def __init__(self, products: list[Product] | None = None) -> None:
        """Initialize fake LLM with optional product context.

        Args:
            products: Products to use in fake responses.
        """
        self._products = products or []
        self._product_map = {p.product_id: p for p in self._products}

    def parse_request(self, query: str) -> dict[str, Any]:
        """Extract search intent and constraints from a natural language query.

        Args:
            query: Natural language research query.

        Returns:
            Dict with search_query and constraints.
        """
        query_lower = query.lower()

        # Extract budget constraints
        budget = None
        if "$" in query:
            import re

            prices = re.findall(r"\$(\d+(?:,\d+)?(?:\.\d+)?)", query)
            if prices:
                # Find "under $X" pattern
                under_match = re.search(
                    r"(?:under|below|less than|max|up to|within)\s*\$(\d+(?:,\d+)?(?:\.\d+)?)",
                    query_lower,
                )
                if under_match:
                    budget = float(under_match.group(1).replace(",", ""))
                else:
                    budget = float(prices[-1].replace(",", ""))

        # Extract brands
        known_brands = [
            "sony", "bose", "apple", "samsung", "jbl", "beats", "anker",
            "sennheiser", "google", "nothing", "technics", "jabra",
        ]
        preferred_brands = [b for b in known_brands if b in query_lower]

        # Extract category
        if any(w in query_lower for w in ["headphones", "headset", "over-ear", "over ear"]):
            category_hint = "headphones"
        elif any(w in query_lower for w in ["earbuds", "earphone", "in-ear", "true wireless"]):
            category_hint = "earbuds"
        elif any(w in query_lower for w in ["speaker", "audio system"]):
            category_hint = "speaker"
        elif any(w in query_lower for w in ["watch", "smartwatch"]):
            category_hint = "watch"
        else:
            category_hint = "audio"

        # Extract features from query
        features = []
        if "noise cancell" in query_lower or "anc" in query_lower:
            features.append("noise_cancellation")
        if "battery" in query_lower or "long lasting" in query_lower:
            features.append("long_battery")
        if "wireless" in query_lower:
            features.append("wireless")
        if "bluetooth" in query_lower:
            features.append("bluetooth")

        constraints = UserConstraints()
        if budget:
            constraints = UserConstraints(max_budget=budget)
        if preferred_brands:
            constraints.preferred_brands = preferred_brands

        return {
            "search_query": query,
            "budget": budget,
            "preferred_brands": preferred_brands,
            "category_hint": category_hint,
            "features": features,
            "constraints": constraints,
        }

    def assess_evidence(self, products: list[Product], query: str) -> list[dict[str, Any]]:
        """Generate evidence assessments for products based on query relevance.

        Args:
            products: Products to assess.
            query: The search query.

        Returns:
            List of evidence dicts.
        """
        query_lower = query.lower()
        evidence_list = []

        for product in products:
            score = self._simple_relevance(product, query_lower)
            keywords_matched = self._match_keywords(product, query_lower)

            evidence = {
                "product_id": product.product_id,
                "relevance_score": score,
                "keywords_matched": keywords_matched,
                "evidence_text": (
                    f"Product '{product.title}' from {product.brand} "
                    f"priced at ${product.price:.2f} with rating {product.rating}/5 "
                    f"({product.review_count} reviews). "
                    f"{product.description or ''} "
                    f"Matched keywords: {', '.join(keywords_matched) if keywords_matched else 'none'}."
                ),
            }
            evidence_list.append(evidence)

        # Sort by relevance
        evidence_list.sort(key=lambda e: e["relevance_score"], reverse=True)
        return evidence_list

    def compare_products(
        self, products: list[Product], query: str, evidence: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Generate product comparison.

        Args:
            products: Products to compare.
            query: The research query.
            evidence: Optional evidence assessments.

        Returns:
            List of comparison dicts.
        """
        query_lower = query.lower()
        comparisons = []

        for product in products:
            score = self._simple_relevance(product, query_lower)
            pros, cons = self._generate_pros_cons(product, query_lower)

            comparisons.append({
                "product_id": product.product_id,
                "title": product.title,
                "brand": product.brand,
                "price": product.price,
                "rating": product.rating,
                "relevance_score": score,
                "pros": pros,
                "cons": cons,
                "recommendation_score": min(10, max(1, round(score * 10))),
            })

        comparisons.sort(key=lambda c: c["recommendation_score"], reverse=True)
        return comparisons

    def validate_constraints(
        self,
        products: list[Product],
        constraints: UserConstraints | None,
    ) -> dict[str, Any]:
        """Validate that products satisfy user constraints.

        Args:
            products: Products to check.
            constraints: User constraints.

        Returns:
            Validation result dict.
        """
        if constraints is None:
            return {"all_satisfied": True, "violations": [], "passed_count": len(products)}

        violations = []
        passed = []

        for product in products:
            product_violations = []

            if constraints.max_budget is not None and product.price is not None:
                if product.price > constraints.max_budget:
                    product_violations.append(
                        f"Price ${product.price:.2f} exceeds budget ${constraints.max_budget:.2f}"
                    )

            if constraints.min_budget is not None and product.price is not None:
                if product.price < constraints.min_budget:
                    product_violations.append(
                        f"Price ${product.price:.2f} below minimum ${constraints.min_budget:.2f}"
                    )

            if constraints.min_rating is not None and product.rating is not None:
                if product.rating < constraints.min_rating:
                    product_violations.append(
                        f"Rating {product.rating} below minimum {constraints.min_rating}"
                    )

            if constraints.preferred_brands:
                if product.brand and product.brand.lower() not in {
                    b.lower() for b in constraints.preferred_brands
                }:
                    product_violations.append(
                        f"Brand {product.brand} not in preferred brands {constraints.preferred_brands}"
                    )

            if product_violations:
                violations.append({
                    "product_id": product.product_id,
                    "title": product.title,
                    "violations": product_violations,
                })
            else:
                passed.append(product.product_id)

        return {
            "all_satisfied": len(violations) == 0,
            "violations": violations,
            "passed_count": len(passed),
            "failed_count": len(violations),
        }

    def generate_report(
        self,
        query: str,
        products: list[Product],
        comparisons: list[dict[str, Any]],
        validation: dict[str, Any],
    ) -> str:
        """Generate a fake research report.

        Args:
            query: Original query.
            products: Recommended products.
            comparisons: Product comparisons.
            validation: Constraint validation result.

        Returns:
            Markdown research report string.
        """
        lines = [
            "# MarketLens Research Report",
            "",
            f"**Query**: {query}",
            f"**Date**: {self._today_str()}",
            f"**Products Analyzed**: {len(products)}",
            f"**Constraints Satisfied**: {'Yes' if validation['all_satisfied'] else 'No'}",
            "",
            "## Executive Summary",
            "",
            f"We analyzed {len(products)} products matching your query.",
            f"Top recommendation is **{products[0].title}** by {products[0].brand} "
            if products else "No products found matching your query.",
            "",
            "## Product Comparisons",
            "",
        ]

        if comparisons:
            lines.append("| Rank | Product | Brand | Price | Rating | Score |")
            lines.append("|------|---------|-------|-------|--------|-------|")
            for i, comp in enumerate(comparisons[:10], 1):
                lines.append(
                    f"| {i} | {comp['title']} | {comp['brand']} | "
                    f"${comp['price']:.2f} | {comp['rating']}/5 | {comp['recommendation_score']}/10 |"
                )
            lines.append("")
        else:
            lines.append("*No products to compare.*")
            lines.append("")

        # Detailed comparisons
        for comp in comparisons[:5]:
            lines.append(f"### {comp['title']}")
            lines.append(f"- **Brand**: {comp['brand']}")
            lines.append(f"- **Price**: ${comp['price']:.2f}")
            lines.append(f"- **Rating**: {comp['rating']}/5")
            if comp["pros"]:
                lines.append(f"- **Pros**: {', '.join(comp['pros'])}")
            if comp["cons"]:
                lines.append(f"- **Cons**: {', '.join(comp['cons'])}")
            lines.append("")

        # Constraint validation
        if validation["violations"]:
            lines.append("## Constraint Violations")
            for v in validation["violations"]:
                lines.append(f"- **{v['title']}**: {'; '.join(v['violations'])}")
            lines.append("")

        lines.append("## Recommendations")
        if products:
            lines.append(f"**Top Pick**: {products[0].title}")
            if len(products) > 1:
                lines.append(f"**Runner-up**: {products[1].title}")
            if len(products) > 2:
                lines.append(f"**Budget Pick**: {products[-1].title}")

        lines.append("")
        lines.append("## Evidence Sources")
        for p in products[:5]:
            lines.append(
                f"- [{p.product_id}] {p.title} — {p.brand}, "
                f"${p.price:.2f}, {p.rating}/5 ({p.review_count} reviews)"
            )

        lines.append("")
        lines.append("---")
        lines.append("*Report generated by MarketLens Research Agent (fake LLM mode)*")

        return "\n".join(lines)

    def _simple_relevance(self, product: Product, query_lower: str) -> float:
        """Compute simple relevance score based on keyword overlap.

        Args:
            product: The product.
            query_lower: Lowercased query.

        Returns:
            Relevance score 0-1.
        """
        text = product.to_search_text().lower()
        query_words = set(query_lower.split())
        text_words = set(text.split())

        if not query_words:
            return 0.5

        # Jaccard-like overlap
        intersection = query_words & text_words
        if not intersection:
            # Check for partial matches
            partial_hits = 0
            for qw in query_words:
                if len(qw) >= 3 and qw in text:
                    partial_hits += 1
            if partial_hits:
                return 0.3
            return 0.1

        return min(len(intersection) / len(query_words), 1.0)

    def _match_keywords(self, product: Product, query_lower: str) -> list[str]:
        """Find matching keywords between query and product.

        Args:
            product: The product.
            query_lower: Lowercased query.

        Returns:
            List of matched keywords.
        """
        text = product.to_search_text().lower()
        query_words = set(query_lower.split())
        matched = []
        for word in query_words:
            if len(word) >= 3 and word in text:
                matched.append(word)
        return matched

    def _generate_pros_cons(
        self, product: Product, query_lower: str
    ) -> tuple[list[str], list[str]]:
        """Generate pros and cons based on product attributes.

        Args:
            product: The product.
            query_lower: Lowercased query.

        Returns:
            Tuple of (pros, cons).
        """
        pros = []
        cons = []

        if product.rating is not None:
            if product.rating >= 4.5:
                pros.append(f"Excellent rating ({product.rating}/5)")
            elif product.rating < 4.0:
                cons.append(f"Below-average rating ({product.rating}/5)")

        if product.review_count is not None:
            if product.review_count >= 5000:
                pros.append(f"Popular ({product.review_count:,} reviews)")
            elif product.review_count < 500:
                cons.append(f"Few reviews ({product.review_count})")

        if product.price is not None:
            if product.price < 100:
                pros.append(f"Budget-friendly (${product.price:.2f})")
            elif product.price > 400:
                cons.append(f"Premium price (${product.price:.2f})")

        text = product.to_search_text().lower()
        if "noise cancell" in text or "anc" in text:
            pros.append("Noise cancellation")
        if "waterproof" in text or "ipx" in text:
            pros.append("Water resistant")
        if "battery" in text or "playtime" in text:
            pros.append("Good battery life")
        if "wireless" in text:
            pros.append("Wireless connectivity")

        return pros, cons

    @staticmethod
    def _today_str() -> str:
        """Get today's date as string."""
        from datetime import datetime

        now = datetime.now()
        return f"{now:%b} {now.day}, {now:%Y}"
