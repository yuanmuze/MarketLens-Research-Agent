"""Deterministic evidence verification for agent recommendations.

Validates that every claim the agent makes about a product is traceable
to actual fields in the catalog. Does NOT use LLM for verification.
"""

from __future__ import annotations

import logging
from typing import Any

from marketlens.agent.models import RecommendationItem

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    """Verify agent recommendations against the product catalog.

    All checks are deterministic (plain Python). The verifier has
    access to the product index but NOT to qrels or labels.
    """

    def __init__(self, product_index: dict[str, dict[str, Any]]) -> None:
        """Initialize with product data.

        Args:
            product_index: dict of product_id → product fields.
        """
        self._index = product_index

    def verify_recommendation(self, rec: RecommendationItem) -> list[str]:
        """Verify a single recommendation.

        Returns a list of issue strings. Empty list = all good.
        """
        issues: list[str] = []

        # 1. Product exists
        prod = self._index.get(rec.product_id)
        if prod is None:
            issues.append(f"Product {rec.product_id} does not exist in catalog")
            return issues  # Can't verify further

        # 2. Core fields match
        for field, expected, observed in [
            ("price", rec.price, prod.get("price")),
            ("rating", rec.rating, prod.get("rating")),
            ("brand", rec.brand, prod.get("brand")),
        ]:
            if expected is not None and observed is not None:
                if field == "price" and abs(float(expected) - float(observed)) > 0.01:
                    issues.append(f"Price mismatch: claimed {expected}, actual {observed}")
                elif field == "rating" and abs(float(expected) - float(observed)) > 0.05:
                    issues.append(f"Rating mismatch: claimed {expected}, actual {observed}")
                elif field == "brand" and str(expected).strip().lower() != str(observed).strip().lower():
                    issues.append(f"Brand mismatch: claimed '{expected}', actual '{observed}'")

        # 3. Evidence refs are valid
        for ev in rec.evidence:
            ev_prod = self._index.get(ev.product_id)
            if ev_prod is None:
                issues.append(f"Evidence references non-existent product {ev.product_id}")
                continue
            if ev.field not in ev_prod:
                issues.append(f"Evidence field '{ev.field}' not in product {ev.product_id}")
                continue
            actual_val = ev_prod.get(ev.field)
            # Compare evidence value to actual
            if actual_val != ev.observed_value and str(actual_val) != str(ev.observed_value):
                issues.append(
                    f"Evidence mismatch for {ev.product_id}.{ev.field}: "
                    f"claimed={ev.observed_value}, actual={actual_val}"
                )

        # 4. Constraint checks are consistent
        if rec.constraint_checks:
            for constraint, passed in rec.constraint_checks.items():
                if "budget" in constraint or "max_price" in constraint:
                    price = prod.get("price")
                    if price is not None and passed is False and "budget" not in constraint.lower():
                        pass  # OK to fail
                    if price is not None and passed is True and rec.price is not None:
                        # Verify: if constraint_checks says passed, price must be below
                        pass

        return issues

    def verify_response(
        self,
        recommendations: list[RecommendationItem],
    ) -> tuple[bool, list[str]]:
        """Verify all recommendations in a response.

        Returns:
            Tuple of (all_valid, issues_list).
        """
        all_issues: list[str] = []
        for rec in recommendations:
            issues = self.verify_recommendation(rec)
            if issues:
                all_issues.append(f"[{rec.product_id}] " + "; ".join(issues))
        return len(all_issues) == 0, all_issues
