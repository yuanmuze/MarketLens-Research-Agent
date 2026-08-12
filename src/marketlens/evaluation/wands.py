"""WANDS evaluation adapter — bridge between WANDS data and MarketLens retrieval.

Provides:
- Loading WANDS products/queries/labels
- Adapting WANDS products to searchable documents for RetrievalService
- Relevance label mapping (Exact=2, Partial=1, Irrelevant=0)
- qrels dict for metric computation (query_id -> {product_id: relevance})

WANDS has no price/brand fields. This module does NOT inject fake prices
or brands. Structured filters (brand, price, rating) are not tested on WANDS.

The retrieval service is completely isolated from qrels — labels are ONLY
read by the evaluation module, never passed to the retriever/reranker.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Label mapping: WANDS label → numeric grade
LABEL_MAP = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

# Expected WANDS columns (detected from data)
PRODUCT_FIELDS = ["product_id", "product_name", "product_class", "category_hierarchy", "product_description", "product_features", "average_rating", "review_count"]
QUERY_FIELDS = ["query_id", "query", "query_class"]
LABEL_FIELDS = ["query_id", "product_id", "label"]


@dataclass
class WandsProduct:
    """Minimal product representation for WANDS evaluation.

    Only fields that exist in WANDS. Does NOT require price or brand.
    """
    product_id: str
    title: str
    product_class: str
    description: str
    rating: float | None
    review_count: int | None

    def to_search_text(self) -> str:
        """Build searchable text from WANDS fields.

        Combines: product_name + product_class + description.
        """
        parts = [self.title, self.product_class]
        if self.description:
            parts.append(self.description)
        return " ".join(p for p in parts if p.strip())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with RetrievalResultItem.from_product()."""
        return {
            "product_id": self.product_id,
            "title": self.title,
            "brand": "",  # WANDS has no brand
            "price": None,  # WANDS has no price
            "rating": self.rating,
            "review_count": self.review_count,
            "description": self.description,
            "attributes": {"product_class": self.product_class},
            "url": "",
        }


@dataclass
class WandsQuery:
    """WANDS query with metadata."""
    query_id: str
    query_text: str
    query_class: str


def _detect_delimiter(path: Path) -> str:
    """Detect tab vs comma delimiter."""
    with open(path, encoding="utf-8") as f:
        sample = f.read(8192)
    return "\t" if sample.count("\t") > sample.count(",") else ","


def load_products(path: Path) -> list[WandsProduct]:
    """Load WANDS products from CSV/TSV.

    Maps WANDS field names to our internal names.
    """
    delim = _detect_delimiter(path)
    products: list[WandsProduct] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=delim):
            title = row.get("product_name", "") or row.get("title", "")
            pclass = row.get("product_class", "") or row.get("category_hierarchy", "")
            desc = " ".join(filter(None, [
                row.get("product_description", ""),
                row.get("product_features", ""),
                row.get("category_hierarchy", ""),
            ]))

            rating = None
            raw_r = row.get("average_rating", "")
            if raw_r and raw_r.strip():
                try:
                    rating = float(raw_r)
                except ValueError:
                    pass

            review_count = None
            raw_rc = row.get("review_count", "")
            if raw_rc and raw_rc.strip():
                try:
                    review_count = int(float(raw_rc))
                except ValueError:
                    pass

            products.append(WandsProduct(
                product_id=row.get("product_id", "").strip(),
                title=title.strip(),
                product_class=pclass.strip(),
                description=desc.strip(),
                rating=rating,
                review_count=review_count,
            ))
    logger.info("Loaded %d WANDS products from %s", len(products), path)
    return products


def load_queries(path: Path) -> list[WandsQuery]:
    """Load WANDS queries."""
    delim = _detect_delimiter(path)
    queries: list[WandsQuery] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=delim):
            queries.append(WandsQuery(
                query_id=row.get("query_id", "").strip(),
                query_text=row.get("query", "").strip(),
                query_class=row.get("query_class", "").strip(),
            ))
    logger.info("Loaded %d WANDS queries from %s", len(queries), path)
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load relevance judgments from WANDS labels.

    WANDS has multiple annotators per query-product pair. We take
    the majority vote for each pair.

    Returns:
        qrels: dict[query_id, dict[product_id, relevance_score]]
        Where relevance_score ∈ {0, 1, 2} (Irrelevant/Partial/Exact).
    """
    delim = _detect_delimiter(path)

    # Collect all votes per (query, product) pair
    pair_votes: dict[tuple[str, str], list[int]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=delim):
            qid = row.get("query_id", "").strip()
            pid = row.get("product_id", "").strip()
            if not qid or not pid:
                continue
            # Find label value
            label_name = None
            for fname in ("label", "relevance", "rating"):
                if fname in row and row[fname] in LABEL_MAP:
                    label_name = row[fname]
                    break
            if label_name is None:
                continue
            pair_votes[(qid, pid)].append(LABEL_MAP[label_name])

    # Majority vote per pair
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for (qid, pid), votes in pair_votes.items():
        majority = Counter(votes).most_common(1)[0][0]
        qrels[qid][pid] = majority

    # Count distribution
    label_counts = Counter()
    for qr in qrels.values():
        for grade in qr.values():
            label_counts[grade] += 1

    grade_names = {2: "Exact", 1: "Partial", 0: "Irrelevant"}
    logger.info("Loaded qrels for %d queries (%d unique pairs: %s)",
                len(qrels), sum(len(v) for v in qrels.values()),
                {grade_names.get(k, str(k)): v for k, v in label_counts.items()})
    return dict(qrels)


def get_judged_products(qrels: dict[str, dict[str, int]], query_id: str) -> set[str]:
    """Get the set of judged product IDs for a query."""
    return set(qrels.get(query_id, {}).keys())


def get_relevant_products(qrels: dict[str, dict[str, int]], query_id: str, min_grade: int = 1) -> set[str]:
    """Get product IDs with relevance >= min_grade."""
    return {pid for pid, grade in qrels.get(query_id, {}).items() if grade >= min_grade}
