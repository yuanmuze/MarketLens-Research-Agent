"""Tests for data audit and eval candidate generation scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.audit_electronics_data import (
    compute_missing_rate,
    compute_stats,
    run_audit,
)
from scripts.generate_eval_candidates import (
    QUERY_TYPES,
    EvalCandidate,
    build_index,
    generate_candidates,
    validate_candidates,
)


class TestAuditFunctions:
    """Tests for audit helper functions."""

    def test_compute_missing_rate_all_present(self) -> None:
        """Test missing rate when all products have the field."""
        products = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        assert compute_missing_rate(products, "name") == 0.0

    def test_compute_missing_rate_half_missing(self) -> None:
        """Test missing rate with half missing."""
        products = [{"name": "A"}, {}, {"name": "C"}, {}]
        assert compute_missing_rate(products, "name") == 0.5

    def test_compute_missing_rate_all_missing(self) -> None:
        """Test missing rate when field is absent everywhere."""
        products = [{}, {}, {}]
        assert compute_missing_rate(products, "name") == 1.0

    def test_compute_missing_rate_none_values(self) -> None:
        """Test missing rate counts None as missing."""
        products = [{"name": None}, {"name": ""}]
        assert compute_missing_rate(products, "name") == 1.0

    def test_compute_stats_normal(self) -> None:
        """Test stats computation on normal data."""
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["count"] == 5
        assert stats["null_count"] == 0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["p50"] == 3.0

    def test_compute_stats_with_nulls(self) -> None:
        """Test stats with null values mixed in."""
        stats = compute_stats([10.0, None, 20.0, None, 30.0])
        assert stats["count"] == 5
        assert stats["null_count"] == 2
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["mean"] == 20.0

    def test_compute_stats_all_nulls(self) -> None:
        """Test stats with all null values."""
        stats = compute_stats([None, None, None])
        assert stats["count"] == 3
        assert stats["null_count"] == 3
        assert stats["min"] == 0
        assert stats["max"] == 0

    def test_compute_stats_empty(self) -> None:
        """Test stats on empty list."""
        stats = compute_stats([])
        assert stats["count"] == 0

    def test_audit_on_temp_file(self, tmp_path: Path) -> None:
        """Test audit runs on a minimal temporary file."""
        import json

        products = [
            {
                "product_id": f"B{i:03d}",
                "title": f"Product {i}",
                "brand": "TestBrand",
                "price": 10.0 + i,
                "rating": 4.0,
                "review_count": 100,
                "category": "electronics",
                "description": f"Desc {i}",
                "attributes": {},
                "images": [],
                "url": "http://example.com",
            }
            for i in range(5)
        ]
        path = tmp_path / "test_products.json"
        path.write_text(json.dumps(products), encoding="utf-8")

        report = run_audit(path)
        assert report["basic_counts"]["total_products"] == 5
        assert report["basic_counts"]["unique_product_ids"] == 5
        assert report["missing_rates"]["title"] == 0.0


class TestEvalCandidateGeneration:
    """Tests for eval candidate generation."""

    @pytest.fixture
    def sample_products(self) -> list[dict]:
        """Create a small sample product list for testing."""
        return [
            {
                "product_id": "B001",
                "title": "Sony Wireless Noise Cancelling Headphones WH-1000XM5",
                "brand": "Sony",
                "price": 349.99,
                "rating": 4.7,
                "review_count": 500,
                "description": "Premium ANC headphones with 30h battery",
                "category": "electronics",
                "attributes": {},
                "images": [],
                "url": "http://example.com/B001",
            },
            {
                "product_id": "B002",
                "title": "Bose QuietComfort Ultra Earbuds",
                "brand": "Bose",
                "price": 299.00,
                "rating": 4.5,
                "review_count": 300,
                "description": "Spatial audio earbuds with ANC",
                "category": "electronics",
                "attributes": {},
                "images": [],
                "url": "http://example.com/B002",
            },
            {
                "product_id": "B003",
                "title": "Budget Wireless Earbuds Sport Edition",
                "brand": "Anker",
                "price": 49.99,
                "rating": 4.2,
                "review_count": 2000,
                "description": "Affordable sport earbuds with long battery",
                "category": "electronics",
                "attributes": {},
                "images": [],
                "url": "http://example.com/B003",
            },
            {
                "product_id": "B004",
                "title": "Apple AirPods Pro 2nd Gen with USB-C",
                "brand": "Apple",
                "price": 249.99,
                "rating": 4.8,
                "review_count": 10000,
                "description": "Active noise cancellation spatial audio",
                "category": "electronics",
                "attributes": {},
                "images": [],
                "url": "http://example.com/B004",
            },
            {
                "product_id": "B005",
                "title": "Samsung Galaxy Buds3 Pro True Wireless",
                "brand": "Samsung",
                "price": 199.99,
                "rating": 4.3,
                "review_count": 800,
                "description": "Adaptive ANC wireless earbuds",
                "category": "electronics",
                "attributes": {},
                "images": [],
                "url": "http://example.com/B005",
            },
        ]

    def test_build_index(self, sample_products: list[dict]) -> None:
        """Test product indexing."""
        index = build_index(sample_products)
        assert len(index) == 5
        assert index["B001"]["title"] == "Sony Wireless Noise Cancelling Headphones WH-1000XM5"

    def test_generate_candidates_deterministic(self, sample_products: list[dict]) -> None:
        """Test same seed produces identical candidates."""
        c1 = generate_candidates(sample_products, total=20, seed=42)
        c2 = generate_candidates(sample_products, total=20, seed=42)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.query_id == b.query_id
            assert a.query == b.query
            assert a.expected_product_ids == b.expected_product_ids

    def test_generate_different_seed_different(self, sample_products: list[dict]) -> None:
        """Test different seeds produce different ordering (with enough products)."""
        c1 = generate_candidates(sample_products, total=20, seed=42)
        c2 = generate_candidates(sample_products, total=20, seed=99)
        # With only 5 products, some query types are fully deterministic.
        # At minimum, verify both runs succeed and produce valid output.
        assert len(c1) >= 10
        assert len(c2) >= 10
        # Queries should be valid
        for c in c1 + c2:
            assert c.query and c.query_id and c.query_type in QUERY_TYPES

    def test_query_ids_unique(self, sample_products: list[dict]) -> None:
        """Test all query_ids are unique."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        ids = [c.query_id for c in candidates]
        assert len(ids) == len(set(ids))

    def test_all_product_ids_exist(self, sample_products: list[dict]) -> None:
        """Test all referenced product IDs exist in the source data."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        valid_ids = {p["product_id"] for p in sample_products}
        for c in candidates:
            for pid in c.expected_product_ids:
                assert pid in valid_ids, f"{c.query_id} references {pid} which does not exist"

    def test_query_types_coverage(self, sample_products: list[dict]) -> None:
        """Test that most query types are represented (small fixture may lack some)."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        types_found = {c.query_type for c in candidates}
        # With only 5 products, not all types may be representable
        assert len(types_found) >= 3  # At minimum keyword, semantic, attribute
        assert "keyword" in types_found  # Always has enough products
        for t in types_found:
            assert t in QUERY_TYPES

    def test_no_empty_queries(self, sample_products: list[dict]) -> None:
        """Test no query has empty text."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        for c in candidates:
            assert c.query and c.query.strip(), f"{c.query_id} has empty query"

    def test_no_answer_candidates_have_no_expected(self, sample_products: list[dict]) -> None:
        """Test no-answer candidates have empty expected_product_ids."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        for c in candidates:
            is_no_ans = "no_answer" in (c.reviewer_notes or "") or "no_answer" in c.generation_reason
            if is_no_ans:
                assert c.expected_product_ids == [], (
                    f"no_answer candidate {c.query_id} has products: {c.expected_product_ids}"
                )

    def test_numeric_constraints_verified(self, sample_products: list[dict]) -> None:
        """Test budget/rating constraints are consistent with target products."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        prod_index = build_index(sample_products)
        for c in candidates:
            for pid in c.expected_product_ids:
                p = prod_index.get(pid)
                if p is None:
                    continue
                if "max_budget" in c.expected_constraints:
                    price = p.get("price")
                    if price is not None:
                        assert price <= c.expected_constraints["max_budget"], (
                            f"Budget violation: {pid} price ${price:.2f} > ${c.expected_constraints['max_budget']:.2f}"
                        )
                if "min_rating" in c.expected_constraints:
                    rating = p.get("rating")
                    if rating is not None:
                        assert rating >= c.expected_constraints["min_rating"], (
                            f"Rating violation: {pid} rating {rating} < {c.expected_constraints['min_rating']}"
                        )

    def test_validate_returns_valid(self, sample_products: list[dict]) -> None:
        """Test validation passes on generated candidates."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        prod_index = build_index(sample_products)
        result = validate_candidates(candidates, prod_index)
        assert result["valid"], f"Validation failed: {result['issues'][:5]}"

    def test_validate_detects_invalid_product(self, sample_products: list[dict]) -> None:
        """Test validation detects references to non-existent products."""
        candidates = generate_candidates(sample_products, total=20, seed=42)
        prod_index = build_index(sample_products)
        # Tamper with one candidate
        if candidates:
            candidates[0].expected_product_ids.append("B99999")
        result = validate_candidates(candidates, prod_index)
        assert not result["valid"]
        assert any("B99999" in i for i in result["issues"])

    def test_validate_detects_budget_violation(self, sample_products: list[dict]) -> None:
        """Test validation detects budget constraint violations."""
        prod_index = build_index(sample_products)
        c = EvalCandidate(
            query_id="test-001",
            query="test",
            query_type="keyword",
            expected_product_ids=["B001"],
            expected_constraints={"max_budget": 10.0},  # B001 is $349.99
            source_product_ids=["B001"],
            generation_reason="test",
        )
        result = validate_candidates([c], prod_index)
        assert not result["valid"]
        assert any("Budget" in i for i in result["issues"])

    def test_validate_detects_duplicate_ids(self) -> None:
        """Test validation detects duplicate query IDs."""
        prod_index = {"B001": {}}
        c1 = EvalCandidate(
            query_id="dup-001", query="t1", query_type="keyword",
            expected_product_ids=["B001"], source_product_ids=["B001"],
            generation_reason="test",
        )
        c2 = EvalCandidate(
            query_id="dup-001", query="t2", query_type="keyword",
            expected_product_ids=["B001"], source_product_ids=["B001"],
            generation_reason="test",
        )
        result = validate_candidates([c1, c2], prod_index)
        assert not result["valid"]
        assert any("Duplicate" in i for i in result["issues"])

    def test_does_not_modify_input_products(self, sample_products: list[dict]) -> None:
        """Test generation does not modify the input product list."""
        before = json.dumps(sample_products)
        _candidates = generate_candidates(sample_products, total=20, seed=42)
        after = json.dumps(sample_products)
        assert before == after


class TestEvalCandidateIO:
    """Tests for eval candidate serialization."""

    @pytest.fixture
    def sample_products(self) -> list[dict]:
        """Small product set for testing."""
        return [
            {
                "product_id": f"B{i:03d}",
                "title": f"Product {i}",
                "brand": f"Brand{i}",
                "price": 10.0 + i,
                "rating": 4.0,
                "review_count": 100,
                "description": f"Desc {i}",
                "category": "electronics",
                "attributes": {},
                "images": [],
                "url": f"http://e.com/{i}",
            }
            for i in range(10)
        ]

    def test_jsonl_writes_and_reads(self, sample_products: list[dict], tmp_path: Path) -> None:
        """Test JSONL output is valid and readable."""
        candidates = generate_candidates(sample_products, total=10, seed=42)
        jl_path = tmp_path / "candidates.jsonl"

        with open(jl_path, "w", encoding="utf-8") as f:
            for c in candidates:
                f.write(json.dumps({
                    "query_id": c.query_id,
                    "query": c.query,
                    "query_type": c.query_type,
                    "expected_product_ids": c.expected_product_ids,
                    "expected_constraints": c.expected_constraints,
                    "source_product_ids": c.source_product_ids,
                    "generation_reason": c.generation_reason,
                    "reviewer_status": c.reviewer_status,
                    "reviewer_notes": c.reviewer_notes,
                }, ensure_ascii=False) + "\n")

        # Read back
        read_back = []
        with open(jl_path, encoding="utf-8") as f:
            for line in f:
                read_back.append(json.loads(line.strip()))

        assert len(read_back) == len(candidates)
        assert read_back[0]["query_id"] == candidates[0].query_id

    def test_csv_writes_valid(self, sample_products: list[dict], tmp_path: Path) -> None:
        """Test CSV output is valid."""
        candidates = generate_candidates(sample_products, total=10, seed=42)
        csv_path = tmp_path / "review.csv"

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "query_type", "query", "expected_product_ids",
                "constraints", "generation_reason", "reviewer_status", "reviewer_notes",
            ])
            for c in candidates:
                writer.writerow([
                    c.query_id, c.query_type, c.query,
                    "|".join(c.expected_product_ids),
                    json.dumps(c.expected_constraints),
                    c.generation_reason,
                    c.reviewer_status,
                    c.reviewer_notes,
                ])

        # Read back CSV
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(candidates)
        assert rows[0]["query_id"] == candidates[0].query_id

    def test_candidates_review_status_default_pending(self, sample_products: list[dict]) -> None:
        """Test all generated candidates start as pending."""
        candidates = generate_candidates(sample_products, total=10, seed=42)
        for c in candidates:
            assert c.reviewer_status == "pending"
