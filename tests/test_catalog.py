"""Tests for ProductCatalog."""

import json
import tempfile
from pathlib import Path

import pytest

from marketlens.catalog import ProductCatalog
from marketlens.models import Product, ProductCategory


class TestProductCatalog:
    """ProductCatalog tests."""

    def test_empty_catalog(self) -> None:
        """Test creating an empty catalog."""
        cat = ProductCatalog()
        assert len(cat) == 0
        assert cat.get_all_products() == []

    def test_add_product(self, sample_products: list[Product]) -> None:
        """Test adding products to the catalog."""
        cat = ProductCatalog()
        for p in sample_products:
            cat.add_product(p)
        assert len(cat) == len(sample_products)

    def test_get_product(self, catalog: ProductCatalog) -> None:
        """Test getting a product by ID."""
        p = catalog.get_product("P001")
        assert p is not None
        assert p.product_id == "P001"
        assert p.title == "Test Wireless Headphones Pro"

    def test_get_product_not_found(self, catalog: ProductCatalog) -> None:
        """Test getting a nonexistent product."""
        assert catalog.get_product("NONEXISTENT") is None

    def test_contains(self, catalog: ProductCatalog) -> None:
        """Test __contains__."""
        assert "P001" in catalog
        assert "NONEXISTENT" not in catalog

    def test_get_all_products(self, catalog: ProductCatalog) -> None:
        """Test retrieving all products."""
        all_p = catalog.get_all_products()
        assert len(all_p) == 10
        ids = {p.product_id for p in all_p}
        assert "P001" in ids

    def test_brand_indexing(self, catalog: ProductCatalog) -> None:
        """Test brand-based lookup."""
        testbrand = catalog.get_by_brand("TestBrand")
        assert len(testbrand) == 3  # P001, P002, P008
        assert all(p.brand == "TestBrand" for p in testbrand)

    def test_brand_case_insensitive(self, catalog: ProductCatalog) -> None:
        """Test brand lookup is case-insensitive."""
        assert len(catalog.get_by_brand("testbrand")) > 0
        assert len(catalog.get_by_brand("TESTBRAND")) > 0

    def test_brand_not_found(self, catalog: ProductCatalog) -> None:
        """Test brand lookup for nonexistent brand."""
        assert catalog.get_by_brand("NonExistent") == []

    def test_category_indexing(self, catalog: ProductCatalog) -> None:
        """Test category-based lookup."""
        electronics = catalog.get_by_category(ProductCategory.ELECTRONICS)
        assert len(electronics) == 10  # All sample products are electronics

    def test_from_json(self) -> None:
        """Test loading catalog from JSON."""
        data = [
            {
                "product_id": "J001",
                "title": "JSON Test Product",
                "brand": "JSONBrand",
                "category": "electronics",
                "price": 99.99,
                "rating": 4.0,
                "review_count": 100,
            }
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            cat = ProductCatalog.from_json(path)
            assert len(cat) == 1
            p = cat.get_product("J001")
            assert p is not None
            assert p.title == "JSON Test Product"
            assert p.brand == "JSONBrand"
        finally:
            Path(path).unlink()

    def test_from_json_file_not_found(self) -> None:
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ProductCatalog.from_json("/nonexistent/path/products.json")

    def test_from_json_invalid_format(self) -> None:
        """Test loading from invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"not": "a list"}')
            path = f.name

        try:
            with pytest.raises(ValueError, match="must be a list"):
                ProductCatalog.from_json(path)
        finally:
            Path(path).unlink()

    def test_from_fixture(self) -> None:
        """Test loading from built-in fixture."""
        cat = ProductCatalog.from_fixture("electronics_sample.json")
        assert len(cat) == 20
        assert "B001" in cat
        sony_products = cat.get_by_brand("Sony")
        assert len(sony_products) == 5

    def test_to_dicts(self, catalog: ProductCatalog) -> None:
        """Test exporting to dictionaries."""
        dicts = catalog.to_dicts()
        assert len(dicts) == 10
        assert isinstance(dicts[0], dict)
        assert "product_id" in dicts[0]

    def test_get_search_texts(self, catalog: ProductCatalog) -> None:
        """Test getting search texts."""
        texts = catalog.get_search_texts()
        assert len(texts) == 10
        assert all(isinstance(t, str) for t in texts)

    def test_empty_catalog_search_texts(self, empty_catalog: ProductCatalog) -> None:
        """Test search texts on empty catalog."""
        assert empty_catalog.get_search_texts() == []

    def test_empty_catalog_get_product_ids(self, empty_catalog: ProductCatalog) -> None:
        """Test product IDs on empty catalog."""
        assert empty_catalog.get_product_ids() == []


class TestHardFiltering:
    """Hard constraint filtering tests."""

    def test_no_filters(self, catalog: ProductCatalog) -> None:
        """Test filtering with no constraints."""
        result = catalog.filter_by_constraints()
        assert len(result) == 10

    def test_budget_filter(self, catalog: ProductCatalog) -> None:
        """Test max budget filtering."""
        result = catalog.filter_by_constraints(max_budget=100.0)
        assert len(result) == 3  # P002 ($49.99), P006 ($89.99), P008 ($19.99)
        for pid in result:
            p = catalog.get_product(pid)
            assert p is not None and p.price is not None
            assert p.price <= 100.0

    def test_min_budget_filter(self, catalog: ProductCatalog) -> None:
        """Test min budget filtering."""
        result = catalog.filter_by_constraints(min_budget=500.0)
        assert len(result) == 2  # P003 ($599.99), P007 ($1299.99)

    def test_budget_range(self, catalog: ProductCatalog) -> None:
        """Test budget range filtering."""
        result = catalog.filter_by_constraints(min_budget=100.0, max_budget=300.0)
        assert all(
            catalog.get_product(pid).price is not None
            and 100.0 <= catalog.get_product(pid).price <= 300.0  # type: ignore
            for pid in result
        )

    def test_brand_filter(self, catalog: ProductCatalog) -> None:
        """Test brand preference filtering."""
        result = catalog.filter_by_constraints(brands=["AudioBrand"])
        # P003, P005, P007 are AudioBrand
        assert len(result) == 3

    def test_excluded_brands(self, catalog: ProductCatalog) -> None:
        """Test brand exclusion."""
        result = catalog.filter_by_constraints(excluded_brands=["TestBrand"])
        assert len(result) == 7  # Exclude P001, P002, P008

    def test_category_filter(self, catalog: ProductCatalog) -> None:
        """Test category filtering."""
        result = catalog.filter_by_constraints(
            categories=[ProductCategory.ELECTRONICS]
        )
        assert len(result) == 10

        result = catalog.filter_by_constraints(
            categories=[ProductCategory.AI_APPLICATION]
        )
        assert len(result) == 0

    def test_min_rating_filter(self, catalog: ProductCatalog) -> None:
        """Test minimum rating filter."""
        result = catalog.filter_by_constraints(min_rating=4.5)
        assert len(result) == 4  # P001 (4.5), P003 (4.8), P004 (4.6), P007 (4.9)

    def test_min_review_count_filter(self, catalog: ProductCatalog) -> None:
        """Test minimum review count filter."""
        result = catalog.filter_by_constraints(min_review_count=3000)
        assert len(result) == 3  # P004 (3400), P007 (3200), P008 (8000)

    def test_excluded_product_ids(self, catalog: ProductCatalog) -> None:
        """Test excluding specific products."""
        result = catalog.filter_by_constraints(excluded_product_ids=["P001", "P002"])
        assert len(result) == 8
        assert "P001" not in result
        assert "P002" not in result

    def test_combined_filters(self, catalog: ProductCatalog) -> None:
        """Test multiple combined filters."""
        result = catalog.filter_by_constraints(
            max_budget=300.0,
            min_rating=4.0,
            brands=["AudioBrand"],
        )
        assert len(result) == 1  # P005 (AudioBrand, $199.99, 4.3)
        assert "P005" in result

    def test_filter_no_results(self, catalog: ProductCatalog) -> None:
        """Test filters that produce no results."""
        result = catalog.filter_by_constraints(max_budget=10.0)
        assert len(result) == 0

    def test_filter_with_specific_ids(self, catalog: ProductCatalog) -> None:
        """Test filtering a specific subset of IDs."""
        result = catalog.filter_by_constraints(
            product_ids=["P001", "P002", "P003"],
            max_budget=100.0,
        )
        assert len(result) == 1  # P002 ($49.99)
        assert "P002" in result
