"""Product catalog loading, validation, and management."""

import json
import logging
from pathlib import Path

from marketlens.models import Product, ProductCategory

logger = logging.getLogger(__name__)


class ProductCatalog:
    """In-memory product catalog with validation and search capabilities.

    Loads products from JSON fixtures or programmatic construction.
    Provides indexing for efficient retrieval.
    """

    def __init__(self, products: list[Product] | None = None) -> None:
        """Initialize the catalog with an optional product list.

        Args:
            products: Optional list of Product objects to populate the catalog.
        """
        self._products: dict[str, Product] = {}
        self._brand_index: dict[str, set[str]] = {}
        self._category_index: dict[ProductCategory, set[str]] = {}

        if products:
            for product in products:
                self.add_product(product)

    def add_product(self, product: Product) -> None:
        """Add a single product to the catalog, updating indices.

        Args:
            product: The Product to add.
        """
        self._products[product.product_id] = product

        # Update brand index
        if product.brand:
            brand_lower = product.brand.lower()
            if brand_lower not in self._brand_index:
                self._brand_index[brand_lower] = set()
            self._brand_index[brand_lower].add(product.product_id)

        # Update category index
        if product.category not in self._category_index:
            self._category_index[product.category] = set()
        self._category_index[product.category].add(product.product_id)

    def get_product(self, product_id: str) -> Product | None:
        """Get a product by ID.

        Args:
            product_id: The product identifier.

        Returns:
            The Product if found, else None.
        """
        return self._products.get(product_id)

    def get_all_products(self) -> list[Product]:
        """Get all products in the catalog.

        Returns:
            List of all Product objects.
        """
        return list(self._products.values())

    def get_product_ids(self) -> list[str]:
        """Get all product IDs.

        Returns:
            List of product ID strings.
        """
        return list(self._products.keys())

    def get_by_brand(self, brand: str) -> list[Product]:
        """Get products by brand (case-insensitive).

        Args:
            brand: The brand name.

        Returns:
            List of matching Product objects.
        """
        ids = self._brand_index.get(brand.lower(), set())
        return [self._products[pid] for pid in ids if pid in self._products]

    def get_by_category(self, category: ProductCategory) -> list[Product]:
        """Get products by category.

        Args:
            category: The ProductCategory.

        Returns:
            List of matching Product objects.
        """
        ids = self._category_index.get(category, set())
        return [self._products[pid] for pid in ids if pid in self._products]

    def get_search_texts(self) -> list[str]:
        """Get concatenated search text for all products in order.

        Returns:
            List of search text strings, aligned with get_product_ids().
        """
        return [p.to_search_text() for p in self._products.values()]

    def __len__(self) -> int:
        """Return the number of products in the catalog."""
        return len(self._products)

    def __contains__(self, product_id: str) -> bool:
        """Check if a product ID exists in the catalog."""
        return product_id in self._products

    @classmethod
    def from_json(cls, path: str | Path) -> "ProductCatalog":
        """Load a catalog from a JSON file.

        The JSON file should contain a list of product objects matching
        the Product model schema.

        Args:
            path: Path to the JSON file.

        Returns:
            A ProductCatalog instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON is invalid or products fail validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("Catalog JSON must be a list of product objects")

        products = []
        errors = []
        for i, item in enumerate(raw_data):
            try:
                product = Product(**item)
                products.append(product)
            except Exception as e:
                errors.append(f"Product at index {i}: {e}")

        if errors:
            logger.warning("Skipped %d invalid products: %s", len(errors), errors[:5])

        catalog = cls(products)
        logger.info(
            "Loaded %d products from %s (%d invalid skipped)",
            len(catalog),
            path,
            len(errors),
        )
        return catalog

    @classmethod
    def from_fixture(cls, name: str = "electronics_sample.json") -> "ProductCatalog":
        """Load a catalog from a built-in fixture.

        Args:
            name: The fixture filename (searched in src/marketlens/fixtures/).

        Returns:
            A ProductCatalog instance.
        """
        fixture_dir = Path(__file__).parent / "fixtures"
        fixture_path = fixture_dir / name
        return cls.from_json(fixture_path)

    def to_dicts(self) -> list[dict]:
        """Export all products as dictionaries (for serialization).

        Returns:
            List of product dictionaries.
        """
        return [p.model_dump() for p in self._products.values()]

    def filter_by_constraints(
        self,
        product_ids: list[str] | None = None,
        max_budget: float | None = None,
        min_budget: float | None = None,
        brands: list[str] | None = None,
        excluded_brands: list[str] | None = None,
        categories: list[ProductCategory] | None = None,
        min_rating: float | None = None,
        min_review_count: int | None = None,
        excluded_product_ids: list[str] | None = None,
    ) -> list[str]:
        """Filter product IDs by hard constraints (deterministic).

        Args:
            product_ids: List of product IDs to filter. If None, filters all.
            max_budget: Maximum price.
            min_budget: Minimum price.
            brands: Only include these brands.
            excluded_brands: Exclude these brands.
            categories: Only include these categories.
            min_rating: Minimum rating threshold.
            min_review_count: Minimum review count threshold.
            excluded_product_ids: Product IDs to exclude.

        Returns:
            Filtered list of product IDs.
        """
        if product_ids is None:
            product_ids = list(self._products.keys())

        excluded = set(excluded_product_ids or [])
        brands_lower = {b.lower() for b in (brands or [])} if brands else None
        excluded_lower = {b.lower() for b in (excluded_brands or [])} if excluded_brands else None

        result = []
        for pid in product_ids:
            product = self._products.get(pid)
            if product is None:
                continue
            if pid in excluded:
                continue

            # Budget filter
            if max_budget is not None and product.price is not None and product.price > max_budget:
                continue
            if min_budget is not None and product.price is not None and product.price < min_budget:
                continue

            # Brand filter
            if brands_lower is not None:
                if not product.brand or product.brand.lower() not in brands_lower:
                    continue
            if excluded_lower is not None:
                if product.brand and product.brand.lower() in excluded_lower:
                    continue

            # Category filter
            if categories is not None and product.category not in categories:
                continue

            # Rating filter
            if min_rating is not None:
                if product.rating is None or product.rating < min_rating:
                    continue

            # Review count filter
            if min_review_count is not None:
                if product.review_count is None or product.review_count < min_review_count:
                    continue

            result.append(pid)

        return result
