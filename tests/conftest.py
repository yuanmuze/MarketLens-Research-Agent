"""Shared pytest fixtures for MarketLens tests."""

import pytest

from marketlens.catalog import ProductCatalog
from marketlens.models import Product, ProductCategory

SAMPLE_PRODUCTS = [
    Product(
        product_id="P001",
        title="Test Wireless Headphones Pro",
        brand="TestBrand",
        category=ProductCategory.ELECTRONICS,
        price=299.99,
        rating=4.5,
        review_count=1000,
        description="Great wireless headphones with noise cancellation.",
        attributes={"color": "Black", "battery_life": "30h"},
    ),
    Product(
        product_id="P002",
        title="Budget Wireless Earbuds",
        brand="TestBrand",
        category=ProductCategory.ELECTRONICS,
        price=49.99,
        rating=4.0,
        review_count=500,
        description="Affordable earbuds with good sound quality.",
        attributes={"color": "White", "battery_life": "20h"},
    ),
    Product(
        product_id="P003",
        title="Premium Over-Ear Studio Headphones",
        brand="AudioBrand",
        category=ProductCategory.ELECTRONICS,
        price=599.99,
        rating=4.8,
        review_count=2500,
        description="Reference studio headphones with crystal clear audio.",
        attributes={"color": "Silver", "battery_life": "N/A"},
    ),
    Product(
        product_id="P004",
        title="Smart Voice Assistant Speaker",
        brand="SmartBrand",
        category=ProductCategory.ELECTRONICS,
        price=129.99,
        rating=4.6,
        review_count=3400,
        description="Smart speaker with AI voice assistant and multi-room audio.",
        attributes={"color": "Charcoal", "smart_home": "Yes"},
    ),
    Product(
        product_id="P005",
        title="AI-Powered Noise Cancelling Earbuds",
        brand="AudioBrand",
        category=ProductCategory.ELECTRONICS,
        price=199.99,
        rating=4.3,
        review_count=750,
        description="Earbuds with AI adaptive noise cancellation. 40h battery.",
        attributes={"color": "Blue", "battery_life": "40h"},
    ),
    Product(
        product_id="P006",
        title="Wireless Gaming Headset RGB",
        brand="GameBrand",
        category=ProductCategory.ELECTRONICS,
        price=89.99,
        rating=4.2,
        review_count=1200,
        description="Low latency wireless gaming headset with RGB lighting.",
        attributes={"color": "Black", "battery_life": "25h"},
    ),
    Product(
        product_id="P007",
        title="Ultra-Premium Wireless Audio System",
        brand="AudioBrand",
        category=ProductCategory.ELECTRONICS,
        price=1299.99,
        rating=4.9,
        review_count=3200,
        description="Flagship wireless audio system with room calibration.",
        attributes={"color": "Gold", "battery_life": "N/A", "power": "AC"},
    ),
    Product(
        product_id="P008",
        title="Basic Wired Earbuds",
        brand="TestBrand",
        category=ProductCategory.ELECTRONICS,
        price=19.99,
        rating=3.8,
        review_count=8000,
        description="Simple wired earbuds. No frills, just sound.",
        attributes={"color": "White", "battery_life": "N/A"},
    ),
    Product(
        product_id="P009",
        title="Sports Wireless Earbuds Pro",
        brand="SportBrand",
        category=ProductCategory.ELECTRONICS,
        price=159.99,
        rating=4.4,
        review_count=890,
        description="Waterproof wireless earbuds for sports. Secure fit. 35h battery.",
        attributes={"color": "Red", "battery_life": "35h", "waterproof": "IPX7"},
    ),
    Product(
        product_id="P010",
        title="AI Translation Earbuds",
        brand="AIBrand",
        category=ProductCategory.ELECTRONICS,
        price=179.99,
        rating=4.1,
        review_count=320,
        description="Real-time AI translation earbuds supporting 40 languages.",
        attributes={"color": "Black", "battery_life": "22h", "ai_feature": "translation"},
    ),
]


@pytest.fixture
def sample_products() -> list[Product]:
    """Return a list of 10 sample products for testing."""
    return [p.model_copy() for p in SAMPLE_PRODUCTS]


@pytest.fixture
def catalog(sample_products: list[Product]) -> ProductCatalog:
    """Return a ProductCatalog with sample products."""
    return ProductCatalog(sample_products)


@pytest.fixture
def empty_catalog() -> ProductCatalog:
    """Return an empty ProductCatalog."""
    return ProductCatalog()


@pytest.fixture
def single_product_catalog() -> ProductCatalog:
    """Return a catalog with a single product."""
    return ProductCatalog([SAMPLE_PRODUCTS[0].model_copy()])
