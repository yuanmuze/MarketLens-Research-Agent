"""Explicit ORM <-> Pydantic conversion helpers.

Avoids implicit coupling between SQLAlchemy records and the
marketlens.models.Product Pydantic model.
"""

from __future__ import annotations

from decimal import Decimal

from marketlens.models import Product, ProductCategory
from marketlens.persistence.models import ProductRecord


def product_to_record(product: Product) -> ProductRecord:
    """Convert a Pydantic Product to a ProductRecord (not persisted)."""
    return ProductRecord(
        product_id=product.product_id,
        title=product.title,
        description=product.description,
        brand=product.brand,
        category=product.category.value if product.category else None,
        price=Decimal(str(product.price)) if product.price is not None else None,
        rating=Decimal(str(product.rating)) if product.rating is not None else None,
        review_count=product.review_count,
        extra={
            "attributes": product.attributes or {},
            "images": product.images or [],
            "url": product.url,
        },
    )


def record_to_product(record: ProductRecord) -> Product:
    """Convert a ProductRecord to a Pydantic Product."""
    meta = record.extra or {}
    category = ProductCategory(record.category) if record.category else ProductCategory.ELECTRONICS
    return Product(
        product_id=record.product_id,
        title=record.title,
        description=record.description,
        brand=record.brand,
        category=category,
        price=float(record.price) if record.price is not None else None,
        rating=float(record.rating) if record.rating is not None else None,
        review_count=record.review_count,
        attributes=meta.get("attributes", {}),
        images=meta.get("images", []),
        url=meta.get("url"),
    )


def record_to_product_dict(record: ProductRecord) -> dict:
    """Convert a ProductRecord to a plain dict matching Product schema."""
    return record_to_product(record).model_dump()
