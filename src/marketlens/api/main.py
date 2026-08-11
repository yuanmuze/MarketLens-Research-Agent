"""FastAPI application entry point for MarketLens Research API."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from marketlens.api.database import init_db
from marketlens.api.routes import init_catalog, router
from marketlens.catalog import ProductCatalog

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — initializes database and catalog."""
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MarketLens API...")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Load default catalog — prefer real data, fall back to fixture
    real_data_path = Path("data/processed/electronics_2000.json")
    fixture_path = Path(__file__).parent.parent / "fixtures" / "electronics_sample.json"

    if real_data_path.exists():
        catalog = ProductCatalog.from_json(real_data_path)
        init_catalog(catalog, data_path=real_data_path)
        logger.info("Loaded %d products from %s", len(catalog), real_data_path)
    elif fixture_path.exists():
        catalog = ProductCatalog.from_fixture("electronics_sample.json")
        init_catalog(catalog)
        logger.info("Loaded %d products from fixture", len(catalog))
    else:
        logger.warning("No catalog data found")
        init_catalog(ProductCatalog())

    yield

    # Shutdown
    logger.info("Shutting down MarketLens API")


app = FastAPI(
    title="MarketLens Research API",
    description="Vertical product research system with evidence-grounded recommendations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request_id to every response."""
    request_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler — returns structured error, never leaks stack traces."""
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "request_id": request.headers.get("X-Request-ID", "unknown"),
        },
    )


# Include routes
app.include_router(router)
