"""FastAPI application entry point for MarketLens Research API."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from marketlens.api.database import dispose_db, init_db
from marketlens.api.routes import init_catalog, mark_startup_unavailable, router
from marketlens.catalog import ProductCatalog
from marketlens.config import MarketLensSettings

logger = logging.getLogger(__name__)


def _load_catalog_from_postgres() -> ProductCatalog:
    """Load the product catalog from PostgreSQL via ProductRepository."""
    from marketlens.persistence.engine import session_scope
    from marketlens.persistence.repositories import ProductRepository

    with session_scope() as session:
        repo = ProductRepository(session)
        products = repo.list_products(offset=0, limit=100_000)
    return ProductCatalog(products)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — initializes database and catalog."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MarketLens API...")
    try:
        try:
            settings = MarketLensSettings.from_env()
            init_db()
            logger.info("Database initialized")

            if settings.catalog_backend == "postgres":
                catalog = _load_catalog_from_postgres()
                initialized = init_catalog(
                    catalog,
                    data_path=settings.catalog_path,
                    settings=settings,
                )
                logger.info("Loaded %d products from PostgreSQL", len(catalog))
            else:
                real_data_path = settings.catalog_path or Path(
                    "data/processed/electronics_2000.json"
                )
                fixture_path = (
                    Path(__file__).parent.parent / "fixtures" / "electronics_sample.json"
                )
                if real_data_path.exists():
                    catalog = ProductCatalog.from_json(real_data_path)
                    initialized = init_catalog(
                        catalog,
                        data_path=real_data_path,
                        settings=settings,
                    )
                    logger.info("Loaded %d products from %s", len(catalog), real_data_path)
                elif fixture_path.exists():
                    catalog = ProductCatalog.from_fixture("electronics_sample.json")
                    initialized = init_catalog(catalog, settings=settings)
                    logger.info("Loaded %d products from fixture", len(catalog))
                else:
                    logger.warning("No catalog data found")
                    initialized = init_catalog(ProductCatalog(), settings=settings)

            if not initialized:
                logger.warning("API started live but not ready")
        except Exception as exc:
            logger.error("Startup dependency initialization failed: %s", type(exc).__name__)
            mark_startup_unavailable(
                f"startup dependency unavailable: {type(exc).__name__}"
            )

        # Always yield exactly once so liveness remains available even when an
        # external dependency is temporarily unavailable.
        yield
    finally:
        from marketlens.persistence.engine import reset_engine

        dispose_db()
        reset_engine()
        logger.info("Shutting down MarketLens API")


app = FastAPI(
    title="MarketLens Research API",
    description="Vertical product research system with evidence-grounded recommendations",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS is deliberately static for the process lifetime and fails closed when
# environment configuration is invalid.
_cors_settings = MarketLensSettings.from_env()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_cors_settings.cors_origins),
    allow_credentials=_cors_settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add a unique request_id to every response."""
    request_id = request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return stable structured errors without dependency details or URLs."""
    codes = {
        404: "not_found",
        409: "conflict",
        422: "invalid_request",
        503: "service_unavailable",
    }
    message = str(exc.detail) if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": codes.get(exc.status_code, "http_error"),
            "message": message,
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler — structured error, never leaks stack traces/keys."""
    logger.error("Unhandled error on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown"),
        },
    )


# Include routes
app.include_router(router)
