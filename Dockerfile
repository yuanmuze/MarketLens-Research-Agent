# MarketLens API image (Python 3.12, non-root).
FROM python:3.12-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy source (marketlens package is importable via PYTHONPATH)
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini ./

# Install runtime dependencies directly (avoids building the project wheel,
# which references the tests/ directory excluded from the image).
RUN pip install --no-cache-dir \
    fastapi uvicorn pydantic pydantic-settings \
    sqlalchemy alembic psycopg2-binary pgvector \
    langchain-core langchain langgraph httpx numpy

# marketlens package lives under src/
ENV PYTHONPATH=/app/src

# Switch to non-root
USER appuser

EXPOSE 8000

CMD ["uvicorn", "marketlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
