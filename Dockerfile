FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary pydantic langgraph langchain-core

# Copy source
COPY src/ src/
COPY tests/ tests/

# Run API
CMD ["uvicorn", "marketlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
