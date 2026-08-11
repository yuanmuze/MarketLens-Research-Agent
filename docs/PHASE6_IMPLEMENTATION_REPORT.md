# Phase 6 Implementation Report

## Summary

Phase 6 adds production-ready data pipelines, real embedding support, a comprehensive retrieval comparison framework, and a 60-query auto-curated evaluation set to MarketLens.

## Data Preparation Pipeline

### Script: `scripts/prepare_electronics_data.py`

Streams Amazon Reviews 2023 Electronics metadata from HuggingFace datasets (or local JSONL) and produces validated, deduplicated JSON suitable for `ProductCatalog.from_json()`.

**Usage:**
```bash
# Default: ~2000 products, seed 42
uv run python scripts/prepare_electronics_data.py

# Custom size (max 5000)
uv run python scripts/prepare_electronics_data.py --max-products 5000 --seed 123

# From local file
uv run python scripts/prepare_electronics_data.py --local-file /path/to/data.jsonl

# Dry run (validate only)
uv run python scripts/prepare_electronics_data.py --dry-run
```

**Pipeline steps:**
1. Stream from HuggingFace (`McAuley-Lab/Amazon-Reviews-2023`, `raw_meta_Electronics`) or read local JSONL
2. Shuffle with fixed seed for reproducibility
3. Clean and validate each field: price, rating, review_count, title, parent_asin
4. Deduplicate by product_id
5. Build attributes from features/details arrays
6. Output JSON + manifest (SHA256, provenance, skip stats)

**Data validation rules:**
- Required: `parent_asin` (non-empty), `title` (non-empty)
- Price: strip `$`, parse float, reject negative
- Rating: clamp to [0, 5], reject out-of-range
- Review count: reject negative
- Duplicates: skip by parent_asin

**Output files:**
- `data/processed/electronics_products.json` — Clean product list
- `data/processed/electronics_manifest.json` — Provenance and statistics

## Sentence-Transformers Integration

### Enhanced `SentenceTransformersBackend`

Changes from Phase 1:
- **Batch encoding**: Configurable `batch_size` (default 32) for memory-efficient encoding
- **Model metadata**: `model_info` property returns `{backend_type, model_name, dim, batch_size}`
- **Progress bars**: Shown when encoding >100 texts
- **Normalization control**: Configurable `normalize` flag

```python
from marketlens.retrieval.embedding import SentenceTransformersBackend

backend = SentenceTransformersBackend(
    model_name="all-MiniLM-L6-v2",  # 384-dim, ~80MB, CPU-friendly
    batch_size=32,
    normalize=True,
)
embeddings = backend.encode(["text1", "text2", ...])
print(backend.model_info)  # {"backend_type": "sentence-transformers", ...}
```

**Installation:**
```bash
uv pip install sentence-transformers
```

## Retrieval Comparison Framework

### Module: `src/marketlens/evaluation/retrieval_comparison.py`

Runs BM25, Embedding, Hybrid RRF, and Hybrid+Reranker on identical data/queries/constraints.

**Metrics computed per strategy:**
- Recall@10
- nDCG@10
- Hard constraint satisfaction rate
- No-result query correct handling rate
- P50, P95, mean, min, max latency (ms)

**Output:**
- `data/processed/results_{strategy}.json` — Per-query results
- `data/processed/comparison_summary.json` — Aggregated metrics
- `data/processed/eval_queries.json` — Query definitions
- Markdown report via `generate_markdown_report()`

## Evaluation Query Set

### 60 queries across 9 categories

| Category | Count | Label Source |
|----------|-------|-------------|
| exact_match | 8 | auto_curated (from catalog titles) |
| synonym | 8 | auto_curated (manual paraphrases) |
| brand_filter | 8 | auto_curated |
| budget | 8 | auto_curated |
| multi_constraint | 8 | auto_curated |
| attribute | 6 | auto_curated |
| no_result | 6 | auto_curated |
| contradiction | 4 | auto_curated |
| insufficient_evidence | 4 | auto_curated |

All queries are labeled `review_status: "pending"` and `label_source: "auto_curated"` or `"synthetic"`. None are `human_verified`.

## Fixture Benchmark Results

Run on 2026-08-11 with 20 fixture products, FakeEmbeddingBackend (128-dim):

| Strategy | Recall@10 | nDCG@10 | Constraint% | NoResult% | P50 (ms) | Mean (ms) |
|----------|-----------|---------|-------------|-----------|----------|-----------|
| BM25 | 1.0000 | 1.0000 | 0.8667 | 0.5000 | 0.0 | 0.0 |
| Embedding | 0.6250 | 0.2555 | 0.8000 | 0.4000 | 0.0 | 0.3 |
| Hybrid | 0.8750 | 0.6830 | 0.8667 | 0.4000 | 0.0 | 0.5 |
| Hybrid+Rerank | 1.0000 | 1.0000 | 0.8667 | 0.4000 | 0.0 | 0.5 |

**Interpretation**: BM25 excels at exact keyword matches (the exact_match queries come from the same catalog). Embedding-only performs poorly on this small fixture. Hybrid RRF recovers some of embedding's weakness. On this small fixture, latency differences are negligible.

⚠ **FIXTURE BENCHMARK** — These results are from a 20-product hand-crafted fixture. They are NOT representative of real Amazon data performance.

## Testing

27 new tests in `tests/test_retrieval_comparison.py`:
- Eval query generation and serialization
- Label validation (all synthetic/pending)
- Timing percentile computation
- BM25 embedding comparison strategy reports
- Full four-strategy comparison
- Result save/load roundtrip
- Markdown report generation
- FakeEmbeddingBackend determinism
- SentenceTransformersBackend error handling
- Data pipeline field cleaning
- Product construction validation

All tests run without API keys or model downloads.
