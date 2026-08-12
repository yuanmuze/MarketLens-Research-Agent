# Phase 3.1: Retrieval Core — Real Data Integration & Cross-Encoder

## Retrieval Chain

```
query → _apply_structured_filter() → candidate_ids
    → strategy dispatch:
        bm25:      _search_bm25(query, top_k, candidate_ids)
        embedding: _search_embedding(query, top_k, candidate_ids)
        hybrid:    _search_hybrid(query, top_k, candidate_ids)  # RRF k=60
        rerank:    _search_hybrid(query, candidate_k) → _rerank_candidates → sort
    → RetrievalOutput with unified [RetrievalResultItem...]
```

## Real Models Used

| Component | Model | Loaded |
|-----------|-------|--------|
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (384-dim) | ✅ Yes |
| Reranker | CrossEncoder(cross-encoder/ms-marco-MiniLM-L-6-v2) | ✅ Yes |

## Cross-Encoder Status

- **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2 (loaded via sentence_transformers.CrossEncoder)
- **Interface**: `.predict(pairs)` → list of relevance scores
- **Scope**: Only `candidate_k` candidates from Hybrid retrieval (not full catalog)
- **Lazy loading**: Model loaded on first rerank query, reused across requests

## No Silent Fake Fallback

- **Embedding**: `_create_real_backend()` now raises `RuntimeError` on failure instead of silently returning `FakeEmbeddingBackend`
- **Reranker**: `_rerank_candidates()` raises on CrossEncoder failure
- **Explicit fake mode**: Must pass `use_fake_embeddings=True` to use fake backends
- **Tests**: All unit tests explicitly pass `FakeEmbeddingBackend` or `use_fake_embeddings=True`

## Cache Fingerprint

Cache key composed from:
- Data file content SHA256
- Model name
- Text schema version (`TEXT_SCHEMA_VERSION = "v1"`)
- Product count
- Embedding dimension

Cache metadata records: `data_sha256`, `text_schema_version`, `model_name`, `dim`, `count`, `dtype`, `created`.

Cache loads validated against: product count, embedding dimension, dtype (float32), row count.

## Structured Filtering

- **Position**: Pre-filter — `_apply_structured_filter()` runs before retrieval
- **Strategy**: Products violating any constraint never enter the candidate pool
- **Missing price**: Excluded from price-filtered queries (cannot verify compliance)
- **Brand**: Case-insensitive exact match
- **Empty results**: Returned as-is (never padded with violating products)

## 2000-Product Demo (Real Data)

### Timing

| Operation | Time |
|-----------|------|
| First init (model download + embed 2000) | ~124s |
| Cached init (numpy load + BM25 build) | ~25s |
| Embedding query | ~2ms |
| BM25 query | ~3ms |
| Hybrid query | ~5ms |
| Rerank query (first, model lazy-load) | ~13s |
| Rerank query (subsequent) | ~500ms |

### 5 Query Demo Results (Top Product per Strategy)

| Query | BM25 | Embedding | Hybrid | Rerank |
|-------|------|-----------|--------|--------|
| Sony NC headphones | Sony XM5 ANC headphones | Sony XM5 ANC headphones | Sony XM5 ANC headphones | Sony XM5 ANC headphones |
| Earbuds for running | Sport wireless earbuds | Bluetooth sport earbuds | Sport wireless earbuds | Sport wireless earbuds |
| Bluetooth under $50 | Wireless earbuds $29.99 | Bluetooth headphones $49.99 | Wireless earbuds $29.99 | Wireless earbuds $29.99 |
| Premium office 4.0★+ | Office headset 4.3★ | Office headset 4.3★ | Office headset 4.3★ | Office headset 4.3★ |
| Quantum GPU PCIe card | 0 results | 0 results | 0 results | 0 results |

**Note**: No human ground truth exists. These are relative comparisons, not quality judgments.

## Constraint Enforcement Verified

- max_budget=50.0 → all results have price ≤ $50.00
- max_budget=0.01 → 0 results (no violations filled)
- min_rating=4.5 → all results have rating ≥ 4.5
- brand=Sony → all results brand = "Sony" (case-insensitive)
- Missing price products never appear in price-filtered results

## Current Limitations

1. No human-reviewed evaluation set (50 candidates pending review)
2. Reranker uses CPU-only (no GPU acceleration)
3. First rerank query lazy-loads model (~13s cold start)
4. Cache invalidates on any data change (full recompute)
5. No result diversity control (same brand may dominate)
