# Phase 3: Retrieval Core v1 — Report

## Architecture

```
                    ┌──────────────────────────┐
                    │    FastAPI /search        │
                    │  q=, strategy=, top_k=,   │
                    │  brand=, min_rating=, ... │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │    RetrievalService       │
                    │  .search(query, strategy) │
                    └──────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐   ┌────────▼────────┐   ┌───────▼──────┐
   │    BM25     │   │   Embedding     │   │    Rerank    │
   │  keyword    │   │   semantic      │   │  two-stage   │
   │  (Okapi)    │   │  (cosine sim)   │   │  H→Reranker  │
   └──────┬──────┘   └────────┬────────┘   └───────┬──────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │      Hybrid RRF           │
                    │  1/(60+rank_bm25) +       │
                    │  1/(60+rank_emb)          │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │   Structured Filter       │
                    │  brand, price, rating     │
                    │  (missing price excluded) │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │   RetrievalOutput         │
                    │  [RetrievalResultItem...] │
                    └──────────────────────────┘
```

## Four Retrieval Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **BM25** | Okapi BM25 keyword matching (TF × IDF with saturation) | Exact brand names, model numbers, specific keywords |
| **Embedding** | Cosine similarity on 384-dim normalized vectors (all-MiniLM-L6-v2) | Natural language needs, synonyms, semantic intent |
| **Hybrid RRF** | Fuses BM25 + Embedding ranks via Reciprocal Rank Fusion (k=60) | General queries where both precision and recall matter |
| **Rerank** | Hybrid → KeywordReranker rescore on top candidates | When extra precision on top-K matters (adds latency) |

## Why RRF (Not Score Addition)

BM25 scores range from 0-10+. Embedding cosine similarities range from 0-1. Adding them directly would make BM25 dominate. RRF works on **ranks**, which are scale-free:

```
RRF(d) = w1/(K + rank_bm25(d)) + w2/(K + rank_emb(d))
```

A document ranked #1 in both lists gets the highest RRF score, regardless of absolute BM25 or embedding scores.

## Why Two-Stage (Recall → Rerank)

Scanning all 2,000 products with a cross-encoder would be slow (2,000 encode passes per query). Instead:

1. **Stage 1 (Hybrid)**: BM25 + embedding → RRF → top 50 candidates (fast, < 1ms)
2. **Stage 2 (Reranker)**: Score only 50 candidates (50 pairs, fast)

At 2k scale this is not needed, but the architecture is correct for larger catalogs.

## Real Query Examples (Fixture, 20 Products)

| Query | BM25 Top | Embedding Top | Hybrid Top | Rerank Top |
|-------|----------|---------------|------------|------------|
| "Sony wireless headphones noise cancelling" | Sony XM5 ($349) | Sony XM5 ($349) | Sony XM5 ($349) | Sony XM5 ($349) |
| "affordable bluetooth earbuds under $50" | Anker A40 ($79→excluded) | JBL Tour ($299→excluded) | Anker A40 ($79→excluded) | Anker A40 ($79→excluded) |
| "high quality office headphones with mic, rating 4+ under $200" | Sony XM4 ($248→excluded) | Samsung Buds3 ($249→excluded) | Sony XM4 ($248→excluded) | Sony CH720N ($149) |

Note: "excluded" = filtered by constraint, not shown to user. Results vary with real data.

## Timing

| Operation | Fixture (20) | Real (2000, estimated) |
|-----------|-------------|----------------------|
| First init (model + index) | ~40s (model download) | ~40s + 2s embedding |
| Cached init | <1s | <1s (numpy load) |
| BM25 query | <1ms | ~2ms |
| Embedding query | <1ms | ~5ms |
| Hybrid query | <1ms | ~8ms |
| Rerank query | <1ms | ~10ms |

Measured on fixture with real all-MiniLM-L6-v2. Real data timings estimated.

## Embedding Cache

- **Location**: `data/cache/embeddings_{sha256}.npy`
- **Metadata**: `data/cache/embeddings_{sha256}.json`
- **Cache key**: SHA256(data_path + model_name)
- **Invalidation**: Auto on data change or model change
- **Format**: float32 numpy array (2000 × 384)

## Current Limitations

1. **KeywordReranker (Jaccard)**, not Cross-Encoder — ms-marco-MiniLM-L-6-v2 not yet integrated
2. **No human-reviewed eval set** — 50 auto-curated queries pending review
3. **Real 2000-product data** not loaded by API (fixture used in tests)
4. **Missing price = excluded** from price filters, which may be too aggressive for some use cases
5. **No diversity control** — top results may all be from the same brand

## Next Stage: Phase 4 (Evaluation)

1. Complete human review of 50 eval candidates
2. Compute Recall@10, nDCG@10, P50/P95 per strategy on real data
3. Compare BM25 vs Embedding vs Hybrid vs Rerank with real metrics
4. Write evaluation report documenting methodology and results
