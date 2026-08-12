# WANDS Evaluation Report — MarketLens Phase 4 Final

## Data Source

| Field | Value |
|-------|-------|
| Repository | https://github.com/wayfair/WANDS |
| Commit SHA | `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5` |
| License | CC BY-NC 4.0 |

### File Hashes (SHA256)

| File | SHA256 |
|------|--------|
| product.csv | `d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6` |
| query.csv | `63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95` |
| label.csv | `c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef` |

### Counts

| Metric | Count |
|--------|-------|
| Products | 42,994 |
| Queries | 480 |
| Raw label rows | 233,448 |
| Unique query-product pairs | 231,873 |
| Multi-annotator duplicates | 1,575 |

## Multi-Annotator Label Audit

| Statistic | Count |
|-----------|-------|
| Single-annotator pairs | 230,406 |
| Multi-annotator pairs | 1,467 |
| All annotators agree | 1,453 |
| Has majority (some disagree) | 2 |
| No strict majority (tie-broken) | 12 |

Vote counts: 1 annotator=230,406, 2=1,390, 3=49, 4=25, 5=3.

**Tie-breaking rule**: When no label has strict majority (>50%):
pick the highest-priority label. Priority: Exact (2) > Partial (1) >
Irrelevant (0). All ties resolved deterministically via `max(votes)`.

### Label Distribution (after aggregation, 231,873 unique pairs)

| Label | Count | Percentage |
|-------|-------|------------|
| Exact | 25,478 | 11.0% |
| Partial | 145,675 | 62.8% |
| Irrelevant | 60,720 | 26.2% |

## Latency Chart (WANDS 480 queries, CPU)

### Initialization

| Phase | Time |
|-------|------|
| BM25 index build (42,994 docs) | ~1s |
| Embedding model load (all-MiniLM-L6-v2) | ~2s |
| First embedding computation (42,994 × 384) | ~29 min |
| Embedding cache file size | 66 MB (.npy) |
| Cached init (second run) | ~39s |
| Reranker cold start (CrossEncoder lazy-load) | ~8s |

### Warm-Query P50/P95 Latency (per query, excluding cold start)

| Strategy | P50 | P95 | Mean |
|----------|-----|-----|------|
| BM25 | ~15ms | ~30ms | ~17ms |
| Embedding | ~10ms | ~15ms | ~11ms |
| Hybrid | ~18ms | ~35ms | ~20ms |
| Rerank | ~3,937ms | ~8,750ms | ~4,500ms |

> Note: Rerank latency is dominated by CrossEncoder CPU inference (480 candidates × ~8ms each).

## Metrics Update

All metrics are macro-averaged across 480 queries.

| Strategy | nDCG@10 | Prec@10 | MRR@10 | Success@10 | Recall@50 |
|----------|---------|---------|--------|------------|-----------|
| BM25 | 0.6304 | 0.7201 | 0.5274 | 0.6521 | 0.0575 |
| Embedding | 0.6364 | 0.7610 | 0.4776 | 0.6292 | 0.0598 |
| Hybrid RRF | 0.6761 | 0.7819 | 0.5384 | 0.6854 | 0.0628 |
| Rerank | **0.7256** | **0.8171** | **0.5953** | **0.6979** | **0.0637** |

Rerank shows consistent improvement over Hybrid across all metrics.
nDCG@10: +0.0495 (+7.3%). Exact MRR@10: +0.0569 (+10.6%).

## Rerank Diagnostics

| Metric | Value |
|--------|-------|
| Hybrid Top-50 Exact coverage | 464/480 (96.7%) |
| Hybrid→Rerank improved queries | 242 |
| Unchanged | 114 |
| Degraded | 124 |
| Cross-Encoder pairs processed | ~24,000 (480 × 50 candidates) |

### Failure Analysis

Two distinct failure modes:
1. **Hybrid recall failure** (16/480, 3.3%): Hybrid fails to return any exact match
   in its top 50 candidates. Reranker cannot fix what wasn't recalled.
2. **Reranker ordering failure** (124/480, 25.8%): Hybrid recalled relevant items
   but Cross-Encoder failed to rank them higher.

## Per Query Class nDCG@10

| Class | BM25 | Emb | Hybrid | Rerank |
|-------|------|-----|--------|--------|
| Accent Chairs | 0.485 | 0.523 | 0.496 | **0.566** |
| Area Rugs | 0.726 | 0.634 | 0.704 | **0.784** |
| Bar Stools | 0.607 | 0.588 | 0.620 | **0.819** |
| Bathroom Sink Faucets | 0.696 | 0.617 | 0.735 | 0.724 |
| Beds | 0.544 | 0.606 | 0.615 | **0.735** |
| Desks | 0.679 | 0.690 | 0.772 | **0.801** |
| Office Chairs | 0.647 | 0.837 | 0.771 | 0.820 |
| Sofas | 0.358 | 0.332 | 0.347 | **0.486** |
| TV Stands | 0.287 | 0.263 | 0.335 | **0.383** |
| Wall Art | 0.675 | 0.710 | 0.713 | 0.710 |

(Selected 10 representative classes out of 187. Full table in data output.)

## Representative Query Cases

### Case 1: "modern sofa" (Sofas)
BM25: nDCG=0.42, all generic. Rerank: nDCG=0.78, top results are higher quality.

### Case 2: "black desk" (Desks)
BM25: nDCG=0.55. Rerank: nDCG=0.82. Cross-Encoder correctly identifies black-colored desks.

### Case 3: "outdoor fire pit" (Outdoor Fireplaces)
All strategies: nDCG=1.0. Strong keyword match in product names.

### Case 4: "bath accessories" (Bath Accessories)
BM25/Hybrid/Rerank: nDCG=0. Very few products match. Embedding: nDCG=0.07.

### Case 5: "adjustable bed" (Adjustable Beds)
BM25: nDCG=0.83. Rerank: nDCG=0.82. Keyword match already strong.

## Failure Cases

1. **"Licensed Products"**: All strategies nDCG=0. Very few labeled products in this category.
2. **"Cabinet and Drawer Knobs"**: BM25 nDCG=0, Rerank nDCG=0. Hybrid recall failure — no relevant items in top 50.
3. **"Accent Stools"**: BM25 nDCG=0. Embedding nDCG=0.49. BM25 fails on less keyword-focused queries where embedding catches synonyms.

## Evaluation Limitations

1. **Incomplete labeling**: 231,873 labeled pairs out of 480×42,994=20.6M combinations (~1.1%).
   Top-10 unjudged ratio is 14.3-21.4% depending on strategy. Unjudged products are
   treated as irrelevant (0) in metrics — this underestimates true relevance at high ranks.
2. **No price/brand data**: WANDS has no business price or brand fields.
   Structured filtering (budget, brand, rating) NOT evaluated on this benchmark.
3. **Domain mismatch**: WANDS is furniture (Wayfair), not electronics.
   Retrieval behavior may differ by product category.
4. **Recall bounded by known labels**: Recall@50 measures recall of labeled
   relevant products only — true recall against ALL relevant products is unknown.

## Labeling Limitations

WANDS labels cover only ~1.1% of all possible query-product pairs. An unjudged
product in the top 10 may or may not be relevant. The 14-21% unjudged rate means
metrics underestimate true retrieval quality, especially at high ranks.

## Price, Brand, and Structured Filtering

WANDS does not contain price or brand fields. These constraints are validated
separately on the 2,000 Amazon Electronics dataset, which has price/brand/rating
fields and correct constraint exclusion logic.

## Results Are Not Conversion Rate

nDCG, Precision, and MRR measure ranking quality of a retrieval pipeline — not
user click-through rate, add-to-cart rate, or revenue. Higher retrieval metrics
validate the search stack, not the business outcome.

---

*Generated by MarketLens Phase 4 WANDS evaluation framework.*
*MarketLens commit: see data/evaluation/wands/metadata_v1.json*
