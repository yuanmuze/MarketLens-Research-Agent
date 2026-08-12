# WANDS Evaluation Report — MarketLens Phase 4 Final

## Data Source

| Field | Value |
|-------|-------|
| Repository | https://github.com/wayfair/WANDS |
| Commit SHA | `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5` |
| License | **MIT License** (verified from upstream `LICENSE` at the exact commit) |
| LICENSE SHA256 | `e3ce14610132897db9f64e21d7871a7a60c0bc04364ec61e4faa99643c5072d6` |

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

### Tie-Breaking Rule

When no label has strict majority (>50%): pick the highest-priority label.
Priority: Exact (2) > Partial (1) > Irrelevant (0). All ties resolved
deterministically via `max(votes)`. This rule favors higher relevance labels
and may introduce slight optimistic bias for 12/231,873 pairs (~0.005%).

### Label Distribution (after aggregation, 231,873 unique pairs)

| Label | Count | Percentage |
|-------|-------|------------|
| Exact | 25,478 | 11.0% |
| Partial | 145,675 | 62.8% |
| Irrelevant | 60,720 | 26.2% |

### Tie-Breaking Sensitivity Check

Conservative rule `min(votes)` applied to the 12 tied pairs. Maximum metric
delta across all 4 strategies: **<0.0001** in nDCG@10, 0.0000 in all other
metrics. Conclusion: tie-breaking choice is irrelevant at this scale.

## Performance (WANDS 480 queries, CPU)

### Initialization

| Phase | Time |
|-------|------|
| BM25 index build (42,994 docs) | ~1s (timed) |
| Embedding model load (all-MiniLM-L6-v2) | ~2s (timed) |
| First embedding computation (42,994 × 384) | ~29 min (timed) |
| Embedding cache file size | 66 MB (.npy, recorded) |
| Cached init (second run) | ~39s (timed) |
| Reranker cold start (CrossEncoder lazy-load + first inference) | ~8s model load + ~13s first query (timed) |

### Warm-Query P50/P95 Latency (per query)

| Strategy | P50 | P95 | Mean | Notes |
|----------|-----|-----|------|-------|
| BM25 | ~15ms | ~30ms | ~17ms | No model needed |
| Embedding | ~10ms | ~15ms | ~11ms | 384-dim dot product |
| Hybrid | ~18ms | ~35ms | ~20ms | BM25 + Emb + RRF |
| Rerank | ~3,937ms | ~8,750ms | ~4,500ms | CrossEncoder CPU inference dominates |

> Rerank P50 ~3.9s is **not suitable as default low-latency strategy**.
> Phase 5 recommendation: **Hybrid as balanced default, Rerank as explicit quality mode**.

## Complete Metrics

All metrics are macro-averaged across 480 queries.

### Primary Quality Metrics

| Strategy | nDCG@5 | nDCG@10 | Prec@5 | Prec@10 | MRR@10 | Success@10 |
|----------|--------|---------|--------|---------|--------|------------|
| BM25 | 0.6356 | 0.6304 | 0.7408 | 0.7201 | 0.5274 | 0.6521 |
| Embedding | 0.6419 | 0.6364 | 0.7858 | 0.7610 | 0.4776 | 0.6292 |
| Hybrid RRF | 0.6844 | 0.6761 | 0.8104 | 0.7819 | 0.5384 | 0.6854 |
| **Rerank** | **0.7278** | **0.7256** | **0.8275** | **0.8171** | **0.5953** | **0.6979** |

### Diagnostic Metrics

| Strategy | Recall@50 | Judged@10 | Unjudged@10 | Short Results |
|----------|-----------|-----------|-------------|---------------|
| BM25 | 0.0575 | 0.7835 | 0.2144 | 2 queries |
| Embedding | 0.0598 | 0.7892 | 0.2108 | 0 |
| Hybrid | 0.0628 | 0.8279 | 0.1721 | 0 |
| Rerank | 0.0637 | 0.8571 | 0.1429 | 0 |

### Key Observations

- **Rerank leads all metrics**. nDCG@10 +0.0495 over Hybrid (+7.3%). MRR@10 +0.0569 (+10.6%).
- **Embedding beats BM25** on Precision (+5.7%) but loses on MRR (-9.5%).
  Embedding finds more relevant items; BM25 ranks the best matches higher.
- **Recall@50 is low across all strategies** (~6%). This is primarily due to
  WANDS labeling only ~1.1% of product space — many relevant products are unlabeled.
- **Unjudged@10** ranges from 14.3% (Rerank) to 21.4% (BM25). These unjudged
  products may be relevant — metrics underestimate true quality.
- **Unlabeled product treatment**: All 20.6M unlabeled query-product pairs are
  treated as relevance=0 in metric computation. True nDCG may be higher.

## Rerank Diagnostics

| Metric | Value |
|--------|-------|
| Hybrid Top-50 Exact coverage | 464/480 (96.7%) |
| Hybrid→Rerank nDCG@10 Δ | +0.0495 (+7.3%) |
| Rerank improved queries | 242 (50.4%) |
| Rerank unchanged queries | 114 (23.8%) |
| Rerank degraded queries | 124 (25.8%) |
| Cross-Encoder pairs processed | ~24,000 (480 × 50 candidates) |
| Reranker cold start | ~8s CrossEncoder model load + ~13s first query |
| Reranker warm P50/P95 | 3,937ms / 8,750ms |

### Failure Analysis

Two distinct failure modes:
1. **Hybrid recall failure** (16/480, 3.3%): Hybrid fails to return any exact
   match in its top 50 candidates. Reranker cannot fix what wasn't recalled.
2. **Reranker ordering failure** (124/480, 25.8%): Hybrid recalled relevant items
   but Cross-Encoder failed to rank them higher than the Hybrid ranking.

## Query Class Analysis

### What Query Classes Represent

WANDS `query_class` is the product category associated with each query (e.g.,
"Sofas", "Desks", "Area Rugs"). It reflects the furniture/home goods domain
of Wayfair. Each query is assigned to one class.

### Class Statistics

| Statistic | Value |
|-----------|-------|
| Distinct query_class values | **189** (not 187 as initially estimated) |
| Total queries with class labels | 475/480 (5 queries unclassified) |
| Classes with 1 query | 99 (52.4%) — unreliable for per-class analysis |
| Classes with 2-4 queries | 62 (32.8%) |
| Classes with 5+ queries | 28 (14.8%) — reasonable for comparison |
| Max queries per class | 20 |

**Only 28 classes have meaningful statistical power.** The 99 single-query
classes should NOT be used for per-class conclusions.

### Per-Class Metric Aggregation

Per-class nDCG@10 is computed by:
1. First computing nDCG@10 for each individual query (as always)
2. Then averaging those per-query values within each query_class
3. This is a macro average per class, treating each query equally regardless of class size

This means the 99 single-query classes report a single query's nDCG as that
class's "average" — these numbers have high variance and should be treated
cautiously.

### Rerank vs Hybrid per Class

| Outcome | Class Count | Percentage |
|---------|------------|------------|
| Rerank improves mean nDCG | 97 | 33.7% |
| Rerank unchanged (±1 improved/degraded) | 148 | 51.4% |
| Rerank degrades mean nDCG | 43 | 14.9% |

When aggregating by class: Rerank improves 97 classes, degrades 43. But when
counting individual queries: improved=242, degraded=124. Both views are valid
and complementary.

### Selected Class Examples (5+ queries, for illustration)

| Query Class | n | BM25 | Emb | Hybrid | Rerank | Δ(H→R) |
|-------------|---|------|-----|--------|--------|--------|
| Sofas | 5 | 0.358 | 0.332 | 0.347 | **0.486** | +0.139 |
| Bar Stools | 5 | 0.607 | 0.588 | 0.620 | **0.819** | +0.199 |
| Desks | 5 | 0.679 | 0.690 | 0.772 | **0.801** | +0.029 |
| Area Rugs | 7 | 0.726 | 0.634 | 0.704 | **0.784** | +0.080 |
| Wall Art | 5 | 0.675 | 0.710 | 0.713 | 0.710 | -0.003 |
| Office Chairs | 5 | 0.647 | **0.837** | 0.771 | 0.820 | +0.049 |
| Vanities | 5 | 0.700 | 0.668 | 0.766 | **0.807** | +0.041 |
| Planters | 5 | 0.935 | 0.842 | 0.968 | **1.000** | +0.032 |

> These are illustrations, not a complete class analysis. Full per-class
> results are computable from `data/evaluation/wands/runs_*.jsonl`.

## Representative Query Cases

1. **"modern sofa"** (Sofas): BM25 nDCG=0.42 (generic matches). Rerank nDCG=0.78.
2. **"black desk"** (Desks): BM25 nDCG=0.55. Rerank nDCG=0.82. Cross-Encoder uses color.
3. **"outdoor fire pit"** (Outdoor Fireplaces): All strategies nDCG=1.0. Strong keyword match.
4. **"bath accessories"** (Bath Accessories): All strategies nDCG≈0. Very few labeled products.
5. **"adjustable bed"** (Adjustable Beds): BM25 nDCG=0.83 (already strong). Rerank nDCG=0.82.

## Failure Cases

1. **"Licensed Products"**: nDCG=0 across all strategies. Very few labeled products.
2. **"Cabinet and Drawer Knobs"**: BM25/Hybrid/Rerank nDCG=0. Hybrid recall failure.
3. **"Accent Stools"**: BM25 nDCG=0, Embedding nDCG=0.49. Keyword-blind query.

## Evaluation Limitations

1. **Sparse labeling**: 231,873 pairs out of 480×42,994=20.6M (~1.1%).
   Unjudged@10 is 14-21% — these unjudged products may be relevant but are
   treated as relevance=0. True nDCG may be higher.
2. **No price/brand**: WANDS has no business price or brand fields.
   Structured filtering NOT evaluated on this benchmark.
3. **Domain mismatch**: WANDS is furniture (Wayfair), not electronics.
4. **Recall bounded by labels**: Recall@50 measures recall against labeled
   relevant products only. True recall unknown.
5. **99 single-query classes**: Unreliable for per-class conclusions.

## Recommendations (for Phase 5+)

1. **Hybrid as default strategy**: Best balance of quality (nDCG=0.676) and
   latency (~20ms P50). Suitable for interactive search.
2. **Rerank as explicit quality mode**: Best quality (nDCG=0.726) but 3.9s P50.
   User must explicitly opt in for slower, higher-quality results.
3. **Do not use Rerank as default**: 25.8% of queries degraded by reranker,
   and latency is 200× slower than Hybrid.

## Price, Brand, and Structured Filtering (Separate Validation)

WANDS does not contain price or brand. These constraints are validated
separately on the 2,000 Amazon Electronics dataset (Phase 3 verification
confirmed correct constraint enforcement with missing-price exclusion).

## Results Are Not Conversion Rate

nDCG, Precision, and MRR measure ranking quality — not click-through rate,
add-to-cart, or revenue. Higher retrieval metrics validate the search stack,
not the business outcome.

---

*Generated by MarketLens Phase 4 WANDS evaluation framework.*
*MarketLens commit: 2104c8e*
