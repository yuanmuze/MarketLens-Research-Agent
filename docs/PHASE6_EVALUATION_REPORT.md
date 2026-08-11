# Phase 6 Evaluation Report

## Executive Summary

Phase 6 evaluation benchmarks four retrieval strategies on 60 auto-curated queries against a 20-product fixture catalog. Results are **fixture-level only** and do not represent real-world performance.

## Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| Data source | Fixture (20 electronics products) |
| Query count | 60 (9 categories) |
| Embedding backend | FakeEmbeddingBackend (128-dim, hash-based) |
| Top-K | 10 |
| Seed | 42 |
| Python | 3.11.9 |
| OS | Windows (win32) |
| Date | 2026-08-11 |

## Results

| Strategy | Recall@10 | nDCG@10 | Constraint% | NoResult% | P50 (ms) | P95 (ms) | Mean (ms) |
|----------|-----------|---------|-------------|-----------|----------|----------|-----------|
| **BM25** | 1.0000 | 1.0000 | 0.8667 | 0.5000 | 0.0 | 0.0 | 0.0 |
| **Embedding** | 0.6250 | 0.2555 | 0.8000 | 0.4000 | 0.0 | 0.0 | 0.3 |
| **Hybrid RRF** | 0.8750 | 0.6830 | 0.8667 | 0.4000 | 0.0 | 0.0 | 0.5 |
| **Hybrid+Rerank** | 1.0000 | 1.0000 | 0.8667 | 0.4000 | 0.0 | 0.0 | 0.5 |

## Analysis

### BM25 achieves perfect scores
On this small fixture, the 8 exact_match queries are derived directly from product titles in the catalog. BM25's exact keyword matching finds these trivially, producing perfect Recall@10.

### Embedding struggles
FakeEmbeddingBackend (128-dim hash-based) is not a real semantic model. Its low nDCG@10 (0.2555) reflects the lack of actual semantic understanding. This is expected behavior for a fake backend.

### Hybrid RRF recovers partially
RRF fusion raises nDCG from 0.2555 (embedding-only) to 0.6830 by incorporating BM25's strong keyword signals. This demonstrates RRF's robustness to weak individual signals.

### Hybrid+Rerank matches BM25
The KeywordReranker (Jaccard similarity) adds another layer of keyword matching, bringing embedding-based results closer to BM25 quality on this keyword-heavy fixture.

### Constraint satisfaction
All strategies achieve 80-87% constraint satisfaction. The remaining 13% are no_result and contradiction category queries where no products match.

### Latency
All methods complete in <1ms on 20 products — too small a catalog for meaningful latency comparison.

## Label Disclaimer

⚠ **All 60 queries are `review_status: "pending"` and `label_source: "auto_curated"` or `"synthetic"`.** They have NOT been human-reviewed. The relevant_product_ids for exact_match queries are derived from the catalog itself (circular evaluation). For all other categories, relevant_product_ids are empty, inflating the BM25 and Hybrid+Rerank Recall scores.

These results demonstrate the evaluation framework's correctness, not real retrieval quality.

## Running the Evaluation

```bash
# Run the fixture benchmark
uv run python -c "
from marketlens.catalog import ProductCatalog
from marketlens.evaluation.retrieval_comparison import *
from pathlib import Path

catalog = ProductCatalog.from_fixture('electronics_sample.json')
queries = build_eval_queries(catalog)
reports = run_full_comparison(catalog, queries)
save_comparison_results(reports, queries, Path('data/processed'))
print(generate_markdown_report(reports, queries))
"

# Run with real embeddings (if sentence-transformers installed)
uv run python -c "
from marketlens.catalog import ProductCatalog
from marketlens.evaluation.retrieval_comparison import *
from pathlib import Path

catalog = ProductCatalog.from_fixture('electronics_sample.json')
queries = build_eval_queries(catalog)
reports = run_full_comparison(catalog, queries, use_real_embeddings=True)
save_comparison_results(reports, queries, Path('data/processed'))
"
```

## Next Steps for Real Evaluation

1. **Obtain real data**: Run `python scripts/prepare_electronics_data.py --max-products 2000`
2. **Install real embeddings**: `uv pip install sentence-transformers`
3. **Review queries**: Follow `docs/EVALUATION_ANNOTATION_GUIDE.md`
4. **Re-run with real data and real embeddings**
5. **Compare with fixture baseline**
