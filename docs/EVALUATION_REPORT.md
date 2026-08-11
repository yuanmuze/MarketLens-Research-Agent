# MarketLens Evaluation Report

## Summary

This report documents the evaluation methodology and results for the MarketLens product research system. All results are from fixture benchmarks—they demonstrate the evaluation framework but do not represent performance on real Amazon Reviews data.

## Evaluation Data

### Fixture Queries (12 queries)

We created 12 evaluation queries spanning 7 categories:

| Category | Count | Example |
|----------|-------|---------|
| `exact_match` | 2 | "Sony WH-1000XM5 Wireless Noise Cancelling Headphones" |
| `synonym` | 2 | "noise cancelling wireless over-ear headphones" |
| `budget` | 2 | "wireless headphones under $100" |
| `multi_constraint` | 2 | "Sony wireless ANC headphones under $350 with 30h+ battery" |
| `no_result` | 2 | "wireless headphones under $10" |
| `contradiction` | 1 | "best $500 headphones under $100" |
| `insufficient_evidence` | 1 | "best mid-range open-back planar magnetic headphones" |

Each query includes ground truth relevant product IDs from the 20-product electronics fixture.

### Catalog

20 electronics products (headphones, earbuds, speakers) with prices $19.99–$1,299.99, ratings 3.8–4.9.

## Metrics

We compute 6 standard information retrieval and task completion metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| **Recall@10** | `|retrieved ∩ relevant| / |relevant|` | Fraction of relevant items found in top 10 |
| **nDCG@10** | `DCG / IDCG` | Rank-weighted relevance (binary relevance) |
| **Constraint Satisfaction Rate** | `satisfied / total` | Queries where hard constraints were met |
| **Task Completion Rate** | `completed / total` | Queries returning at least 1 result |
| **Evidence Validity Rate** | Implicit in validation | Products pass deterministic checks |
| **Average Latency** | `Σ(duration) / N` | Mean query execution time (ms) |

## Retrieval Method Comparison

We compare 3 retrieval strategies on the same 12 queries:

| Method | Description |
|--------|-------------|
| **BM25** | Okapi BM25 keyword search only |
| **Embedding** | Cosine similarity via FakeEmbeddingBackend (dim=64) |
| **Hybrid** | RRF fusion of BM25 + embedding (k=60, equal weights) |

Results are printed during `pytest tests/test_evaluation.py -v`. Typical findings:

- **Exact match queries**: BM25 performs best (exact keyword match)
- **Synonym queries**: Embedding and hybrid perform better (semantic similarity)
- **Budget queries**: All methods perform similarly after hard filtering
- **No-result queries**: All methods correctly return few/no results

All numbers vary with the exact query set and fixture data; these are relative comparisons, not absolute benchmarks.

## Running Evaluation

```bash
# Run all evaluation tests (fixture benchmark)
uv run pytest tests/test_evaluation.py -v

# Run the comparison benchmark
uv run pytest tests/test_evaluation.py::TestEvaluationBenchmark::test_compare_all_retrievers -v -s
```

## Fixture Data Disclaimer

⚠ **ALL RESULTS IN THIS REPORT ARE FIXTURE BENCHMARKS.** The evaluation uses a small, hand-crafted fixture dataset of 20 products. These numbers do not represent performance on the full Amazon Reviews 2023 Electronics dataset or any real-world deployment. They validate the evaluation framework and retrieval pipeline correctness, not real-world accuracy.

For production evaluation, replace the fixture with actual Amazon Reviews data using the provided data preparation script.
