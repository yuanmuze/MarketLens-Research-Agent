# Evaluation

MarketLens keeps frozen machine-readable configuration in
`benchmarks/manifests/` and local-load evidence in `benchmarks/results/`.
Detailed datasets, query judgments, embeddings, and model files are not stored
in Git.

## Datasets and splits

WANDS uses Wayfair's public dataset at commit
`3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`. The fixed split shuffles sorted
query IDs with seed 42: 288 train, 96 validation, and 96 test queries over
42,994 products. An earlier evaluation used all 480 queries before this split
was defined, so the test subset is not historically untouched.

ESCI uses Amazon Science's Shopping Queries Dataset at commit
`7916cdf6ab75a462e77f20ab40428a10923998d5` under Apache-2.0. Eligible rows are
English-US and `small_version == 1`. SHA-256 ranking with seed 20260814 selects
300 train queries, 100 validation queries from official train, and 100 queries
from official test. The 10,346-product union has pairwise-disjoint query groups.
Source file sizes, hashes, row counts, subset hashes, and selection rules are in
`benchmarks/manifests/esci.json`.

## Retrieval strategies and metrics

The frozen runs compare popularity, BM25, semantic memory, semantic pgvector,
hybrid memory, hybrid pgvector, and quality pgvector at `k=10`. Semantic search
uses `sentence-transformers/all-MiniLM-L6-v2` at pinned revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; quality mode reranks with
`cross-encoder/ms-marco-MiniLM-L-6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`.

- Recall@10 is the fraction of relevant judgments retrieved in the first ten.
- Relevant MRR@10 is the reciprocal rank of the first relevant result, or zero.
- Exact MRR@10 uses the strictest relevance label only.
- nDCG@10 divides discounted cumulative gain by the ideal ordering. WANDS uses
  Exact=2, Partial=1, Irrelevant=0; ESCI uses E=3, S=2, C=1, I=0.
- Latency percentiles cover retrieval calls after the configured warmup.

## Frozen results

| Dataset / strategy | Recall@10 | relevant MRR@10 | exact MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| WANDS BM25 | 0.05461 | 0.82517 | 0.43051 | 0.61309 | 50.14 | 103.13 |
| WANDS semantic memory | 0.06285 | 0.88218 | 0.43116 | 0.66662 | 58.64 | 68.24 |
| WANDS semantic pgvector | 0.06285 | 0.88218 | 0.43116 | 0.66662 | 256.29 | 271.71 |
| WANDS hybrid pgvector | 0.06001 | 0.85965 | 0.44641 | 0.66366 | 311.36 | 367.85 |
| WANDS quality pgvector | 0.06449 | 0.90972 | 0.54203 | 0.72505 | 2171.77 | 5441.87 |
| ESCI BM25 | 0.40173 | 0.83133 | 0.67435 | 0.62460 | 36.58 | 80.24 |
| ESCI semantic memory | 0.43677 | 0.91093 | 0.73254 | 0.65920 | 81.95 | 91.85 |
| ESCI semantic pgvector | 0.43677 | 0.91093 | 0.73254 | 0.65920 | 186.83 | 233.72 |
| ESCI hybrid pgvector | 0.43436 | 0.88278 | 0.71836 | 0.66477 | 127.93 | 155.47 |
| ESCI quality pgvector | 0.46589 | 0.90350 | 0.78875 | 0.71306 | 3856.74 | 4667.29 |

Memory and pgvector rankings matched exactly for semantic and hybrid retrieval
on 96/96 WANDS and 100/100 ESCI queries. The recorded query plan was exact
`Seq Scan + Sort`; approximate HNSW behavior was not evaluated.

## Local API benchmark

`benchmarks/results/load_test.json` records 800/800 valid HTTP 200 responses,
zero transport or payload failures, no memory fallback, real local embeddings,
and zero external LLM calls. It ran in Docker on a 2,000-product catalog with
24 available CPUs and 8.17 GB Docker memory. Hybrid pgvector measured 32.88 RPS
at concurrency 10; quality reranking measured 0.88 RPS and is the observed CPU
bottleneck. These are local development measurements, not production capacity
or latency guarantees.

## Reproduction

Use `scripts/benchmark_wands_backends.py` or
`scripts/benchmark_esci_backends.py` only with the exact source hashes, model
revisions, database coverage, configuration hash, and split described in the
matching manifest. Labels are loaded only for scoring. Do not replace frozen
results with a run that changes data, dependencies, hardware, or parameters;
store such work as a separately scoped experiment.
