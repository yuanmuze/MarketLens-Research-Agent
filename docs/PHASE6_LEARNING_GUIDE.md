# Phase 6 Learning Guide — Real Data, Real Embeddings, Real Evaluation

Target audience: Python/AI backend beginners who have completed Phase 1-5.

## 1. What Problem Does Phase 6 Solve?

Phase 1-5 built a working product research system on a 20-product hand-crafted fixture. Phase 6 makes it ready for real-world use:

- **Real data**: Amazon Reviews 2023 Electronics — actual products people buy
- **Real embeddings**: sentence-transformers with all-MiniLM-L6-v2 (384-dim, CPU-friendly)
- **Real evaluation**: 60-query benchmark comparing 4 retrieval strategies
- **Real metrics**: Recall@10, nDCG@10, constraint rates, P50/P95 latency

## 2. The Complete Data Processing Chain

```
Amazon Reviews 2023 (HuggingFace)
    ↓ streaming read (no full download)
Raw metadata rows (parent_asin, title, price, rating, features...)
    ↓ clean_price, clean_rating, clean_review_count
    ↓ build_product: normalize fields, extract attributes
    ↓ deduplicate by parent_asin
Validated Product dicts
    ↓ json.dump
electronics_products.json
    ↓ ProductCatalog.from_json()
In-memory catalog with BM25 + embedding indices
    ↓ HybridRetriever.search()
Search results with scores and evidence
```

**Key insight**: The pipeline converts messy real-world data into clean, validated Product objects. Every step has error handling — missing prices become `None`, invalid ratings are rejected, duplicates are skipped.

## 3. BM25: The Intuitive Explanation

BM25 (Best Match 25) is a bag-of-words ranking function. Think of it as:

> "How many times do query words appear in this document, penalized if those words are common across all documents, and penalized if this document is unusually long?"

The formula:
```
score(D, Q) = Σ IDF(qi) × TF(qi, D) / (TF(qi, D) + normalization)
```

- **TF (Term Frequency)**: More mentions → higher score, but with diminishing returns (k1 parameter)
- **IDF (Inverse Document Frequency)**: Common words like "the" contribute less than rare words like "Sennheiser"
- **Length normalization**: Long documents are penalized (b parameter)

BM25 is:
- ✅ Fast, deterministic, no training
- ✅ Great for exact keyword matches ("Sony WH-1000XM5" → product with that exact name)
- ❌ No semantic understanding ("earbuds" ≠ "in-ear headphones" to BM25)

## 4. Embeddings and Cosine Similarity

An embedding is a dense vector (list of floats) that represents the meaning of text.

```python
"wireless noise cancelling headphones" → [0.12, -0.34, 0.08, ..., 0.21]  # 384 numbers
"ANC bluetooth over-ear headset"      → [0.10, -0.31, 0.12, ..., 0.18]  # Similar!
"refrigerator freezer combo"          → [-0.45, 0.67, -0.89, ..., -0.12] # Very different!
```

**Cosine similarity** measures the angle between two vectors:
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```
- 1.0 = identical direction (same meaning)
- 0.0 = orthogonal (unrelated)
- -1.0 = opposite direction

All-MiniLM-L6-v2 is a lightweight model (~80MB) that maps sentences to 384-dimensional vectors. It's trained to put similar sentences close together.

## 5. BM25 vs Embedding: When to Use Each

| Aspect | BM25 | Embedding |
|--------|------|-----------|
| **Exact product names** | Excellent | Good |
| **Synonyms** | Poor ("earbuds" ≠ "earphones") | Good |
| **Typo handling** | Poor | Moderate |
| **Cross-lingual** | No | Some models support it |
| **Speed** | Very fast (no GPU needed) | Fast on CPU, faster on GPU |
| **Interpretability** | Why was this ranked high? (TF, IDF explain it) | Black box |
| **Cold start** | Works immediately | Needs model download |

## 6. Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists into one. The idea:

> "A document that appears near the top in multiple ranking lists is probably relevant."

```
RRF_score(d) = w1/(k + rank1(d)) + w2/(k + rank2(d))
```

- `k`: Smoothing constant (typically 60). Prevents extreme weighting of #1 vs #2.
- `w1, w2`: Weights for each ranker (default 0.5 each).

Example:
```
BM25 ranks:         [A=1st, B=2nd, C=3rd]
Embedding ranks:    [B=1st, C=2nd, A=3rd]

RRF(A) = 0.5/(60+1) + 0.5/(60+3) = 0.00820 + 0.00794 = 0.01614
RRF(B) = 0.5/(60+2) + 0.5/(60+1) = 0.00806 + 0.00820 = 0.01626  ← Winner!
RRF(C) = 0.5/(60+3) + 0.5/(60+2) = 0.00794 + 0.00806 = 0.01600
```

B wins because it's consistently high in both lists, even though it's not #1 in either.

## 7. What a Reranker Does

After initial retrieval returns top-K candidates, a reranker rescore them with more expensive computation:

- **KeywordReranker** (used in MarketLens): Jaccard similarity between query tokens and document tokens. Fast, no model.
- **Cross-encoder** (not yet implemented): Feeds (query, document) pairs through a transformer. Much slower, much better.

```python
# Without reranker: 1000 docs → top 100 by BM25 → top 10 returned
# With reranker:    1000 docs → top 100 by BM25 → reranker scores 100 → top 10 returned
```

## 8. Recall@10 and nDCG@10 — Intuitive Explanation

**Recall@10**: "Out of all relevant products, how many did we find in the top 10?"

```
Catalog has 5 Sony headphones. You search "Sony headphones".
Top 10 results contain 3 Sony products. → Recall@10 = 3/5 = 0.60
```

**nDCG@10**: "Are the best products at the top of the list?"

```
Perfect ranking: [Relevant, Relevant, Relevant, Irrelevant, Irrelevant] → nDCG = 1.0
Poor ranking:    [Irrelevant, Irrelevant, Relevant, Relevant, Relevant] → nDCG < 1.0
```

nDCG penalizes highly relevant products appearing late in the results.

## 9. P50 and P95 Latency

- **P50 (median)**: Half of queries complete faster than this. "Typical" experience.
- **P95**: 95% of queries complete faster than this. "Worst reasonable case" — ignore the slowest 5%.

Why P95 matters: A search system might have 10ms median but 5000ms P95 (some queries timeout). Users remember the worst experience, not the average.

## 10. 8 Core Files to Read (in order)

1. `scripts/prepare_electronics_data.py` — Real data pipeline
2. `src/marketlens/retrieval/embedding.py` — SentenceTransformersBackend
3. `src/marketlens/evaluation/retrieval_comparison.py` — Comparison framework
4. `src/marketlens/evaluation/benchmark.py` — Recall, nDCG metrics
5. `src/marketlens/retrieval/bm25.py` — BM25 algorithm
6. `src/marketlens/retrieval/hybrid.py` — RRF fusion
7. `src/marketlens/retrieval/reranker.py` — KeywordReranker
8. `docs/PHASE6_EVALUATION_REPORT.md` — Fixture benchmark results

## 11. A Query's Complete Path Through the System

```
"best Sony noise cancelling headphones under $300"
    ↓ build_eval_queries creates EvalQuery with constraints={max_budget: 300}
    ↓ run_full_comparison runs 4 strategies
    ↓ For BM25:
        _apply_constraints filters catalog to products ≤ $300
        bm25.search(query, k=20) → ranked list with scores
        Top 10 returned → [product_ids], [scores]
    ↓ For Embedding:
        emb_retriever.search(query, k=20) → cosine similarity
        Filter to budget-constrained candidates
        Top 10 returned
    ↓ For Hybrid:
        HybridRetriever.search(SearchQuery(text=query, filters=..., use_reranker=False))
        BM25 ranking + Embedding ranking → RRF fusion → sorted by fused score
    ↓ For Hybrid+Rerank:
        Same as Hybrid, then KeywordReranker scores top candidates by Jaccard
    ↓ compute_strategy_report computes Recall@10, nDCG@10, timing percentiles
    ↓ save_comparison_results writes JSON files to data/processed/
    ↓ generate_markdown_report produces the comparison table
```

## 12. 15 Interview Questions and Answers

### Data Pipeline
1. **Q**: Why stream from HuggingFace instead of downloading the full dataset?
   **A**: The full Amazon Reviews 2023 is hundreds of GB. Streaming reads only what we need (2000-5000 products), saving bandwidth and disk. The `streaming=True` flag in `datasets.load_dataset` makes this possible.

2. **Q**: How do you ensure data quality?
   **A**: Multi-step validation: required field checks, type coercion (price: strip `$`, parse float), range validation (rating 0-5, count ≥ 0), deduplication (by parent_asin), and skip statistics tracking.

3. **Q**: Why SHA256 in the manifest?
   **A**: Cryptographic hash of the output file proves the data hasn't been tampered with. If someone else runs the same script with the same seed, they should get the same SHA256.

### Embeddings
4. **Q**: Why all-MiniLM-L6-v2 instead of a larger model?
   **A**: It's 80MB, runs on CPU, produces 384-dim vectors, and is fast enough for interactive use. For a demo/portfolio project, lightweight is better than state-of-the-art. Production would use a larger model.

5. **Q**: Why L2-normalize embedding vectors?
   **A**: Normalization makes all vectors unit length, so cosine similarity = dot product. This is both mathematically convenient and practically faster (dot product is a single numpy operation).

6. **Q**: What happens if the model isn't installed?
   **A**: The system falls back to FakeEmbeddingBackend. The `SentenceTransformersBackend.__init__` doesn't load the model immediately — it lazy-loads on first `encode()` call. If import fails, it raises `ImportError` with a clear install instruction.

### Retrieval Comparison
7. **Q**: Why is BM25 perfect (Recall=1.0) on the fixture benchmark?
   **A**: The exact_match queries are constructed directly from product titles in the same catalog. BM25 finds them trivially. This is a known artifact of fixture evaluation and would not hold on real data.

8. **Q**: Why does embedding perform poorly on the fixture?
   **A**: FakeEmbeddingBackend uses hash-based random projection, not real semantics. It has no understanding of synonyms or meaning. Real embeddings would perform much better.

9. **Q**: When would Hybrid underperform BM25?
   **A**: When the embedding signal is noisy or misleading. If the embedding model doesn't understand the domain (e.g., electronics jargon), it may rank irrelevant products high, and RRF may dilute BM25's correct rankings.

### Evaluation
10. **Q**: What's the difference between fixture, auto-curated, and human-verified?
    **A**: Fixture = hand-crafted toy data. Auto-curated = programmatically generated queries and labels. Human-verified = a human has checked and approved each query's relevance judgments. Only human-verified should be used for credible benchmarks.

11. **Q**: Why track P95 latency, not just mean?
    **A**: Mean is skewed by outliers. If 95% of queries are fast but 5% take 10 seconds, mean looks fine but users are unhappy. P95 captures the worst reasonable case.

12. **Q**: How do you know the constraint satisfaction rate is correct?
    **A**: Hard constraints are enforced by plain Python in `catalog.filter_by_constraints()`, not by the LLM. The rate is computed by checking whether returned products pass the same filter function. It's deterministic, not estimated.

### System Design
13. **Q**: How would you deploy this to production?
    **A**: (1) Replace FakeEmbeddingBackend with SentenceTransformersBackend, (2) index 2000+ real products, (3) add PostgreSQL/pgvector for persistent vector storage, (4) add authentication, (5) add monitoring (Prometheus/Grafana), (6) add caching for frequent queries.

14. **Q**: How do you measure if the reranker is worth the cost?
    **A**: Compare Hybrid vs Hybrid+Rerank nDCG@10 and P95 latency. If nDCG improves significantly and P95 stays acceptable, the reranker is worth it. The fixture benchmark shows both at 1.0 because the keyword reranker matches BM25's strength on keyword-heavy queries.

15. **Q**: What would you change about the evaluation?
    **A**: (1) Add cross-encoder reranker comparison, (2) run on real 2000-product data, (3) get human-reviewed relevance judgments, (4) add diversity metrics (not just relevance — are all top results from the same brand?), (5) add A/B test framework for production comparison.

## 13. 7-Day Learning Plan (2-3 hours/day)

### Day 1: Understand the data pipeline
- Read `scripts/prepare_electronics_data.py`
- Try the dry run: `uv run python scripts/prepare_electronics_data.py --dry-run`
- Understand: streaming, cleaning, deduplication, manifest generation

### Day 2: Real embeddings
- Read `src/marketlens/retrieval/embedding.py` (SentenceTransformersBackend)
- If you have internet: `uv pip install sentence-transformers` and try encoding a few sentences
- Compare: fake embedding vs real embedding quality

### Day 3: BM25 deep dive
- Read `src/marketlens/retrieval/bm25.py` line by line
- Understand: TF, IDF, k1, b parameters, the scoring formula
- Run: `uv run pytest tests/test_bm25.py -v`

### Day 4: RRF and Hybrid
- Read `src/marketlens/retrieval/hybrid.py` — focus on `_reciprocal_rank_fusion`
- Understand how RRF combines BM25 and embedding rankings
- Try different k values: 0, 30, 60, 120

### Day 5: Evaluation framework
- Read `src/marketlens/evaluation/retrieval_comparison.py`
- Run the fixture benchmark with real output
- Understand: Recall@10, nDCG@10, timing percentiles

### Day 6: Run the benchmark
- Run the full 4-strategy comparison
- Read the generated JSON results and markdown report
- Try modifying queries or adding your own

### Day 7: Documentation and review
- Read `docs/PHASE6_LEARNING_GUIDE.md` (this file)
- Answer the 15 interview questions without looking at the answers
- Review all 8 core files

## 14. 5 Practice Exercises

1. **Add a new query**: Edit `build_eval_queries()` to add a new category ("price_range" for queries with both min and max budget). Run the benchmark and see how scores change.

2. **Compare real vs fake embeddings**: Install sentence-transformers, run the benchmark with `use_real_embeddings=True`, and compare the results table to the fixture table in `docs/PHASE6_EVALUATION_REPORT.md`.

3. **Tune RRF parameters**: Modify `HybridRetriever` to try k=0, k=30, k=120. Which `k` gives the best nDCG@10 on the fixture?

4. **Implement a new reranker**: Create a `LengthReranker` that scores longer product descriptions higher (simulating "more detailed = better"). Add it to the comparison.

5. **Write a data quality report**: Run `prepare_electronics_data.py` with real data. Analyze the manifest's skip_stats. What percentage of raw items are valid? Where is most data lost?
