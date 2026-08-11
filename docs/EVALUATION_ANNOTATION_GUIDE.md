# Evaluation Annotation Guide

How to manually review and approve MarketLens evaluation queries.

## Background

The evaluation query set (`data/processed/eval_queries.json`) contains 60 queries across 9 categories. All queries are currently:

- `label_source`: `"auto_curated"` or `"synthetic"`
- `review_status`: `"pending"`

They were generated programmatically from the product catalog and generic templates. They have NOT been verified by a human.

## Annotation Process

### Step 1: Load the queries

```bash
cat data/processed/eval_queries.json | python -m json.tool | less
```

Or open in any JSON viewer/editor.

### Step 2: Review each query

For each query, check:

1. **Query text**: Is it a natural, realistic search query? Would a real user type this?
2. **Category**: Is the category correct? (exact_match, synonym, budget, etc.)
3. **Constraints**: Do the hard constraints make sense for the query?
4. **Relevant product IDs**: Are these actually relevant? Add, remove, or adjust as needed.
5. **Relevance grades**: Assign grades (0=irrelevant, 1=somewhat, 2=relevant, 3=perfect) for each product.

### Step 3: Update review_status

After reviewing a query:

- If you're satisfied: set `review_status` to `"approved"`
- If the query needs changes: fix the fields, then set `review_status` to `"approved"`
- If the query is bad and should be removed: set `review_status` to `"rejected"`

### Step 4: Generate the reviewed set

```bash
# After reviewing, save as a new file
cp data/processed/eval_queries.json data/processed/eval_queries_reviewed.json
# Edit the review_status fields in the new file
```

### Step 5: Run evaluation with reviewed queries

```python
# After reviewing, load the reviewed queries for evaluation
import json
from marketlens.evaluation.retrieval_comparison import (
    build_eval_queries, run_full_comparison, generate_markdown_report
)
from marketlens.catalog import ProductCatalog

catalog = ProductCatalog.from_fixture("electronics_sample.json")

# Load reviewed queries
with open("data/processed/eval_queries_reviewed.json") as f:
    query_dicts = json.load(f)
queries = [EvalQuery(**qd) for qd in query_dicts]
approved = [q for q in queries if q.review_status == "approved"]

reports = run_full_comparison(catalog, approved)
print(generate_markdown_report(reports, approved, config={"human_reviewed": True}))
```

## What to Check

### For exact_match queries
- Does the product ID match the title? (Should be the exact product from the catalog)
- If the catalog doesn't have this exact product, set `relevant_product_ids` to `[]` and lower `review_status` to `"rejected"`

### For synonym queries
- Are synonym queries truly paraphrases? (e.g. "noise cancelling" ≈ "ANC")
- Add relevant product IDs that match the semantic intent
- Assign relevance grades based on match quality

### For brand_filter queries
- Does the catalog actually contain products from these brands?
- If not, add relevant products from similar brands

### For budget queries
- Are the budget thresholds reasonable?
- Add products that genuinely fit the budget

### For contradiction/no_result queries
- These should return empty results. Verify that no products in the catalog match.
- If products DO match (the constraint isn't actually contradictory), fix the query or category.

## Final Notes

- **Human_verified** means you personally checked and approved each field
- Keep the reviewed file separate from the auto-generated one
- After review, update the benchmark report to mark it as "human-reviewed"
- The fixture benchmark and auto-curated benchmark can still be useful baselines even after review
