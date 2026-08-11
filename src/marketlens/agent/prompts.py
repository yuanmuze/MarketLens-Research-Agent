"""Prompt templates for the MarketLens product research agent."""

PARSE_REQUEST_PROMPT = """You are a product research assistant. Analyze the user's query and extract structured search parameters.

User query: {query}

Extract:
1. The core search intent (what product are they looking for?)
2. Any budget constraints (max/min price)
3. Preferred or excluded brands
4. Required features (noise cancellation, battery life, etc.)
5. Minimum rating or review count

Return a structured analysis of the request."""

RETRIEVE_PROMPT = """You have access to a product catalog search tool.
Use it to find products matching the user's query.

User query: {query}
Search constraints: {constraints}

Call the search_catalog tool with appropriate search queries.
Search multiple variations if needed (e.g., "wireless headphones", "noise cancelling headphones")."""

ASSESS_EVIDENCE_PROMPT = """Review the search results and assess the quality of evidence for each product.

For each product, evaluate:
1. How well does it match the user's needs?
2. Is the evidence (reviews, ratings, description) sufficient?
3. Are there any gaps in the product data?

Search results:
{search_results}

User constraints:
{constraints}"""

COMPARE_PROMPT = """Compare the products that matched the user's query.

For each product, identify:
1. Key advantages (pros)
2. Disadvantages (cons)
3. How it compares to alternatives
4. Recommendation score (1-10)

Products to compare:
{products}

User query: {query}"""

VALIDATE_PROMPT = """Validate that all recommended products satisfy the user's hard constraints.

User constraints: {constraints}
Products: {products}

Check:
1. Budget compliance
2. Brand preferences
3. Rating minimums
4. Category matching

Note any violations."""

GENERATE_REPORT_PROMPT = """Generate a comprehensive product research report.

Query: {query}
Comparisons: {comparisons}
Validation: {validation}

Write a well-structured report with:
1. Executive summary
2. Product comparison table
3. Detailed analysis of top picks
4. Recommendations
5. Evidence sources

Format in Markdown."""
