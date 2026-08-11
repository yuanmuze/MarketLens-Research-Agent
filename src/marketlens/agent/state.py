"""Agent state definitions for the MarketLens product research agent."""

from datetime import datetime
from typing import Annotated, Any, NotRequired

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from marketlens.models import (
    ComparisonItem,
    Product,
    ProductEvidence,
    SearchResult,
    UserConstraints,
)


class AgentState(TypedDict):
    """State for the MarketLens product research agent.

    This state flows through the LangGraph workflow nodes and
    accumulates the results of each processing step.
    """

    # Input
    messages: Annotated[list[Any], add_messages]
    query: str
    request_id: str

    # Parsed from query
    search_query: str
    constraints: NotRequired[UserConstraints]

    # Retrieval results
    search_results: NotRequired[list[SearchResult]]
    products: NotRequired[list[Product]]

    # Evidence assessment
    evidence: NotRequired[list[ProductEvidence]]

    # Optional web research
    web_search_used: NotRequired[bool]
    web_search_results: NotRequired[list[dict[str, Any]]]

    # Product comparison
    comparisons: NotRequired[list[ComparisonItem]]

    # Constraint validation
    constraints_satisfied: NotRequired[bool]
    constraint_violations: NotRequired[list[dict[str, Any]]]

    # Report generation
    final_report: NotRequired[str]

    # Error handling
    error: NotRequired[str]
    status: NotRequired[str]  # running, completed, failed

    # Metrics
    started_at: NotRequired[datetime]
    completed_at: NotRequired[datetime]
    tool_calls: NotRequired[int]
    retries: NotRequired[int]
    node_timings: NotRequired[dict[str, float]]


class AgentInputState(TypedDict):
    """Input state for the agent (external facing)."""

    messages: list[Any]
    query: str
    request_id: str
