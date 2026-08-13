"""Single-agent orchestrator with step budget and evidence verification.

Core loop:
  LLM decides → tool dispatch → observe → LLM decides → ... → verify → respond

Key constraints:
  - Max steps: 6 (each step = one LLM call + optional tool dispatch)
  - Max tool calls: 8 total
  - Evidence verification runs before response
  - Retry once on verification failure, then degrade
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from marketlens.agent.evidence import EvidenceVerifier
from marketlens.agent.models import (
    AgentRequest,
    AgentResponse,
    EvidenceRef,
    RecommendationItem,
)
from marketlens.agent.prompts import AGENT_SYSTEM_PROMPT
from marketlens.agent.providers.base import LLMClient
from marketlens.agent.tools import TOOL_DEFINITIONS, AgentTools

logger = logging.getLogger(__name__)

MAX_STEPS = 6
MAX_TOOL_CALLS = 8
RETRY_VERIFICATION_ONCE = True


class AgentOrchestrator:
    """Single-agent orchestrator with tool-calling loop."""

    def __init__(
        self,
        llm: LLMClient,
        tools: AgentTools,
        product_index: dict[str, dict[str, Any]],
        *,
        max_steps: int = MAX_STEPS,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            llm: LLM client (real or fake).
            tools: AgentTools instance.
            product_index: Full product catalog index for evidence verification.
            max_steps: Maximum LLM-call steps.
            max_tool_calls: Maximum total tool invocations.
        """
        self._llm = llm
        self._tools = tools
        self._verifier = EvidenceVerifier(product_index)
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        # Observability: records executed tool calls (not decision logic).
        # Each entry: step_number, tool_name, arguments, result_product_ids,
        # success, error_type. Reset at the start of each run().
        self.tool_call_log: list[dict[str, Any]] = []

    def run(self, request: AgentRequest) -> AgentResponse:
        """Run the agent on a user request.

        Args:
            request: Validated AgentRequest.

        Returns:
            AgentResponse with status, recommendations, evidence.
        """
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()
        warnings: list[str] = []
        tool_call_count = 0
        mode_used = MODE_MAP.get(request.mode, "hybrid")
        self.tool_call_log = []  # Reset per run

        # --- Step 1: Build initial messages ---
        system_msg = AGENT_SYSTEM_PROMPT.format(mode=request.mode)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": request.message},
        ]

        # --- Agent loop ---
        tool_results_for_evidence: list[dict[str, Any]] = []
        final_text = ""
        degraded = False

        for step in range(self._max_steps):
            try:
                response = self._llm.send(messages, TOOL_DEFINITIONS, timeout_s=30.0)
            except Exception as e:
                logger.error("LLM error at step %d: %s", step, e)
                return self._degraded_fallback(request, request_id, mode_used, t0, warnings, str(e))

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # Append assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # No more tool calls → LLM is done
            if not tool_calls:
                final_text = content
                break

            # Check budget
            if tool_call_count + len(tool_calls) > self._max_tool_calls:
                warnings.append(f"Tool call budget exceeded at step {step}")
                final_text = content
                break

            # Dispatch tools
            for tc in tool_calls:
                tname = tc.get("function", {}).get("name", "")
                try:
                    targs = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(targs, str):
                        import json
                        targs = json.loads(targs)
                except Exception:
                    targs = {}

                try:
                    result = self._tools.dispatch(tname, targs)
                    tool_call_count += 1
                    # Record for evidence verification
                    tool_results_for_evidence.append({
                        "tool_name": tname,
                        "arguments": targs,
                        "result": result,
                    })
                    # Record observability (sanitized args only)
                    self.tool_call_log.append({
                        "step_number": step + 1,
                        "tool_name": tname,
                        "arguments": _sanitize_args(targs),
                        "result_product_ids": _extract_product_ids(result),
                        "success": True,
                        "error_type": None,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step}"),
                        "name": tname,
                        "content": _serialize_result(result),
                    })
                except ValueError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step}"),
                        "name": tname,
                        "content": f"Error: {e}",
                    })
                    warnings.append(f"Tool error ({tname}): {e}")
                    tool_call_count += 1
                    self.tool_call_log.append({
                        "step_number": step + 1,
                        "tool_name": tname,
                        "arguments": _sanitize_args(targs),
                        "result_product_ids": [],
                        "success": False,
                        "error_type": "ValueError",
                    })

        # --- Evidence verification ---
        recommendations = _build_recommendations(tool_results_for_evidence, final_text)

        # Collect all evidence refs
        all_evidence: list[EvidenceRef] = []
        for rec in recommendations:
            all_evidence.extend(rec.evidence)

        # Verify
        valid, issues = self._verifier.verify_response(recommendations)

        if not valid:
            if RETRY_VERIFICATION_ONCE:
                logger.warning("Evidence verification failed, retrying once: %s", issues[:3])
                # Try to fix: remove invalid recommendations
                recommendations = [r for r in recommendations if self._verifier.verify_recommendation(r) == []]
                valid2, issues2 = self._verifier.verify_response(recommendations)
                if not valid2:
                    logger.error("Evidence still invalid after retry: %s", issues2[:3])
                    degraded = True
                else:
                    logger.info("Evidence fixed after retry")
            else:
                degraded = True

        # --- Status determination ---
        if not recommendations and not final_text:
            status = "no_results"
        elif degraded:
            status = "degraded"
        else:
            status = "completed"

        elapsed_ms = (time.monotonic() - t0) * 1000

        return AgentResponse(
            request_id=request_id,
            status=status,
            answer=final_text or "No recommendations found.",
            recommendations=recommendations,
            comparison=None,
            constraints={},
            evidence=all_evidence,
            mode_requested=request.mode,
            mode_used=mode_used,
            degraded=degraded,
            warnings=warnings,
            tool_calls=tool_call_count,
            latency_ms=round(elapsed_ms, 2),
        )

    def _degraded_fallback(
        self,
        request: AgentRequest,
        request_id: str,
        mode_used: str,
        t0: float,
        warnings: list[str],
        error: str,
    ) -> AgentResponse:
        """Return degraded fallback when LLM is unavailable."""
        from marketlens.agent.models import SearchCatalogParams

        try:
            result = self._tools.search_catalog(
                SearchCatalogParams(query=request.message, mode="balanced", top_k=request.max_results)
            )
            recs = [
                RecommendationItem(
                    product_id=r.product_id,
                    title=r.title,
                    brand=r.brand if r.brand else None,
                    price=r.price,
                    rating=r.rating,
                    review_count=r.review_count,
                    reason=f"Catalog match (score={r.score:.3f})",
                    constraint_checks={},
                )
                for r in result.results[:request.max_results]
            ]
        except Exception as e2:
            logger.error("Fallback also failed: %s", e2)
            recs = []

        return AgentResponse(
            request_id=request_id,
            status="degraded",
            answer=f"LLM unavailable ({error}). Showing catalog results without AI analysis.",
            recommendations=recs,
            mode_requested=request.mode,
            mode_used=mode_used,
            degraded=True,
            warnings=warnings + [f"LLM error: {error}"],
            tool_calls=1,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            error=error,
        )


MODE_MAP: dict[str, str] = {"fast": "bm25", "balanced": "hybrid", "quality": "rerank"}


def _build_recommendations(
    tool_results: list[dict[str, Any]],
    _final_text: str,
) -> list[RecommendationItem]:
    """Extract recommendations from tool results.

    Scans search_catalog and get_product_details results for product IDs
    mentioned and builds recommendation items with evidence.
    """
    seen_ids: set[str] = set()
    recs: list[RecommendationItem] = []

    for tr in tool_results:
        tname = tr["tool_name"]
        result = tr["result"]

        if tname == "search_catalog":
            for item in result.results:
                pid = item.product_id
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                recs.append(RecommendationItem(
                    product_id=pid,
                    title=item.title,
                    brand=item.brand if item.brand else None,
                    price=item.price,
                    rating=item.rating,
                    review_count=item.review_count,
                    reason=f"Search match (score={item.score:.3f})",
                    evidence=[EvidenceRef(product_id=pid, field="title", observed_value=item.title)],
                    constraint_checks={},
                ))

        elif tname == "get_product_details":
            for prod in result.products:
                pid = prod.product_id
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    recs.append(RecommendationItem(
                        product_id=pid,
                        title=prod.title,
                        brand=prod.brand,
                        price=prod.price,
                        rating=prod.rating,
                        review_count=prod.review_count,
                        reason="Inspected via product details",
                        evidence=[EvidenceRef(product_id=pid, field="title", observed_value=prod.title)],
                        constraint_checks={},
                    ))

    return recs


def _serialize_result(result: Any) -> str:
    """Serialize a tool result for the LLM context window."""
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2)
    import json
    try:
        return json.dumps(result, default=str, indent=2)
    except Exception:
        return str(result)


def _sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow sanitized copy of tool args (no keys/secrets)."""
    if not isinstance(args, dict):
        return {}
    return {str(k): v for k, v in args.items()}


def _extract_product_ids(result: Any) -> list[str]:
    """Extract product IDs from a tool result for audit logging."""
    ids: list[str] = []
    # search_catalog → SearchCatalogResult.results
    if hasattr(result, "results"):
        for item in result.results:
            if hasattr(item, "product_id"):
                ids.append(str(item.product_id))
    # get_product_details / compare_products → .products
    if hasattr(result, "products"):
        for p in result.products:
            if hasattr(p, "product_id"):
                ids.append(str(p.product_id))
    return ids
