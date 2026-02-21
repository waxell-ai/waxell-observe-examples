#!/usr/bin/env python3
"""SDK enrichment showcase demo with modern decorator patterns.

Demonstrates all waxell-observe enrichment features using the decorator-first
SDK: @observe for agent scoping, @step for pipeline stages, scores, tags,
and metadata via top-level convenience functions.

Multi-agent architecture:
  enrichment-orchestrator (parent)
  ├── enrichment-runner (child) — analysis + assessment LLM calls
  └── enrichment-evaluator (child) — quality scoring + final enrichment

Usage::

    # Dry-run (no API key needed)
    python examples/10_enrichment_agent/enrichment_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/10_enrichment_agent/enrichment_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from _common import setup_observe

setup_observe()

import waxell_observe as waxell
from waxell_observe import generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "How can organizations implement responsible AI practices?"


# ---------------------------------------------------------------------------
# @step decorators — auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="analyze_input")
async def analyze_input(query: str) -> dict:
    """Analyze the input query and return basic metrics."""
    return {"query_length": len(query), "word_count": len(query.split())}


@waxell.step_dec(name="record_scores")
async def record_scores_step() -> dict:
    """Record all quality scores."""
    return {
        "quality": 0.92,
        "relevance": 0.85,
        "helpfulness": 0.88,
        "safety": True,
        "category": "informational",
        "confidence": 0.78,
    }


@waxell.step_dec(name="final_enrichment")
async def final_enrichment_step(analysis_tokens: int, assessment_tokens: int) -> dict:
    """Perform final enrichment with token counts and status."""
    return {
        "analysis_tokens": analysis_tokens,
        "assessment_tokens": assessment_tokens,
        "status": "complete",
    }


# ---------------------------------------------------------------------------
# Child Agent 1: enrichment-runner — handles LLM calls for analysis
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="enrichment-runner", workflow_name="enrichment-analysis")
async def run_enrichment_analysis(query: str, openai_client, waxell_ctx=None) -> dict:
    """Run the analysis and assessment LLM calls."""
    waxell.tag("agent_role", "runner")
    waxell.tag("provider", "openai")

    # Step 1: Analyze input
    print("[Enrichment Runner] Analyzing input...")
    await analyze_input(query)

    analysis_response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI advisor. Analyze the user's question and provide actionable insights.",
            },
            {"role": "user", "content": query},
        ],
    )
    analysis = analysis_response.choices[0].message.content
    print(f"  Analysis: {analysis[:120]}...")

    # Step 2: Quality assessment
    print("[Enrichment Runner] Assessing quality...")
    assessment_response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Rate the quality of this analysis on a scale of 1-10. Be specific about strengths and weaknesses.",
            },
            {"role": "user", "content": f"Rate this analysis:\n\n{analysis}"},
        ],
    )
    assessment = assessment_response.choices[0].message.content
    print(f"  Assessment: {assessment[:120]}...")

    waxell.metadata("analysis_tokens", analysis_response.usage.total_tokens)
    waxell.metadata("assessment_tokens", assessment_response.usage.total_tokens)

    return {
        "analysis": analysis,
        "assessment": assessment,
        "analysis_tokens": analysis_response.usage.total_tokens,
        "assessment_tokens": assessment_response.usage.total_tokens,
    }


# ---------------------------------------------------------------------------
# Child Agent 2: enrichment-evaluator — scoring + final enrichment
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="enrichment-evaluator", workflow_name="enrichment-scoring")
async def run_enrichment_evaluation(analysis_tokens: int, assessment_tokens: int, waxell_ctx=None) -> dict:
    """Score the enrichment results and apply final tags/metadata."""
    waxell.tag("agent_role", "evaluator")

    # Step 3: Record scores
    print("[Enrichment Evaluator] Recording scores...")
    waxell.score("quality", 0.92)
    waxell.score("relevance", 0.85)
    waxell.score("helpfulness", 0.88)
    waxell.score("safety", True, data_type="boolean")
    waxell.score("category", "informational", data_type="categorical")
    waxell.score("confidence", 0.78, comment="Based on source availability")

    await record_scores_step()
    print("  Scores recorded:")
    print("    quality=0.92, relevance=0.85, helpfulness=0.88")
    print("    safety=True (boolean), category=informational (categorical)")
    print("    confidence=0.78 (with comment)")

    # Step 4: Final enrichment
    print("[Enrichment Evaluator] Final enrichment...")
    waxell.metadata("analysis_tokens", analysis_tokens)
    waxell.metadata("assessment_tokens", assessment_tokens)
    waxell.tag("status", "complete")

    await final_enrichment_step(analysis_tokens, assessment_tokens)
    print("  Added final metadata and tags")

    return {
        "scores": {
            "quality": 0.92,
            "relevance": 0.85,
            "helpfulness": 0.88,
        },
        "status": "complete",
    }


# ---------------------------------------------------------------------------
# Parent Orchestrator: enrichment-orchestrator
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="enrichment-orchestrator", workflow_name="enrichment-showcase")
async def run_enrichment_pipeline(query: str, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Orchestrate the enrichment demo pipeline across child agents."""
    waxell.tag("demo", "enrichment")
    waxell.tag("pipeline", "enrichment-showcase")
    waxell.metadata("user_tier", "premium")
    waxell.metadata("request_source", "api")
    waxell.metadata("model_config", {"temperature": 0.7, "max_tokens": 500})
    waxell.metadata("pipeline_version", "2.1.0")
    waxell.metadata("mode", "dry-run" if dry_run else "live")

    openai_client = get_openai_client(dry_run=dry_run)

    # Tags: categorize this run
    print("[Enrichment Orchestrator] Recording tags...")
    waxell.tag("intent", "question")
    waxell.tag("domain", "technology")
    waxell.tag("priority", "high")
    print("  Tags: intent=question, domain=technology, priority=high, demo=enrichment")

    # Metadata: add context
    print("[Enrichment Orchestrator] Recording metadata...")
    print("  Metadata: user_tier=premium, request_source=api, pipeline_version=2.1.0")
    print()

    # Phase 1: enrichment-runner child agent
    print("[Enrichment Orchestrator] Phase 1: Running enrichment analysis (enrichment-runner)...")
    analysis_result = await run_enrichment_analysis(
        query=query,
        openai_client=openai_client,
    )
    print()

    # Phase 2: enrichment-evaluator child agent
    print("[Enrichment Orchestrator] Phase 2: Running enrichment evaluation (enrichment-evaluator)...")
    eval_result = await run_enrichment_evaluation(
        analysis_tokens=analysis_result["analysis_tokens"],
        assessment_tokens=analysis_result["assessment_tokens"],
    )
    print()

    result = {
        "analysis": analysis_result["analysis"],
        "assessment": analysis_result["assessment"],
        "scores": eval_result["scores"],
        "pipeline": "enrichment-orchestrator -> enrichment-runner -> enrichment-evaluator",
    }

    print(f"[Enrichment Orchestrator] Complete. (2 LLM calls, 4 steps, 6 scores, 4 tags, 6 metadata)")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY
    session = generate_session_id()
    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]
    observe_active = not is_observe_disabled()

    print("[Enrichment Demo] Starting SDK enrichment showcase...")
    print(f"[Enrichment Demo] Session: {session}")
    print(f"[Enrichment Demo] End user: {user_id} ({user_group})")
    print(f"[Enrichment Demo] Query: {query}")
    print(f"[Enrichment Demo] Using @observe decorator (Level 2 pattern)")
    print()

    try:
        await run_enrichment_pipeline(
            query=query,
            dry_run=args.dry_run,
            session_id=session,
            user_id=user_id,
            user_group=user_group,
            enforce_policy=observe_active,
            mid_execution_governance=observe_active,
            client=get_observe_client(),
        )
    except PolicyViolationError as e:
        print(f"\n[Enrichment Demo] POLICY VIOLATION: {e}")
        print("[Enrichment Demo] Agent halted by governance policy.")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
