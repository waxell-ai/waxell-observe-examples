#!/usr/bin/env python3
"""SDK enrichment showcase demo.

Demonstrates all waxell-observe enrichment features: scores, tags, metadata,
the @observe decorator, and top-level convenience functions.

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
import os

from _common import setup_observe

setup_observe()

import waxell_observe
from waxell_observe import observe, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "How can organizations implement responsible AI practices?"

_client = None


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    global _client
    _client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Enrichment Demo] Starting SDK enrichment showcase...")
    print(f"[Enrichment Demo] Session: {session}")
    print(f"[Enrichment Demo] End user: {user_id} ({user_group})")
    print(f"[Enrichment Demo] Query: {query}")
    print(f"[Enrichment Demo] Using @observe decorator (Level 2 pattern)")
    print()

    # Define decorated function inside main() so it captures dynamic identity
    @observe(
        agent_name="enrichment-demo",
        workflow_name="enrichment-showcase",
        session_id=session,
        user_id=user_id,
        user_group=user_group,
    )
    async def run_enrichment_pipeline(query: str, waxell_ctx=None) -> dict:
        """Run the enrichment demo pipeline with all SDK features."""

        # --- Tags: categorize this run ---
        print("[Enrichment Demo] Recording tags...")
        waxell_observe.tag("intent", "question")
        waxell_observe.tag("domain", "technology")
        waxell_observe.tag("priority", "high")
        waxell_observe.tag("demo", "enrichment")
        print("  Tags: intent=question, domain=technology, priority=high, demo=enrichment")

        # --- Metadata: add context ---
        print("[Enrichment Demo] Recording metadata...")
        waxell_observe.metadata("user_tier", "premium")
        waxell_observe.metadata("request_source", "api")
        waxell_observe.metadata("model_config", {"temperature": 0.7, "max_tokens": 500})
        waxell_observe.metadata("pipeline_version", "2.1.0")
        print("  Metadata: user_tier=premium, request_source=api, pipeline_version=2.1.0")
        print()

        # --- Step 1: Analyze input ---
        print("[Enrichment Demo] Step 1/4: Analyzing input...")

        if waxell_ctx:
            waxell_ctx.record_step("analyze_input", output={"query_length": len(query)})

        analysis_response = await _client.chat.completions.create(
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

        if waxell_ctx:
            waxell_ctx.record_llm_call(
                model=analysis_response.model,
                tokens_in=analysis_response.usage.prompt_tokens,
                tokens_out=analysis_response.usage.completion_tokens,
                task="analyze_input",
                prompt_preview=query[:200],
                response_preview=analysis[:200],
            )

        print(f"  Analysis: {analysis[:120]}...")
        print()

        # --- Step 2: Quality assessment ---
        print("[Enrichment Demo] Step 2/4: Assessing quality...")

        assessment_response = await _client.chat.completions.create(
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

        if waxell_ctx:
            waxell_ctx.record_step("assess_quality", output={"assessment": assessment[:200]})
            waxell_ctx.record_llm_call(
                model=assessment_response.model,
                tokens_in=assessment_response.usage.prompt_tokens,
                tokens_out=assessment_response.usage.completion_tokens,
                task="assess_quality",
                prompt_preview=f"Rate this analysis:\n\n{analysis[:100]}...",
                response_preview=assessment[:200],
            )

        print(f"  Assessment: {assessment[:120]}...")
        print()

        # --- Scores: record quality metrics ---
        print("[Enrichment Demo] Step 3/4: Recording scores...")
        waxell_observe.score("quality", 0.92)
        waxell_observe.score("relevance", 0.85)
        waxell_observe.score("helpfulness", 0.88)
        waxell_observe.score("safety", True, data_type="boolean")
        waxell_observe.score("category", "informational", data_type="categorical")
        waxell_observe.score("confidence", 0.78, comment="Based on source availability")

        if waxell_ctx:
            waxell_ctx.record_step("record_scores", output={
                "quality": 0.92,
                "relevance": 0.85,
                "helpfulness": 0.88,
                "safety": True,
                "category": "informational",
                "confidence": 0.78,
            })

        print("  Scores recorded:")
        print("    quality=0.92, relevance=0.85, helpfulness=0.88")
        print("    safety=True (boolean), category=informational (categorical)")
        print("    confidence=0.78 (with comment)")
        print()

        # --- Step 4: Final enrichment ---
        print("[Enrichment Demo] Step 4/4: Final enrichment...")
        waxell_observe.metadata("analysis_tokens", analysis_response.usage.total_tokens)
        waxell_observe.metadata("assessment_tokens", assessment_response.usage.total_tokens)
        waxell_observe.tag("status", "complete")

        if waxell_ctx:
            waxell_ctx.record_step("final_enrichment", output={"status": "complete"})

        print("  Added final metadata and tags")
        print()

        result = {
            "analysis": analysis,
            "assessment": assessment,
            "scores": {
                "quality": 0.92,
                "relevance": 0.85,
                "helpfulness": 0.88,
            },
        }

        print(f"[Enrichment Demo] Complete. (2 LLM calls, 4 steps, 6 scores, 4 tags, 6 metadata)")

        return result

    try:
        result = await run_enrichment_pipeline(query)
    except PolicyViolationError as e:
        print(f"\n[Enrichment Demo] POLICY VIOLATION: {e}")
        print("[Enrichment Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
