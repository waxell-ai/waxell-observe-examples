#!/usr/bin/env python3
"""Anthropic content analysis demo.

Demonstrates waxell-observe's Anthropic auto-instrumentor with a 3-step
content analysis pipeline: classify, extract entities, summarize.

Usage::

    # Dry-run (no Anthropic API key needed)
    python examples/01_anthropic_agent/anthropic_agent.py --dry-run

    # With real Anthropic calls
    ANTHROPIC_API_KEY=sk-ant-... python examples/01_anthropic_agent/anthropic_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os

from _common import setup_observe

setup_observe()

import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    get_anthropic_client,
    get_observe_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Analyze the impact of artificial intelligence on modern healthcare systems"


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    client = get_anthropic_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Anthropic Demo] Starting content analysis pipeline...")
    print(f"[Anthropic Demo] Session: {session}")
    print(f"[Anthropic Demo] End user: {user_id} ({user_group})")
    print(f"[Anthropic Demo] Input: {query[:80]}...")
    print()

    async with WaxellContext(
        agent_name="anthropic-demo",
        workflow_name="content-analysis",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "anthropic")
        ctx.set_tag("provider", "anthropic")
        ctx.set_metadata("sdk", "anthropic-python")

        try:
            # Step 1/3: Classify content
            print("[Anthropic Demo] Step 1/3: Classifying content...")

            classify_response = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Classify the following text into one of these categories: "
                            f"technology, healthcare, finance, education, science, politics. "
                            f"Also rate the complexity from 1-5.\n\nText: {query}"
                        ),
                    },
                ],
            )
            classification = classify_response.content[0].text

            ctx.record_step("classify_content", output={"classification": classification[:200]})
            ctx.record_llm_call(
                model=classify_response.model,
                tokens_in=classify_response.usage.input_tokens,
                tokens_out=classify_response.usage.output_tokens,
                task="classify_content",
                prompt_preview=query[:200],
                response_preview=classification[:200],
            )
            print(f"           Classification: {classification[:120]}...")

            # Step 2/3: Extract entities
            print("[Anthropic Demo] Step 2/3: Extracting key entities...")

            extract_response = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Extract the key entities (people, organizations, technologies, "
                            f"concepts) from this text. List each entity with its type.\n\n"
                            f"Text: {query}"
                        ),
                    },
                ],
            )
            entities = extract_response.content[0].text

            ctx.record_step("extract_entities", output={"entities": entities[:200]})
            ctx.record_llm_call(
                model=extract_response.model,
                tokens_in=extract_response.usage.input_tokens,
                tokens_out=extract_response.usage.output_tokens,
                task="extract_entities",
                prompt_preview=query[:200],
                response_preview=entities[:200],
            )
            print(f"           Entities: {entities[:120]}...")

            # Step 3/3: Summarize
            print("[Anthropic Demo] Step 3/3: Generating summary...")

            summary_response = await client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Provide a concise 2-3 sentence summary of the following text, "
                            f"highlighting the main argument and key implications.\n\n"
                            f"Text: {query}\n\n"
                            f"Classification: {classification}\n"
                            f"Key entities: {entities}"
                        ),
                    },
                ],
            )
            summary = summary_response.content[0].text

            ctx.record_step("summarize", output={"summary_length": len(summary)})
            ctx.record_llm_call(
                model=summary_response.model,
                tokens_in=summary_response.usage.input_tokens,
                tokens_out=summary_response.usage.output_tokens,
                task="summarize",
                prompt_preview=query[:200],
                response_preview=summary[:200],
            )

            ctx.set_result({
                "classification": classification,
                "entities": entities,
                "summary": summary,
            })

            print()
            print(f"[Anthropic Demo] Summary: {summary[:200]}...")
            print(f"[Anthropic Demo] Complete. (3 LLM calls, 3 steps, provider: Anthropic)")

        except PolicyViolationError as e:
            print(f"\n[Anthropic Demo] POLICY VIOLATION: {e}")
            print("[Anthropic Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
