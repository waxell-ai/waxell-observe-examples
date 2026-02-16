#!/usr/bin/env python3
"""Streaming comparison demo.

Demonstrates waxell-observe's streaming token capture with both OpenAI
and Anthropic. Shows chunk-by-chunk accumulation and final token counts.

Usage::

    # Dry-run (no API keys needed)
    python examples/05_streaming_agent/streaming_agent.py --dry-run

    # With real API calls
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python examples/05_streaming_agent/streaming_agent.py
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
    get_observe_client,
    get_streaming_openai_client,
    get_streaming_anthropic_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Explain quantum computing in simple terms"


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    openai_client = get_streaming_openai_client(dry_run=args.dry_run)
    anthropic_client = get_streaming_anthropic_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Streaming Demo] Starting streaming comparison...")
    print(f"[Streaming Demo] Session: {session}")
    print(f"[Streaming Demo] End user: {user_id} ({user_group})")
    print(f"[Streaming Demo] Query: {query}")
    print()

    async with WaxellContext(
        agent_name="streaming-demo",
        workflow_name="streaming-comparison",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "streaming")
        ctx.set_tag("providers", "openai,anthropic")

        try:
            # Step 1: OpenAI streaming
            print("[Streaming Demo] Step 1/3: OpenAI streaming call...")
            print("  OpenAI response: ", end="", flush=True)

            openai_stream = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": query},
                ],
                stream=True,
            )

            openai_content = ""
            openai_tokens_in = 0
            openai_tokens_out = 0
            chunk_count = 0

            async for chunk in openai_stream:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    openai_content += text
                    print(text, end="", flush=True)
                if chunk.usage:
                    openai_tokens_in = getattr(chunk.usage, "prompt_tokens", 150)
                    openai_tokens_out = getattr(chunk.usage, "completion_tokens", 80)

            if not openai_tokens_out:
                openai_tokens_in = 150
                openai_tokens_out = max(len(openai_content.split()) * 2, 80)

            print()
            print(f"  ({chunk_count} chunks, {openai_tokens_in}+{openai_tokens_out} tokens)")
            print()

            ctx.record_step("openai_stream", output={
                "chunks": chunk_count,
                "content_length": len(openai_content),
                "tokens_in": openai_tokens_in,
                "tokens_out": openai_tokens_out,
            })
            ctx.record_llm_call(
                model="gpt-4o-mini",
                tokens_in=openai_tokens_in,
                tokens_out=openai_tokens_out,
                task="openai_stream",
                prompt_preview=query[:200],
                response_preview=openai_content[:200],
            )

            # Step 2: Anthropic streaming
            print("[Streaming Demo] Step 2/3: Anthropic streaming call...")
            print("  Anthropic response: ", end="", flush=True)

            anthropic_stream = await anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": query},
                ],
                stream=True,
            )

            anthropic_content = ""
            anthropic_tokens_in = 0
            anthropic_tokens_out = 0
            event_count = 0

            async for event in anthropic_stream:
                event_count += 1
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    text = event.delta.text
                    anthropic_content += text
                    print(text, end="", flush=True)
                elif event.type == "message_start" and hasattr(event, "message"):
                    if hasattr(event.message, "usage"):
                        anthropic_tokens_in = event.message.usage.input_tokens
                elif event.type == "message_delta" and hasattr(event, "usage"):
                    if event.usage:
                        anthropic_tokens_out = event.usage.output_tokens

            if not anthropic_tokens_out:
                anthropic_tokens_in = 150
                anthropic_tokens_out = max(len(anthropic_content.split()) * 2, 80)

            print()
            print(f"  ({event_count} events, {anthropic_tokens_in}+{anthropic_tokens_out} tokens)")
            print()

            ctx.record_step("anthropic_stream", output={
                "events": event_count,
                "content_length": len(anthropic_content),
                "tokens_in": anthropic_tokens_in,
                "tokens_out": anthropic_tokens_out,
            })
            ctx.record_llm_call(
                model="claude-sonnet-4-5-20250929",
                tokens_in=anthropic_tokens_in,
                tokens_out=anthropic_tokens_out,
                task="anthropic_stream",
                prompt_preview=query[:200],
                response_preview=anthropic_content[:200],
            )

            # Step 3: Comparison
            print("[Streaming Demo] Step 3/3: Comparing results...")

            comparison = {
                "openai": {
                    "content_length": len(openai_content),
                    "tokens": openai_tokens_in + openai_tokens_out,
                    "chunks": chunk_count,
                },
                "anthropic": {
                    "content_length": len(anthropic_content),
                    "tokens": anthropic_tokens_in + anthropic_tokens_out,
                    "events": event_count,
                },
            }

            ctx.record_step("compare_results", output=comparison)

            ctx.set_result({
                "openai_response": openai_content[:500],
                "anthropic_response": anthropic_content[:500],
                "comparison": comparison,
            })

            print(f"  OpenAI:    {len(openai_content)} chars, {openai_tokens_in + openai_tokens_out} tokens")
            print(f"  Anthropic: {len(anthropic_content)} chars, {anthropic_tokens_in + anthropic_tokens_out} tokens")
            print()
            print("[Streaming Demo] Complete. (2 streaming LLM calls, 3 steps)")

        except PolicyViolationError as e:
            print(f"\n[Streaming Demo] POLICY VIOLATION: {e}")
            print("[Streaming Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
