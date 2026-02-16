#!/usr/bin/env python3
"""All-providers-in-one-trace demo.

Demonstrates multi-provider tracing by calling OpenAI, Anthropic, and
LiteLLM within a single WaxellContext session.

Usage::

    # Dry-run (no API keys needed)
    python examples/12_all_providers_agent/all_providers_agent.py --dry-run

    # With real API calls
    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... python examples/12_all_providers_agent/all_providers_agent.py
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
    MockChatCompletion,
    _contextual_response,
    get_anthropic_client,
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "What are the key trends in AI for 2025?"


async def call_litellm(model: str, messages: list, dry_run: bool):
    """Call LiteLLM or return a mock response."""
    if dry_run:
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model.split("/")[-1])

    import litellm
    return await litellm.acompletion(model=model, messages=messages)


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    openai_client = get_openai_client(dry_run=args.dry_run)
    anthropic_client = get_anthropic_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[All Providers] Starting multi-provider trace demo...")
    print(f"[All Providers] Session: {session}")
    print(f"[All Providers] End user: {user_id} ({user_group})")
    print(f"[All Providers] Query: {query}")
    print(f"[All Providers] Providers: OpenAI, Anthropic, LiteLLM (Groq)")
    print()

    async with WaxellContext(
        agent_name="all-providers",
        workflow_name="provider-showcase",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "all-providers")
        ctx.set_tag("providers", "openai,anthropic,litellm")
        ctx.set_metadata("num_providers", 3)

        try:
            all_responses = []

            # Step 1: OpenAI call
            print("[All Providers] Step 1/4: OpenAI (gpt-4o-mini)...")

            openai_response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a tech industry analyst. Be concise."},
                    {"role": "user", "content": f"From OpenAI's perspective: {query}"},
                ],
            )
            openai_content = openai_response.choices[0].message.content

            ctx.record_step("openai_call", output={
                "provider": "openai",
                "model": openai_response.model,
                "tokens": openai_response.usage.prompt_tokens + openai_response.usage.completion_tokens,
            })
            ctx.record_llm_call(
                model=openai_response.model,
                tokens_in=openai_response.usage.prompt_tokens,
                tokens_out=openai_response.usage.completion_tokens,
                task="openai_call",
                prompt_preview=query[:200],
                response_preview=openai_content[:200],
            )
            all_responses.append(("OpenAI", openai_content))
            print(f"  Response: {openai_content[:100]}...")
            print()

            # Step 2: Anthropic call
            print("[All Providers] Step 2/4: Anthropic (claude-sonnet)...")

            anthropic_response = await anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": f"From Anthropic's perspective: {query}"},
                ],
            )
            anthropic_content = anthropic_response.content[0].text

            ctx.record_step("anthropic_call", output={
                "provider": "anthropic",
                "model": anthropic_response.model,
                "tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            })
            ctx.record_llm_call(
                model=anthropic_response.model,
                tokens_in=anthropic_response.usage.input_tokens,
                tokens_out=anthropic_response.usage.output_tokens,
                task="anthropic_call",
                prompt_preview=query[:200],
                response_preview=anthropic_content[:200],
            )
            all_responses.append(("Anthropic", anthropic_content))
            print(f"  Response: {anthropic_content[:100]}...")
            print()

            # Step 3: LiteLLM call (Groq)
            print("[All Providers] Step 3/4: LiteLLM/Groq (llama-3.3-70b)...")

            litellm_response = await call_litellm(
                model="groq/llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an open-source AI advocate. Be concise."},
                    {"role": "user", "content": f"From the open-source community's perspective: {query}"},
                ],
                dry_run=args.dry_run,
            )
            litellm_content = litellm_response.choices[0].message.content

            ctx.record_step("litellm_call", output={
                "provider": "groq",
                "model": litellm_response.model,
                "tokens": litellm_response.usage.prompt_tokens + litellm_response.usage.completion_tokens,
            })
            ctx.record_llm_call(
                model=litellm_response.model,
                tokens_in=litellm_response.usage.prompt_tokens,
                tokens_out=litellm_response.usage.completion_tokens,
                task="litellm_call",
                prompt_preview=query[:200],
                response_preview=litellm_content[:200],
            )
            all_responses.append(("Groq/LiteLLM", litellm_content))
            print(f"  Response: {litellm_content[:100]}...")
            print()

            # Step 4: Synthesis
            print("[All Providers] Step 4/4: Synthesizing perspectives...")

            synthesis = "Multi-provider analysis:\n"
            for provider, content in all_responses:
                synthesis += f"\n{provider}: {content[:150]}..."

            ctx.record_step("synthesis", output={
                "providers_used": len(all_responses),
                "synthesis_length": len(synthesis),
            })

            ctx.set_result({
                "responses": {p: c[:500] for p, c in all_responses},
                "providers_used": [p for p, _ in all_responses],
            })

            print(f"  Combined {len(all_responses)} provider perspectives")
            print()
            print(f"[All Providers] Complete. (3 LLM calls across 3 providers, 4 steps, 1 session)")

        except PolicyViolationError as e:
            print(f"\n[All Providers] POLICY VIOLATION: {e}")
            print("[All Providers] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
