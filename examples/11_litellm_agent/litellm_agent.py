#!/usr/bin/env python3
"""LiteLLM multi-provider demo.

Demonstrates waxell-observe's LiteLLM instrumentor by calling multiple
models through LiteLLM's unified API: OpenAI, Anthropic, and Groq.

Usage::

    # Dry-run (no API keys needed)
    python examples/11_litellm_agent/litellm_agent.py --dry-run

    # With real API calls (needs at least one LLM API key)
    OPENAI_API_KEY=sk-... python examples/11_litellm_agent/litellm_agent.py
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
    get_observe_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
)

DEFAULT_QUERY = "Compare the approaches to AI safety across different organizations"

# Model configurations for the multi-provider comparison
MODELS = [
    {"model": "gpt-4o-mini", "provider": "OpenAI", "tier": "fast"},
    {"model": "anthropic/claude-sonnet-4-5-20250929", "provider": "Anthropic", "tier": "premium"},
    {"model": "groq/llama-3.3-70b-versatile", "provider": "Groq", "tier": "open-source"},
]


async def call_llm(model: str, messages: list, dry_run: bool):
    """Call LiteLLM or return a mock response."""
    if dry_run:
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model.split("/")[-1])

    import litellm
    return await litellm.acompletion(model=model, messages=messages)


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[LiteLLM Demo] Starting multi-provider comparison...")
    print(f"[LiteLLM Demo] Session: {session}")
    print(f"[LiteLLM Demo] End user: {user_id} ({user_group})")
    print(f"[LiteLLM Demo] Query: {query}")
    print(f"[LiteLLM Demo] Models: {', '.join(m['model'] for m in MODELS)}")
    print()

    async with WaxellContext(
        agent_name="litellm-demo",
        workflow_name="multi-provider",
        inputs={"query": query, "models": [m["model"] for m in MODELS]},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "litellm")
        ctx.set_tag("providers", "openai,anthropic,groq")
        ctx.set_metadata("num_models", len(MODELS))

        try:
            results = []

            for i, model_config in enumerate(MODELS, 1):
                model = model_config["model"]
                provider = model_config["provider"]
                tier = model_config["tier"]

                print(f"[LiteLLM Demo] Step {i}/{len(MODELS)}: Calling {provider} ({tier})...")
                print(f"  Model: {model}")

                messages = [
                    {
                        "role": "system",
                        "content": f"You are an AI safety expert. Provide a {tier}-tier analysis.",
                    },
                    {"role": "user", "content": query},
                ]

                response = await call_llm(model, messages, dry_run=args.dry_run)

                content = response.choices[0].message.content
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens

                ctx.record_step(f"call_{provider.lower()}", output={
                    "model": model,
                    "provider": provider,
                    "tier": tier,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "content_length": len(content),
                })
                ctx.record_llm_call(
                    model=response.model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    task=f"call_{provider.lower()}",
                    prompt_preview=query[:200],
                    response_preview=content[:200],
                )

                results.append({
                    "provider": provider,
                    "model": model,
                    "tier": tier,
                    "content": content,
                    "tokens": tokens_in + tokens_out,
                })

                print(f"  Response: {content[:100]}...")
                print(f"  Tokens: {tokens_in} in + {tokens_out} out = {tokens_in + tokens_out}")
                print()

            # Comparison step
            print(f"[LiteLLM Demo] Step {len(MODELS) + 1}/{len(MODELS) + 1}: Comparing results...")

            comparison = {
                r["provider"]: {
                    "model": r["model"],
                    "tokens": r["tokens"],
                    "content_length": len(r["content"]),
                }
                for r in results
            }

            ctx.record_step("compare_providers", output=comparison)

            for r in results:
                print(f"  {r['provider']:10s} ({r['tier']:12s}): {r['tokens']:4d} tokens, {len(r['content']):4d} chars")

            ctx.set_result({
                "comparison": comparison,
                "models_tested": len(MODELS),
            })

            print()
            print(f"[LiteLLM Demo] Complete. ({len(MODELS)} LLM calls via LiteLLM, {len(MODELS) + 1} steps)")

        except PolicyViolationError as e:
            print(f"\n[LiteLLM Demo] POLICY VIOLATION: {e}")
            print("[LiteLLM Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
