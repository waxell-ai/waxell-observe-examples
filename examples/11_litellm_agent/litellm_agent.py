#!/usr/bin/env python3
"""LiteLLM multi-provider demo.

Demonstrates waxell-observe's modern decorator patterns by calling multiple
models through LiteLLM's unified API: OpenAI, Anthropic, and Groq.

Uses:
  - @waxell.observe for parent orchestrator and child provider-caller agents
  - @waxell.tool(tool_type="llm") for the call_llm function
  - @waxell.step_dec for preprocess_query
  - @waxell.decision for choose_comparison_strategy
  - @waxell.reasoning_dec for compare_provider_results
  - waxell.tag(), waxell.score(), waxell.metadata() for enrichment

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
from waxell_observe import (
    observe,
    generate_session_id,
)
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


@waxell_observe.tool(tool_type="llm")
async def call_llm(model: str, messages: list, dry_run: bool):
    """Call LiteLLM or return a mock response."""
    if dry_run:
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model.split("/")[-1])

    import litellm
    return await litellm.acompletion(model=model, messages=messages)


@waxell_observe.step_dec(name="preprocess_query")
async def preprocess_query(query: str) -> dict:
    """Clean and prepare the query for multi-provider comparison."""
    cleaned = query.strip()
    word_count = len(cleaned.split())
    return {
        "original": query,
        "cleaned": cleaned,
        "word_count": word_count,
        "complexity": "high" if word_count > 10 else "standard",
    }


@waxell_observe.decision(
    name="choose_comparison_strategy",
    options=["all_models", "subset"],
)
async def choose_comparison_strategy(query: str, models: list[dict]) -> dict:
    """Decide whether to run all models or a subset based on query complexity."""
    word_count = len(query.split())
    # For short queries, use all models; for long/complex queries, also use all
    # (in production you might select a subset for cost control)
    chosen = "all_models" if len(models) <= 5 else "subset"
    return {
        "chosen": chosen,
        "reasoning": (
            f"Query has {word_count} words with {len(models)} available models. "
            f"Using {'all models' if chosen == 'all_models' else 'top-tier subset'} "
            f"for comprehensive comparison."
        ),
        "confidence": 0.95 if chosen == "all_models" else 0.80,
    }


@waxell_observe.reasoning_dec(step="compare_provider_results")
async def compare_provider_results(results: list[dict]) -> dict:
    """Compare token usage and quality across providers."""
    if not results:
        return {
            "thought": "No results to compare",
            "evidence": [],
            "conclusion": "No comparison possible",
        }

    total_tokens = sum(r["tokens"] for r in results)
    avg_tokens = total_tokens // len(results)
    most_efficient = min(results, key=lambda r: r["tokens"])
    most_verbose = max(results, key=lambda r: len(r["content"]))

    evidence = [
        f"{r['provider']} ({r['tier']}): {r['tokens']} tokens, {len(r['content'])} chars"
        for r in results
    ]

    return {
        "thought": (
            f"Compared {len(results)} providers. Average token usage: {avg_tokens}. "
            f"Most efficient: {most_efficient['provider']} ({most_efficient['tokens']} tokens). "
            f"Most verbose response: {most_verbose['provider']} ({len(most_verbose['content'])} chars)."
        ),
        "evidence": evidence,
        "conclusion": (
            f"{most_efficient['provider']} is most token-efficient while "
            f"{most_verbose['provider']} provides the longest response. "
            f"Total tokens across all providers: {total_tokens}."
        ),
    }


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    observe_client = get_observe_client()

    print("[LiteLLM Demo] Starting multi-provider comparison...")
    print(f"[LiteLLM Demo] Session: {session}")
    print(f"[LiteLLM Demo] End user: {user_id} ({user_group})")
    print(f"[LiteLLM Demo] Query: {query}")
    print(f"[LiteLLM Demo] Models: {', '.join(m['model'] for m in MODELS)}")
    print()

    # ---------------------------------------------------------------
    # Parent orchestrator
    # ---------------------------------------------------------------
    @observe(
        agent_name="litellm-orchestrator",
        workflow_name="multi-provider",
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        client=observe_client,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
    )
    async def litellm_orchestrator(query: str) -> dict:
        """Orchestrate multi-provider LLM comparison."""

        waxell_observe.tag("demo", "litellm")
        waxell_observe.tag("providers", "openai,anthropic,groq")
        waxell_observe.metadata("num_models", len(MODELS))

        # Step: preprocess the query
        print("[LiteLLM Demo] Preprocessing query...")
        preprocessed = await preprocess_query(query)
        print(f"  Preprocessed: {preprocessed['word_count']} words, complexity={preprocessed['complexity']}")
        print()

        # Decision: choose comparison strategy
        print("[LiteLLM Demo] Choosing comparison strategy...")
        strategy = await choose_comparison_strategy(query, MODELS)
        chosen_strategy = strategy["chosen"] if isinstance(strategy, dict) else strategy
        print(f"  Strategy: {chosen_strategy}")
        print()

        # Determine which models to run
        models_to_run = MODELS if chosen_strategy == "all_models" else MODELS[:2]

        results = []

        for i, model_config in enumerate(models_to_run, 1):
            model = model_config["model"]
            provider = model_config["provider"]
            tier = model_config["tier"]

            # Child agent per provider
            @observe(
                agent_name="litellm-provider-caller",
                workflow_name=f"call-{provider.lower()}",
                session_id=session,
                user_id=user_id,
                user_group=user_group,
                client=observe_client,
                enforce_policy=False,
            )
            async def call_provider(
                query: str,
                model: str = model,
                provider: str = provider,
                tier: str = tier,
            ) -> dict:
                """Call a single LLM provider and return results."""

                waxell_observe.tag("provider", provider.lower())
                waxell_observe.tag("tier", tier)
                waxell_observe.metadata("model", model)

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

                waxell_observe.score("tokens_in", float(tokens_in))
                waxell_observe.score("tokens_out", float(tokens_out))
                waxell_observe.metadata("content_length", len(content))

                return {
                    "provider": provider,
                    "model": model,
                    "tier": tier,
                    "content": content,
                    "tokens": tokens_in + tokens_out,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }

            print(f"[LiteLLM Demo] Step {i}/{len(models_to_run)}: Calling {provider} ({tier})...")
            print(f"  Model: {model}")

            result = await call_provider(query)
            results.append(result)

            print(f"  Response: {result['content'][:100]}...")
            print(f"  Tokens: {result['tokens_in']} in + {result['tokens_out']} out = {result['tokens']}")
            print()

        # Reasoning: compare results across providers
        print(f"[LiteLLM Demo] Step {len(models_to_run) + 1}/{len(models_to_run) + 1}: Comparing results...")
        comparison = await compare_provider_results(results)

        for r in results:
            print(f"  {r['provider']:10s} ({r['tier']:12s}): {r['tokens']:4d} tokens, {len(r['content']):4d} chars")

        # Record final scores and metadata
        total_tokens = sum(r["tokens"] for r in results)
        waxell_observe.score("total_tokens", float(total_tokens))
        waxell_observe.score("models_compared", float(len(results)))
        waxell_observe.metadata("comparison_conclusion",
                                comparison["conclusion"] if isinstance(comparison, dict) else str(comparison))

        print()
        print(f"[LiteLLM Demo] Complete. ({len(results)} LLM calls via LiteLLM)")

        return {
            "comparison": {
                r["provider"]: {
                    "model": r["model"],
                    "tokens": r["tokens"],
                    "content_length": len(r["content"]),
                }
                for r in results
            },
            "models_tested": len(results),
            "total_tokens": total_tokens,
        }

    try:
        await litellm_orchestrator(query)
    except PolicyViolationError as e:
        print(f"\n[LiteLLM Demo] POLICY VIOLATION: {e}")
        print("[LiteLLM Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
