#!/usr/bin/env python3
"""All-providers-in-one-trace demo (modern decorator patterns).

Demonstrates multi-provider tracing by calling OpenAI, Anthropic, and
LiteLLM within a single trace using child agent decorators.

Architecture::

  all-providers-orchestrator (parent)
  ├── @step: preprocess_query
  ├── @decision: choose_provider_order
  ├── openai-agent (child)
  │   └── Auto-instrumented OpenAI call
  ├── anthropic-agent (child)
  │   └── Auto-instrumented Anthropic call
  ├── litellm-agent (child)
  │   └── @tool: call_litellm (LiteLLM/Groq call)
  └── @reasoning: compare_all_responses

SDK primitives demonstrated:

  | Primitive   | Decorator      | Manual          | Auto               |
  |-------------|----------------|-----------------|--------------------|
  | LLM calls   | --             | --              | OpenAI + Anthropic |
  | Tool calls  | @tool          | --              | --                 |
  | Decisions   | @decision      | --              | --                 |
  | Reasoning   | @reasoning     | --              | --                 |
  | Steps       | @step          | --              | --                 |
  | Scores      | --             | score()         | --                 |
  | Tags        | --             | tag()           | --                 |
  | Metadata    | --             | metadata()      | --                 |

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

import waxell_observe as waxell
from waxell_observe import generate_session_id
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


# ---------------------------------------------------------------------------
# @tool decorator -- LiteLLM helper (kept from original)
# ---------------------------------------------------------------------------


@waxell.tool(name="call_litellm", tool_type="llm")
async def call_litellm(model: str, messages: list, dry_run: bool):
    """Call LiteLLM or return a mock response."""
    if dry_run:
        content = _contextual_response(messages)
        return MockChatCompletion(content=content, model=model.split("/")[-1])

    import litellm
    return await litellm.acompletion(model=model, messages=messages)


# ---------------------------------------------------------------------------
# @step decorator -- preprocess query
# ---------------------------------------------------------------------------


@waxell.step_dec(name="preprocess_query")
async def preprocess_query(query: str) -> dict:
    """Clean and normalize the query for multi-provider analysis."""
    cleaned = query.strip()
    tokens = cleaned.lower().split()
    return {"original": query, "cleaned": cleaned, "token_count": len(tokens)}


# ---------------------------------------------------------------------------
# @decision decorator -- choose provider execution order
# ---------------------------------------------------------------------------


@waxell.decision(
    name="choose_provider_order",
    options=["fastest_first", "quality_first", "round_robin"],
)
async def choose_provider_order(query: str, num_providers: int) -> dict:
    """Decide the order in which to call providers."""
    # Heuristic: short queries prioritize speed, longer ones prioritize quality
    word_count = len(query.split())
    if word_count < 10:
        return {
            "chosen": "fastest_first",
            "reasoning": f"Short query ({word_count} words) -- prioritize low-latency providers (Groq first)",
            "confidence": 0.88,
        }
    elif word_count > 25:
        return {
            "chosen": "quality_first",
            "reasoning": f"Complex query ({word_count} words) -- prioritize quality providers (OpenAI/Anthropic first)",
            "confidence": 0.85,
        }
    return {
        "chosen": "round_robin",
        "reasoning": f"Medium query ({word_count} words) -- round robin for balanced comparison",
        "confidence": 0.75,
    }


# ---------------------------------------------------------------------------
# @reasoning decorator -- compare all provider responses
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="compare_all_responses")
async def compare_all_responses(responses: list[tuple[str, str]]) -> dict:
    """Compare responses from all providers and draw conclusions."""
    provider_names = [p for p, _ in responses]
    lengths = {p: len(c) for p, c in responses}

    longest = max(lengths, key=lengths.get)
    shortest = min(lengths, key=lengths.get)

    return {
        "thought": (
            f"Compared {len(responses)} provider responses: {', '.join(provider_names)}. "
            f"Response lengths range from {min(lengths.values())} to {max(lengths.values())} characters. "
            f"Each provider brings a distinct perspective to the analysis."
        ),
        "evidence": [
            f"{p}: {length} chars" for p, length in lengths.items()
        ],
        "conclusion": (
            f"{longest} provided the most detailed response ({lengths[longest]} chars), "
            f"while {shortest} was most concise ({lengths[shortest]} chars). "
            f"Combined perspectives yield comprehensive multi-provider analysis."
        ),
    }


# ---------------------------------------------------------------------------
# Child Agent 1: OpenAI
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="openai-agent", workflow_name="openai-generation")
async def run_openai(query: str, openai_client, waxell_ctx=None):
    """Generate response using OpenAI."""
    waxell.tag("provider", "openai")
    waxell.tag("agent_role", "llm_provider")
    waxell.metadata("model", "gpt-4o-mini")

    print("  [OpenAI] Calling gpt-4o-mini...")
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a tech industry analyst. Be concise."},
            {"role": "user", "content": f"From OpenAI's perspective: {query}"},
        ],
    )
    content = response.choices[0].message.content

    waxell.score("response_quality", 0.88, comment="OpenAI response quality")
    waxell.metadata("tokens_total", response.usage.prompt_tokens + response.usage.completion_tokens)

    print(f"    Response: {content[:100]}...")
    return {"provider": "OpenAI", "content": content, "model": response.model}


# ---------------------------------------------------------------------------
# Child Agent 2: Anthropic
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="anthropic-agent", workflow_name="anthropic-generation")
async def run_anthropic(query: str, anthropic_client, waxell_ctx=None):
    """Generate response using Anthropic."""
    waxell.tag("provider", "anthropic")
    waxell.tag("agent_role", "llm_provider")
    waxell.metadata("model", "claude-sonnet-4-5-20250929")

    print("  [Anthropic] Calling claude-sonnet...")
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[
            {"role": "user", "content": f"From Anthropic's perspective: {query}"},
        ],
    )
    content = response.content[0].text

    waxell.score("response_quality", 0.91, comment="Anthropic response quality")
    waxell.metadata("tokens_total", response.usage.input_tokens + response.usage.output_tokens)

    print(f"    Response: {content[:100]}...")
    return {"provider": "Anthropic", "content": content, "model": response.model}


# ---------------------------------------------------------------------------
# Child Agent 3: LiteLLM/Groq
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="litellm-agent", workflow_name="litellm-generation")
async def run_litellm(query: str, dry_run: bool = False, waxell_ctx=None):
    """Generate response using LiteLLM/Groq."""
    waxell.tag("provider", "groq")
    waxell.tag("agent_role", "llm_provider")
    waxell.metadata("model", "llama-3.3-70b-versatile")

    print("  [LiteLLM/Groq] Calling llama-3.3-70b...")
    response = await call_litellm(
        model="groq/llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an open-source AI advocate. Be concise."},
            {"role": "user", "content": f"From the open-source community's perspective: {query}"},
        ],
        dry_run=dry_run,
    )
    content = response.choices[0].message.content

    waxell.score("response_quality", 0.82, comment="Groq/LiteLLM response quality")
    waxell.metadata("tokens_total", response.usage.prompt_tokens + response.usage.completion_tokens)

    print(f"    Response: {content[:100]}...")
    return {"provider": "Groq/LiteLLM", "content": content, "model": response.model}


# ---------------------------------------------------------------------------
# Orchestrator -- coordinates all providers in one trace
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="all-providers-orchestrator", workflow_name="provider-showcase")
async def run_pipeline(query: str, dry_run: bool = False, waxell_ctx=None):
    """Coordinate multi-provider trace across OpenAI, Anthropic, and LiteLLM.

    This is the parent agent. All child agents auto-link to this parent
    via WaxellContext lineage.
    """
    waxell.tag("demo", "all-providers")
    waxell.tag("providers", "openai,anthropic,litellm")
    waxell.metadata("num_providers", 3)
    waxell.metadata("mode", "dry-run" if dry_run else "live")

    openai_client = get_openai_client(dry_run=dry_run)
    anthropic_client = get_anthropic_client(dry_run=dry_run)

    try:
        # Phase 1: @step -- preprocess query
        print("[Orchestrator] Phase 1: Preprocessing query (@step)...")
        preprocessed = await preprocess_query(query)
        print(f"  Preprocessed: {preprocessed['token_count']} tokens")

        # Phase 2: @decision -- choose provider order
        print("[Orchestrator] Phase 2: Choosing provider order (@decision)...")
        order = await choose_provider_order(query=query, num_providers=3)
        chosen_order = order.get("chosen", "round_robin") if isinstance(order, dict) else str(order)
        print(f"  Order: {chosen_order}")

        all_responses = []

        # Phase 3: openai-agent child
        print("[Orchestrator] Phase 3: OpenAI (gpt-4o-mini)...")
        openai_result = await run_openai(query=query, openai_client=openai_client)
        all_responses.append((openai_result["provider"], openai_result["content"]))
        print()

        # Phase 4: anthropic-agent child
        print("[Orchestrator] Phase 4: Anthropic (claude-sonnet)...")
        anthropic_result = await run_anthropic(query=query, anthropic_client=anthropic_client)
        all_responses.append((anthropic_result["provider"], anthropic_result["content"]))
        print()

        # Phase 5: litellm-agent child
        print("[Orchestrator] Phase 5: LiteLLM/Groq (llama-3.3-70b)...")
        litellm_result = await run_litellm(query=query, dry_run=dry_run)
        all_responses.append((litellm_result["provider"], litellm_result["content"]))
        print()

        # Phase 6: @reasoning -- compare all responses
        print("[Orchestrator] Phase 6: Comparing all responses (@reasoning)...")
        comparison = await compare_all_responses(responses=all_responses)
        print(f"  Conclusion: {comparison.get('conclusion', 'N/A')[:120]}")
        print()

        # Scores
        waxell.score("multi_provider_coverage", 1.0, comment="All 3 providers responded")
        waxell.score("synthesis_quality", 0.87, comment="Multi-perspective synthesis rating")

        print(f"[All Providers] Complete. (3 LLM calls across 3 providers, 1 session)")

        return {
            "responses": {p: c[:500] for p, c in all_responses},
            "providers_used": [p for p, _ in all_responses],
            "provider_order": chosen_order,
        }

    except PolicyViolationError as e:
        print(f"\n[All Providers] POLICY VIOLATION: {e}")
        print("[All Providers] Agent halted by governance policy.")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY
    session = generate_session_id()
    observe_active = not is_observe_disabled()
    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[All Providers Multi-Agent Demo]")
    print(f"  Session:    {session}")
    print(f"  User:       {user_id} ({user_group})")
    print(f"  Query:      {query}")
    print(f"  Mode:       {'dry-run' if args.dry_run else 'LIVE (OpenAI + Anthropic + Groq)'}")
    print(f"  Providers:  OpenAI (gpt-4o-mini), Anthropic (claude-sonnet), LiteLLM/Groq (llama-3.3-70b)")
    print()

    result = await run_pipeline(
        query=query,
        dry_run=args.dry_run,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    )

    if "error" not in result:
        print()
        print("=" * 60)
        print("[Result]")
        print(f"  Providers: {result['providers_used']}")
        print(f"  Order:     {result['provider_order']}")
        for provider, content in result["responses"].items():
            print(f"  {provider}: {content[:100]}...")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
