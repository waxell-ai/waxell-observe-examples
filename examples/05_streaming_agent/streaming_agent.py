#!/usr/bin/env python3
"""Streaming comparison demo — OpenAI vs Anthropic.

Demonstrates waxell-observe SDK decorators with streaming LLM calls
in a multi-agent comparison pipeline:

  streaming-orchestrator (parent)
  ├── @step: preprocess_query
  ├── @decision: choose_comparison_mode
  ├── openai-streamer (child agent) — streaming OpenAI call
  ├── anthropic-streamer (child agent) — streaming Anthropic call
  ├── @reasoning: compare_streaming_results
  └── scores, tags, metadata

Usage::

    python examples/05_streaming_agent/streaming_agent.py --dry-run
    python examples/05_streaming_agent/streaming_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from _common import setup_observe
setup_observe()

import waxell_observe as waxell
from waxell_observe import generate_session_id
from _common import (
    get_observe_client, get_streaming_openai_client, get_streaming_anthropic_client,
    is_observe_disabled, parse_args, pick_demo_user,
)

DEFAULT_QUERY = "Explain quantum computing in simple terms"


@waxell.step_dec(name="preprocess_query")
def preprocess_query(query: str) -> dict:
    cleaned = query.strip()
    word_count = len(cleaned.split())
    return {"cleaned_query": cleaned, "word_count": word_count}


@waxell.decision(name="choose_comparison_mode", options=["side_by_side", "sequential", "race"])
def choose_comparison_mode(query_info: dict) -> dict:
    chosen = "sequential"
    reasoning = "Sequential comparison — run OpenAI first, then Anthropic for clear comparison"
    return {"chosen": chosen, "reasoning": reasoning, "confidence": 0.90}


@waxell.reasoning_dec(step="compare_streaming_results")
def compare_streaming_results(openai_result: dict, anthropic_result: dict) -> dict:
    oi_len = openai_result.get("content_length", 0)
    an_len = anthropic_result.get("content_length", 0)
    oi_tokens = openai_result.get("total_tokens", 0)
    an_tokens = anthropic_result.get("total_tokens", 0)
    winner = "openai" if oi_len > an_len else "anthropic" if an_len > oi_len else "tie"
    return {
        "openai_content_length": oi_len,
        "anthropic_content_length": an_len,
        "openai_tokens": oi_tokens,
        "anthropic_tokens": an_tokens,
        "longer_response": winner,
        "reasoning": f"OpenAI: {oi_len} chars/{oi_tokens} tokens, Anthropic: {an_len} chars/{an_tokens} tokens",
    }


@waxell.observe(agent_name="openai-streamer", workflow_name="openai-streaming", capture_io=True)
async def run_openai_stream(query: str, client, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Stream a response from OpenAI."""
    waxell.tag("provider", "openai")
    waxell.tag("mode", "streaming")

    print("[OpenAI Streamer] Streaming response...")
    print("  OpenAI: ", end="", flush=True)

    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    content = ""
    tokens_in = 0
    tokens_out = 0
    chunk_count = 0

    async for chunk in stream:
        chunk_count += 1
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            content += text
            print(text, end="", flush=True)
        if chunk.usage:
            tokens_in = getattr(chunk.usage, "prompt_tokens", 150)
            tokens_out = getattr(chunk.usage, "completion_tokens", 80)

    if not tokens_out:
        tokens_in = 150
        tokens_out = max(len(content.split()) * 2, 80)

    print()
    print(f"  ({chunk_count} chunks, {tokens_in}+{tokens_out} tokens)")

    waxell.score("stream_quality", 0.85, comment="OpenAI streaming quality")
    return {
        "content": content, "content_length": len(content),
        "chunks": chunk_count, "tokens_in": tokens_in, "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }


@waxell.observe(agent_name="anthropic-streamer", workflow_name="anthropic-streaming", capture_io=True)
async def run_anthropic_stream(query: str, client, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    """Stream a response from Anthropic."""
    waxell.tag("provider", "anthropic")
    waxell.tag("mode", "streaming")

    print("[Anthropic Streamer] Streaming response...")
    print("  Anthropic: ", end="", flush=True)

    stream = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=500,
        messages=[{"role": "user", "content": query}],
        stream=True,
    )

    content = ""
    tokens_in = 0
    tokens_out = 0
    event_count = 0

    async for event in stream:
        event_count += 1
        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
            text = event.delta.text
            content += text
            print(text, end="", flush=True)
        elif event.type == "message_start" and hasattr(event, "message"):
            if hasattr(event.message, "usage"):
                tokens_in = event.message.usage.input_tokens
        elif event.type == "message_delta" and hasattr(event, "usage"):
            if event.usage:
                tokens_out = event.usage.output_tokens

    if not tokens_out:
        tokens_in = 150
        tokens_out = max(len(content.split()) * 2, 80)

    print()
    print(f"  ({event_count} events, {tokens_in}+{tokens_out} tokens)")

    waxell.score("stream_quality", 0.87, comment="Anthropic streaming quality")
    return {
        "content": content, "content_length": len(content),
        "events": event_count, "tokens_in": tokens_in, "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
    }


@waxell.observe(agent_name="streaming-orchestrator", workflow_name="streaming-comparison", capture_io=True)
async def run_agent(query: str, *, dry_run: bool = False, waxell_ctx=None) -> dict:
    waxell.tag("demo", "streaming")
    waxell.tag("providers", "openai,anthropic")
    waxell.metadata("comparison_type", "streaming")

    openai_client = get_streaming_openai_client(dry_run=dry_run)
    anthropic_client = get_streaming_anthropic_client(dry_run=dry_run)

    query_info = preprocess_query(query)
    print(f"[Streaming Demo] Preprocessed: {query_info['word_count']} words")

    mode = choose_comparison_mode(query_info)
    print(f"[Streaming Demo] Mode: {mode['chosen']}")
    print()

    # OpenAI streaming
    openai_result = await run_openai_stream(query, openai_client, dry_run=dry_run, waxell_ctx=waxell_ctx)
    print()

    # Anthropic streaming
    anthropic_result = await run_anthropic_stream(query, anthropic_client, dry_run=dry_run, waxell_ctx=waxell_ctx)
    print()

    # Compare
    comparison = compare_streaming_results(openai_result, anthropic_result)

    waxell.score("comparison_quality", 0.86, comment="Overall streaming comparison")

    print(f"[Streaming Demo] OpenAI:    {openai_result['content_length']} chars, {openai_result['total_tokens']} tokens")
    print(f"[Streaming Demo] Anthropic: {anthropic_result['content_length']} chars, {anthropic_result['total_tokens']} tokens")
    print(f"\n[Streaming Demo] Complete. (2 streaming LLM calls, 2 child agents)")

    return {
        "openai_response": openai_result["content"][:500],
        "anthropic_response": anthropic_result["content"][:500],
        "comparison": comparison,
    }


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY
    session = generate_session_id()
    demo_user = pick_demo_user()

    print("[Streaming Demo] Starting streaming comparison...")
    print(f"[Streaming Demo] Session: {session}")
    print(f"[Streaming Demo] Query: {query}")
    print()

    await run_agent(query, dry_run=args.dry_run)
    waxell_observe.shutdown()


if __name__ == "__main__":
    import waxell_observe
    asyncio.run(main())
