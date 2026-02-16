#!/usr/bin/env python3
"""Multi-step RAG agent demo.

Demonstrates waxell-observe's zero-code auto-instrumentation with plain
Python + OpenAI. The agent performs document retrieval, relevance filtering,
and answer synthesis across 4 observable steps.

Usage::

    # Dry-run (no OpenAI API key needed)
    python examples/02_rag_agent/rag_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/02_rag_agent/rag_agent.py

    # Custom query
    python examples/02_rag_agent/rag_agent.py --dry-run --query "How do I monitor AI agents?"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import openai (auto-instrumentor has patched it)
import waxell_observe
from waxell_observe import WaxellContext, generate_session_id
from waxell_observe.errors import PolicyViolationError

from _common import (
    DEMO_DOCUMENTS,
    LARGE_PROMPT,
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
    retrieve_documents,
    simulate_slow_operation,
)

DEFAULT_QUERY = "What are the best practices for AI safety and model deployment?"


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY
    policy_triggers = args.policy_triggers

    client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    # When observe is disabled, skip policy checks and mid-execution governance
    # (no server to check against). When enabled, enforce both.
    observe_active = not is_observe_disabled()

    if policy_triggers:
        print("[RAG Demo] POLICY TRIGGER MODE -- intentionally crossing policy thresholds")
        print("[RAG Demo] Expected triggers: budget (large prompt), safety (many steps), latency (slow operation)")
        print()

    print(f"[RAG Demo] Starting document QA pipeline...")
    print(f"[RAG Demo] Session: {session}")
    print(f"[RAG Demo] End user: {user_id} ({user_group})")
    print(f"[RAG Demo] Query: {query}")
    print()

    async with WaxellContext(
        agent_name="rag-demo",
        workflow_name="document-qa",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "rag")
        ctx.set_tag("query_type", "multi-step")
        ctx.set_metadata("document_corpus_size", len(DEMO_DOCUMENTS))

        try:
            # ------------------------------------------------------------------
            # Step 1/4: Query Analysis (chain-of-thought)
            # ------------------------------------------------------------------
            print("[RAG Demo] Step 1/4: Analyzing query...")

            # Budget trigger: use LARGE_PROMPT to exceed per_workflow_token_limit
            analysis_content = LARGE_PROMPT if policy_triggers else query

            analysis_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a search query analyzer. Identify 3-5 key "
                            "search terms from the user's question."
                        ),
                    },
                    {"role": "user", "content": analysis_content},
                ],
            )
            analysis = analysis_response.choices[0].message.content

            ctx.record_step("analyze_query", output={"terms": analysis})
            ctx.record_llm_call(
                model=analysis_response.model,
                tokens_in=analysis_response.usage.prompt_tokens,
                tokens_out=analysis_response.usage.completion_tokens,
                task="analyze_query",
                prompt_preview=analysis_content[:200],
                response_preview=analysis[:200],
            )
            print(f"           Analysis: {analysis[:120]}...")

            # ------------------------------------------------------------------
            # Step 2/4: Document Retrieval (tool call simulation)
            # ------------------------------------------------------------------
            print("[RAG Demo] Step 2/4: Retrieving documents...")

            retrieved = retrieve_documents(query)
            doc_ids = [d["id"] for d in retrieved]

            ctx.record_step(
                "retrieve_documents",
                output={"num_docs": len(retrieved), "doc_ids": doc_ids},
            )
            print(f"           Found {len(retrieved)} relevant documents: {doc_ids}")

            # ------------------------------------------------------------------
            # Step 3/4: Relevance Filtering (chain-of-thought)
            # ------------------------------------------------------------------
            print("[RAG Demo] Step 3/4: Filtering for relevance...")

            doc_summaries = "\n".join(
                f"- Document {d['id']}: {d['title']} -- {d['content'][:100]}..."
                for d in retrieved
            )
            filter_prompt = (
                f"Query: {query}\n\n"
                f"Retrieved documents:\n{doc_summaries}\n\n"
                "Select the most relevant documents and explain why."
            )

            filter_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document relevance evaluator. Given these "
                            "documents and the query, select the most relevant ones."
                        ),
                    },
                    {"role": "user", "content": filter_prompt},
                ],
            )
            filter_result = filter_response.choices[0].message.content

            ctx.record_step("filter_documents", output={"selected": len(retrieved)})
            ctx.record_llm_call(
                model=filter_response.model,
                tokens_in=filter_response.usage.prompt_tokens,
                tokens_out=filter_response.usage.completion_tokens,
                task="filter_documents",
                prompt_preview=filter_prompt[:200],
                response_preview=filter_result[:200],
            )
            print(f"           Filtering: {filter_result[:120]}...")

            # ------------------------------------------------------------------
            # Policy trigger: extra tool steps (safety category)
            # ------------------------------------------------------------------
            if policy_triggers:
                print("[RAG Demo] [Policy Trigger] Recording extra tool steps...")
                ctx.record_step("tool:web_scraper", output={"url": "https://example.com"})
                print("           Recorded tool:web_scraper (blocked_tools trigger)")
                ctx.record_step("tool:data_lookup", output={"query": "internal"})
                print("           Recorded tool:data_lookup (exceeds max_steps=3)")

            # ------------------------------------------------------------------
            # Policy trigger: slow operation (operations category)
            # ------------------------------------------------------------------
            if policy_triggers:
                print("[RAG Demo] [Policy Trigger] Simulating slow operation (4s > timeout 3s)...")
                await simulate_slow_operation(4.0)
                print("           Slow operation complete (latency trigger)")

            # ------------------------------------------------------------------
            # Step 4/4: Answer Synthesis (final LLM call)
            # ------------------------------------------------------------------
            print("[RAG Demo] Step 4/4: Synthesizing answer...")

            synthesis_context = "\n\n".join(
                f"[{d['title']}]\n{d['content']}" for d in retrieved
            )
            synthesis_prompt = (
                f"Original question: {query}\n\n"
                f"Reference documents:\n{synthesis_context}\n\n"
                "Synthesize a comprehensive answer based on these documents."
            )

            synthesis_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful research assistant. Synthesize an "
                            "answer from the provided documents."
                        ),
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
            )
            final_answer = synthesis_response.choices[0].message.content

            ctx.record_step("synthesize_answer", output={"answer_length": len(final_answer)})
            ctx.record_llm_call(
                model=synthesis_response.model,
                tokens_in=synthesis_response.usage.prompt_tokens,
                tokens_out=synthesis_response.usage.completion_tokens,
                task="synthesize_answer",
                prompt_preview=synthesis_prompt[:200],
                response_preview=final_answer[:200],
            )

            # ------------------------------------------------------------------
            # Set final result
            # ------------------------------------------------------------------
            ctx.set_result(
                {
                    "answer": final_answer,
                    "documents_used": len(retrieved),
                    "steps": 4,
                }
            )

            print()
            answer_display = final_answer[:200] + ("..." if len(final_answer) > 200 else "")
            print(f"[RAG Demo] Answer: {answer_display}")
            print(
                f"[RAG Demo] Complete. "
                f"(3 LLM calls, 4 steps, {len(retrieved)} documents used)"
            )

        except PolicyViolationError as e:
            print(f"\n[RAG Demo] POLICY VIOLATION: {e}")
            print("[RAG Demo] Agent halted by governance policy.")

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
