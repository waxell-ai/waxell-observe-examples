#!/usr/bin/env python3
"""Research agent demo with agentic behavior tracking.

Demonstrates waxell-observe's new behavior tracking methods (record_tool_call,
record_retrieval, record_decision, record_reasoning, record_retry) through a
multi-step research pipeline that classifies a query, retrieves documents,
searches the web, reasons about sources, and synthesizes a final answer.

Usage::

    # Dry-run (no OpenAI API key needed)
    python examples/08_research_agent/research_agent.py --dry-run

    # With real OpenAI calls
    OPENAI_API_KEY=sk-... python examples/08_research_agent/research_agent.py

    # Custom query
    python examples/08_research_agent/research_agent.py --dry-run --query "How do multi-agent systems coordinate?"
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
    get_observe_client,
    get_openai_client,
    is_observe_disabled,
    parse_args,
    pick_demo_user,
    retrieve_documents,
)

DEFAULT_QUERY = "What are the best practices for AI safety and responsible deployment?"


async def run_research_agent(
    query: str,
    client: object,
    observe_active: bool = True,
    session: str = "",
    user_id: str = "",
    user_group: str = "",
) -> dict:
    """Execute the research agent pipeline.

    Args:
        query: The research query to investigate.
        client: OpenAI-compatible async client (real or mock).
        observe_active: Whether observe/governance is active.
        session: Session ID for trace correlation.
        user_id: End-user identifier.
        user_group: End-user group/tier.

    Returns:
        Dict with final research results.
    """
    async with WaxellContext(
        agent_name="research-agent",
        workflow_name="deep-research",
        inputs={"query": query},
        enforce_policy=observe_active,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    ) as ctx:
        ctx.set_tag("demo", "research")
        ctx.set_tag("pipeline", "deep-research")
        ctx.set_metadata("document_corpus_size", len(DEMO_DOCUMENTS))

        try:
            # ------------------------------------------------------------------
            # Step 1: Classify Query
            # ------------------------------------------------------------------
            print("[Research] Step 1: Classifying query...")

            classify_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the query into one of: technical, general, opinion. "
                            "Also identify the primary domain (safety, deployment, cost, architecture)."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            classification = classify_response.choices[0].message.content

            ctx.record_step("classify_query", output={"classification": classification[:200]})
            ctx.record_llm_call(
                model=classify_response.model,
                tokens_in=classify_response.usage.prompt_tokens,
                tokens_out=classify_response.usage.completion_tokens,
                task="classify_query",
                prompt_preview=query[:200],
                response_preview=classification[:200],
            )
            print(f"           Classification: {classification[:120]}...")

            # ------------------------------------------------------------------
            # Step 2: Decide Research Strategy
            # ------------------------------------------------------------------
            print("[Research] Step 2: Deciding research strategy...")

            ctx.record_decision(
                name="research_strategy",
                options=["deep_research", "quick_search", "balanced_perspectives"],
                chosen="deep_research",
                reasoning="Technical query requires multi-source analysis with cross-referencing",
                confidence=0.92,
            )
            print("           Strategy: deep_research (confidence: 0.92)")

            # ------------------------------------------------------------------
            # Step 3: Retrieve Documents from Knowledge Base
            # ------------------------------------------------------------------
            print("[Research] Step 3: Retrieving documents from knowledge base...")

            retrieved = retrieve_documents(query)

            # Format documents for the retrieval record
            retrieval_docs = []
            for doc in retrieved:
                # Calculate a mock relevance score based on position
                score = round(0.95 - (retrieved.index(doc) * 0.12), 2)
                retrieval_docs.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "score": score,
                    "snippet": doc["content"][:80] + "...",
                })

            ctx.record_retrieval(
                query=query,
                documents=retrieval_docs,
                source="document_store",
                duration_ms=45,
                top_k=5,
            )
            print(f"           Retrieved {len(retrieved)} docs from document_store (45ms)")

            # ------------------------------------------------------------------
            # Step 4: Web Search (tool call)
            # ------------------------------------------------------------------
            print("[Research] Step 4: Executing web search...")

            web_results = [
                {"url": "https://example.com/ai-safety-2024", "title": "AI Safety Standards 2024", "relevance": 0.91},
                {"url": "https://example.com/deployment-guide", "title": "Production AI Deployment Guide", "relevance": 0.84},
                {"url": "https://example.com/responsible-ai", "title": "Responsible AI Framework", "relevance": 0.79},
            ]

            ctx.record_tool_call(
                name="web_search",
                input={"query": "AI safety best practices 2024", "max_results": 5},
                output={"results": web_results, "total_found": 3},
                status="ok",
                duration_ms=250,
                tool_type="api",
            )
            print(f"           Web search returned {len(web_results)} results (250ms)")

            # ------------------------------------------------------------------
            # Step 5: Calculator (tool call)
            # ------------------------------------------------------------------
            print("[Research] Step 5: Computing relevance statistics...")

            all_scores = [d["score"] for d in retrieval_docs] + [r["relevance"] for r in web_results]
            avg_score = round(sum(all_scores) / len(all_scores), 2)

            ctx.record_tool_call(
                name="calculator",
                input={"operation": "average", "values": all_scores},
                output={"result": avg_score},
                status="ok",
                duration_ms=5,
                tool_type="function",
            )
            print(f"           Average relevance score: {avg_score}")

            # ------------------------------------------------------------------
            # Step 6: Reasoning Chain (3 steps)
            # ------------------------------------------------------------------
            print("[Research] Step 6: Reasoning about sources...")

            ctx.record_reasoning(
                step="evaluate_sources",
                thought=(
                    "Document 1 (AI Safety Guidelines) covers safety guardrails comprehensively "
                    "with specific practices like red-teaming and adversarial testing. The web "
                    "results reinforce this with 2024-specific standards. Document 3 on cost "
                    "optimization adds the budgeting dimension of safety."
                ),
                evidence=["doc-001", "web-result-1", "doc-003"],
                conclusion="Strong foundation for safety analysis with multi-source corroboration",
            )
            print("           [1/3] evaluate_sources: Strong foundation established")

            ctx.record_reasoning(
                step="check_consistency",
                thought=(
                    "Doc-001 and web results agree on safety guidelines: both emphasize "
                    "red-teaming, staged rollouts, and monitoring. Doc-002 on deployment "
                    "patterns aligns with the deployment-specific web result. No conflicting "
                    "information detected across knowledge base and web sources."
                ),
                evidence=["doc-001", "doc-002", "web-result-2"],
                conclusion="Sources are consistent; no contradictions found",
            )
            print("           [2/3] check_consistency: Sources are consistent")

            ctx.record_reasoning(
                step="identify_gaps",
                thought=(
                    "The retrieved sources cover safety guidelines, deployment patterns, "
                    "and cost optimization well. However, no sources specifically address "
                    "deployment-specific safety measures for multi-agent architectures. "
                    "Doc-004 on multi-agent patterns exists but was not in the top-k retrieval."
                ),
                conclusion="Gap identified: multi-agent safety patterns not covered in retrieval",
            )
            print("           [3/3] identify_gaps: Gap identified in multi-agent safety")

            # ------------------------------------------------------------------
            # Step 7: Decide on Additional Research
            # ------------------------------------------------------------------
            print("[Research] Step 7: Deciding if additional research is needed...")

            ctx.record_decision(
                name="additional_research",
                options=["sufficient", "expand_search", "expert_review"],
                chosen="sufficient",
                reasoning=(
                    "Source quality meets threshold (avg relevance {:.2f}). "
                    "Gap in multi-agent safety is noted but not critical for "
                    "the primary query about general AI safety practices."
                ).format(avg_score),
                confidence=0.88,
            )
            print("           Decision: sufficient (confidence: 0.88)")

            # ------------------------------------------------------------------
            # Step 8: Synthesize Final Answer
            # ------------------------------------------------------------------
            print("[Research] Step 8: Synthesizing final answer...")

            synthesis_context = "\n\n".join(
                f"[{d['title']}]\n{d['content']}" for d in retrieved
            )
            synthesis_prompt = (
                f"Research query: {query}\n\n"
                f"Knowledge base documents:\n{synthesis_context}\n\n"
                f"Web search results:\n"
                + "\n".join(f"- {r['title']} (relevance: {r['relevance']})" for r in web_results)
                + "\n\nSynthesize a comprehensive, well-structured answer."
            )

            synthesis_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Synthesize findings from "
                            "multiple sources into a comprehensive, well-structured answer."
                        ),
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
            )
            final_answer = synthesis_response.choices[0].message.content

            ctx.record_step("synthesize", output={"answer_length": len(final_answer)})
            ctx.record_llm_call(
                model=synthesis_response.model,
                tokens_in=synthesis_response.usage.prompt_tokens,
                tokens_out=synthesis_response.usage.completion_tokens,
                task="synthesize",
                prompt_preview=synthesis_prompt[:200],
                response_preview=final_answer[:200],
            )
            print(f"           Synthesis complete ({len(final_answer)} chars)")

            # ------------------------------------------------------------------
            # Step 9: Quality Score
            # ------------------------------------------------------------------
            ctx.record_score("research_quality", 0.87)
            print("[Research] Quality score: 0.87")

            # ------------------------------------------------------------------
            # Set final result
            # ------------------------------------------------------------------
            result = {
                "answer": final_answer,
                "documents_used": len(retrieved),
                "web_results_used": len(web_results),
                "avg_relevance": avg_score,
                "strategy": "deep_research",
            }
            ctx.set_result(result)

            print()
            print(f"[Research] Answer: {final_answer[:200]}...")
            print(
                f"[Research] Complete. "
                f"(2 LLM calls, 2 tool calls, 1 retrieval, 3 reasoning steps, "
                f"2 decisions, {len(retrieved)} documents)"
            )
            return result

        except PolicyViolationError as e:
            print(f"\n[Research] POLICY VIOLATION: {e}")
            print("[Research] Agent halted by governance policy.")
            return {"error": str(e)}


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    client = get_openai_client(dry_run=args.dry_run)
    session = generate_session_id()
    observe_active = not is_observe_disabled()

    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]

    print("[Research Agent] Starting deep research pipeline...")
    print(f"[Research Agent] Session: {session}")
    print(f"[Research Agent] End user: {user_id} ({user_group})")
    print(f"[Research Agent] Query: {query[:80]}...")
    print()

    await run_research_agent(
        query=query,
        client=client,
        observe_active=observe_active,
        session=session,
        user_id=user_id,
        user_group=user_group,
    )

    waxell_observe.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
