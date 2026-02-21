#!/usr/bin/env python3
"""Multi-agent research pipeline demo with agentic behavior tracking.

Demonstrates modern SDK decorator patterns across 2 child agents:

  research-orchestrator (parent)
  ├── @step: preprocess_query
  ├── @decision: classify_query
  ├── waxell.decide(): research_strategy
  ├── research-searcher (child)
  │   ├── @tool(tool_type="api"): web_search
  │   ├── @tool(tool_type="function"): calculate_relevance
  │   ├── @retrieval(source="document_store"): retrieve_and_rank
  │   ├── @reasoning: evaluate_sources, check_consistency, identify_gaps
  │   └── @decision: additional_research
  └── research-synthesizer (child)
      ├── @reasoning: assess_quality
      ├── @decision: output_format
      └── LLM synthesis (auto-instrumented)

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

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import provider clients (auto-instrumentors have patched them)
import waxell_observe as waxell
from waxell_observe import generate_session_id
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


# ---------------------------------------------------------------------------
# @step decorator -- auto-record execution steps
# ---------------------------------------------------------------------------


@waxell.step_dec(name="preprocess_query")
async def preprocess_query(query: str) -> dict:
    """Clean and normalize the query."""
    cleaned = query.strip()
    tokens = cleaned.lower().split()
    return {"original": query, "cleaned": cleaned, "token_count": len(tokens)}


# ---------------------------------------------------------------------------
# @decision decorator -- classify the query type
# ---------------------------------------------------------------------------


@waxell.decision(name="classify_query", options=["technical", "general", "opinion"])
async def classify_query(query: str, openai_client) -> dict:
    """Classify query to determine research approach."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the query into one of: technical, general, opinion. "
                    "Also identify the primary domain (safety, deployment, cost, architecture). "
                    'Respond with JSON: {"chosen": "...", "reasoning": "...", "domain": "..."}'
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    content = response.choices[0].message.content
    try:
        import json
        return json.loads(content)
    except Exception:
        return {"chosen": "technical", "reasoning": content[:200], "domain": "safety"}


@waxell.decision(name="output_format", options=["brief", "detailed", "bullet_points"])
def choose_output_format(num_docs: int, num_web: int, context: str) -> dict:
    """Choose output format based on document count."""
    format_choice = "detailed" if num_docs > 2 else "brief"
    return {
        "chosen": format_choice,
        "reasoning": f"Research query with {num_docs} docs + {num_web} web results -- {format_choice} format",
        "confidence": 0.87,
    }


# ---------------------------------------------------------------------------
# @tool decorators -- auto-record tool calls
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="api")
def web_search(query: str, max_results: int = 5) -> dict:
    """Execute a web search for the given query."""
    results = [
        {"url": "https://example.com/ai-safety-2024", "title": "AI Safety Standards 2024", "relevance": 0.91},
        {"url": "https://example.com/deployment-guide", "title": "Production AI Deployment Guide", "relevance": 0.84},
        {"url": "https://example.com/responsible-ai", "title": "Responsible AI Framework", "relevance": 0.79},
    ]
    return {"results": results, "total_found": len(results)}


@waxell.tool(tool_type="function")
def calculate_relevance(scores: list) -> dict:
    """Calculate average relevance from a list of scores."""
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    return {"average": avg_score, "count": len(scores), "min": min(scores, default=0), "max": max(scores, default=0)}


# ---------------------------------------------------------------------------
# @retrieval decorator -- auto-record document retrieval
# ---------------------------------------------------------------------------


@waxell.retrieval(source="document_store")
def retrieve_and_rank(query: str) -> list[dict]:
    """Retrieve and rank documents from the knowledge base."""
    retrieved = retrieve_documents(query)
    ranked = []
    for i, doc in enumerate(retrieved):
        score = round(0.95 - (i * 0.12), 2)
        ranked.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": score,
            "snippet": doc["content"][:80] + "...",
        })
    return ranked


# ---------------------------------------------------------------------------
# @reasoning decorators -- auto-record chain-of-thought
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="evaluate_sources")
async def evaluate_sources(documents: list, web_results: list) -> dict:
    """Evaluate the quality and coverage of all sources."""
    return {
        "thought": (
            f"Document 1 (AI Safety Guidelines) covers safety guardrails comprehensively "
            f"with specific practices like red-teaming and adversarial testing. The web "
            f"results reinforce this with 2024-specific standards. Document 3 on cost "
            f"optimization adds the budgeting dimension of safety."
        ),
        "evidence": [f"doc-{d['id']}" for d in documents[:3]] + [f"web-result-{i+1}" for i in range(len(web_results))],
        "conclusion": "Strong foundation for safety analysis with multi-source corroboration",
    }


@waxell.reasoning_dec(step="check_consistency")
async def check_consistency(documents: list, web_results: list) -> dict:
    """Check consistency across knowledge base and web sources."""
    return {
        "thought": (
            "Doc-001 and web results agree on safety guidelines: both emphasize "
            "red-teaming, staged rollouts, and monitoring. Doc-002 on deployment "
            "patterns aligns with the deployment-specific web result. No conflicting "
            "information detected across knowledge base and web sources."
        ),
        "evidence": [f"doc-{d['id']}" for d in documents[:2]] + ["web-result-2"],
        "conclusion": "Sources are consistent; no contradictions found",
    }


@waxell.reasoning_dec(step="identify_gaps")
async def identify_gaps(documents: list) -> dict:
    """Identify gaps in the retrieved information."""
    return {
        "thought": (
            "The retrieved sources cover safety guidelines, deployment patterns, "
            "and cost optimization well. However, no sources specifically address "
            "deployment-specific safety measures for multi-agent architectures. "
            "Doc-004 on multi-agent patterns exists but was not in the top-k retrieval."
        ),
        "conclusion": "Gap identified: multi-agent safety patterns not covered in retrieval",
    }


@waxell.reasoning_dec(step="quality_assessment")
async def assess_answer_quality(answer: str, documents: list, web_results: list) -> dict:
    """Assess the quality of the synthesized research answer."""
    doc_titles = [d.get("title", "unknown") for d in documents]
    web_titles = [r.get("title", "unknown") for r in web_results]
    total_sources = len(doc_titles) + len(web_titles)
    mentioned = sum(1 for t in doc_titles + web_titles
                    if any(word in answer.lower() for word in t.lower().split()[:3]))
    return {
        "thought": f"Answer references approximately {mentioned}/{total_sources} sources. "
                   f"Checking for comprehensiveness and balanced coverage.",
        "evidence": [f"Source: {t}" for t in doc_titles + web_titles],
        "conclusion": "Research answer is comprehensive" if mentioned > total_sources // 2
                      else "Answer could cite more sources",
    }


# ---------------------------------------------------------------------------
# Agent 1: Research Searcher -- retrieval, web search, and reasoning
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="research-searcher", workflow_name="deep-search")
async def run_searcher(query: str, openai_client, waxell_ctx=None):
    """Search knowledge base and web, then reason about sources."""
    waxell.tag("agent_role", "searcher")
    waxell.tag("pipeline", "deep-research")
    waxell.metadata("document_corpus_size", len(DEMO_DOCUMENTS))

    # Step 1: Classify query (auto-instrumented LLM call)
    print("  [Searcher] Classifying query...")
    classify_response = await openai_client.chat.completions.create(
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
    print(f"    Classification: {classification[:120]}...")

    # Step 2: @retrieval -- retrieve and rank from knowledge base
    print("  [Searcher] Retrieving documents from knowledge base...")
    ranked_docs = retrieve_and_rank(query=query)
    print(f"    Retrieved {len(ranked_docs)} docs from document_store")

    # Step 3: @tool -- web search
    print("  [Searcher] Executing web search...")
    web_result = web_search(query="AI safety best practices 2024", max_results=5)
    web_results = web_result["results"]
    print(f"    Web search returned {len(web_results)} results")

    # Step 4: @tool -- calculate relevance statistics
    print("  [Searcher] Computing relevance statistics...")
    all_scores = [d["score"] for d in ranked_docs] + [r["relevance"] for r in web_results]
    stats = calculate_relevance(scores=all_scores)
    print(f"    Average relevance score: {stats['average']}")

    # Step 5: @reasoning chain (3 steps)
    print("  [Searcher] Reasoning about sources...")
    eval_result = await evaluate_sources(documents=ranked_docs, web_results=web_results)
    print(f"    [1/3] evaluate_sources: {eval_result.get('conclusion', 'N/A')}")

    consistency = await check_consistency(documents=ranked_docs, web_results=web_results)
    print(f"    [2/3] check_consistency: {consistency.get('conclusion', 'N/A')}")

    gaps = await identify_gaps(documents=ranked_docs)
    print(f"    [3/3] identify_gaps: {gaps.get('conclusion', 'N/A')}")

    # Step 6: @decision -- additional research needed?
    print("  [Searcher] Deciding if additional research is needed...")
    waxell.decide(
        "additional_research",
        chosen="sufficient",
        options=["sufficient", "expand_search", "expert_review"],
        reasoning=(
            f"Source quality meets threshold (avg relevance {stats['average']:.2f}). "
            "Gap in multi-agent safety is noted but not critical for "
            "the primary query about general AI safety practices."
        ),
        confidence=0.88,
    )
    print("    Decision: sufficient (confidence: 0.88)")

    waxell.score("source_quality", stats["average"], comment="Average relevance across all sources")

    return {
        "documents": ranked_docs,
        "web_results": web_results,
        "avg_relevance": stats["average"],
        "num_docs": len(ranked_docs),
        "num_web": len(web_results),
    }


# ---------------------------------------------------------------------------
# Agent 2: Research Synthesizer -- answer synthesis + quality assessment
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="research-synthesizer", workflow_name="research-synthesis")
async def run_synthesizer(query: str, documents: list, web_results: list,
                          openai_client, waxell_ctx=None):
    """Synthesize a comprehensive research answer."""
    waxell.tag("agent_role", "synthesizer")
    waxell.tag("provider", "openai")

    synthesis_context = "\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in documents
    )
    synthesis_prompt = (
        f"Research query: {query}\n\n"
        f"Knowledge base documents:\n{synthesis_context}\n\n"
        f"Web search results:\n"
        + "\n".join(f"- {r['title']} (relevance: {r['relevance']})" for r in web_results)
        + "\n\nSynthesize a comprehensive, well-structured answer."
    )

    print("  [Synthesizer] Generating research answer with OpenAI...")
    synthesis_response = await openai_client.chat.completions.create(
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
    answer = synthesis_response.choices[0].message.content

    # @reasoning -- assess answer quality
    print("  [Synthesizer] Assessing answer quality...")
    quality = await assess_answer_quality(answer=answer, documents=documents, web_results=web_results)
    print(f"    Conclusion: {quality.get('conclusion', 'N/A')}")

    # @decision: output format
    format_result = choose_output_format(len(documents), len(web_results), f"Query with {len(documents)} docs")
    print(f"    Output format decision: {format_result['chosen']}")

    waxell.score("research_quality", 0.87, comment="Auto-scored based on source coverage")
    waxell.score("factual_grounding", True, data_type="boolean", comment="Multiple sources corroborate findings")

    print(f"  [Synthesizer] Generated answer ({len(answer)} chars)")
    return {"answer": answer, "model": synthesis_response.model}


# ---------------------------------------------------------------------------
# Orchestrator -- coordinates the full research pipeline
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="research-orchestrator", workflow_name="deep-research")
async def run_pipeline(query: str, dry_run: bool = False, waxell_ctx=None):
    """Coordinate the full research pipeline across 2 child agents.

    This is the parent agent. All child agents auto-link to this parent
    via WaxellContext lineage.
    """
    waxell.tag("demo", "research")
    waxell.tag("pipeline", "deep-research")
    waxell.metadata("document_corpus_size", len(DEMO_DOCUMENTS))
    waxell.metadata("mode", "dry-run" if dry_run else "live")

    openai_client = get_openai_client(dry_run=dry_run)

    try:
        # Phase 1: @step -- preprocess the query
        print("[Orchestrator] Phase 1: Preprocessing query (@step)...")
        preprocessed = await preprocess_query(query)
        print(f"  Preprocessed: {preprocessed['token_count']} tokens")

        # Phase 2: @decision -- classify the query
        print("[Orchestrator] Phase 2: Classifying query (@decision + OpenAI)...")
        classification = await classify_query(query=query, openai_client=openai_client)
        chosen = classification.get("chosen", "technical") if isinstance(classification, dict) else str(classification)
        domain = classification.get("domain", "safety") if isinstance(classification, dict) else "safety"
        print(f"  Classification: {chosen} (domain: {domain})")

        # Phase 3: waxell.decide() -- research strategy
        print("[Orchestrator] Phase 3: Research strategy (waxell.decide())...")
        strategy_map = {"technical": "deep_research", "general": "balanced_perspectives", "opinion": "quick_search"}
        strategy = strategy_map.get(chosen, "deep_research")
        waxell.decide(
            "research_strategy",
            chosen=strategy,
            options=["deep_research", "quick_search", "balanced_perspectives"],
            reasoning=f"Query classified as '{chosen}' in domain '{domain}' -- {strategy} optimal",
            confidence=0.92,
        )
        print(f"  Strategy: {strategy}")

        # Phase 4: research-searcher child agent
        print("[Orchestrator] Phase 4: Searching sources (research-searcher)...")
        search_result = await run_searcher(
            query=query,
            openai_client=openai_client,
        )
        print(f"  Found {search_result['num_docs']} docs + {search_result['num_web']} web results "
              f"(avg relevance: {search_result['avg_relevance']})")

        # Phase 5: research-synthesizer child agent
        print("[Orchestrator] Phase 5: Synthesizing answer (research-synthesizer)...")
        synthesis_result = await run_synthesizer(
            query=query,
            documents=search_result["documents"],
            web_results=search_result["web_results"],
            openai_client=openai_client,
        )

        result = {
            "answer": synthesis_result["answer"],
            "documents_used": search_result["num_docs"],
            "web_results_used": search_result["num_web"],
            "avg_relevance": search_result["avg_relevance"],
            "strategy": strategy,
            "pipeline": "research-orchestrator -> research-searcher -> research-synthesizer",
        }
        return result

    except PolicyViolationError as e:
        print(f"\n[Orchestrator] POLICY VIOLATION: {e}")
        print("[Orchestrator] Agent halted by governance policy.")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY
    session = generate_session_id()
    demo_user = pick_demo_user()
    user_id = args.user_id or demo_user["user_id"]
    user_group = args.user_group or demo_user["user_group"]
    observe_active = not is_observe_disabled()

    print("[Research Multi-Agent Pipeline Demo]")
    print(f"  Session:    {session}")
    print(f"  User:       {user_id} ({user_group})")
    print(f"  Query:      {query[:80]}...")
    print(f"  Mode:       {'dry-run' if args.dry_run else 'LIVE (OpenAI)'}")
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
        print(f"[Result]")
        print(f"  Answer:     {result['answer'][:200]}...")
        print(f"  Docs:       {result['documents_used']}")
        print(f"  Web:        {result['web_results_used']}")
        print(f"  Relevance:  {result['avg_relevance']}")
        print(f"  Strategy:   {result['strategy']}")
        print(f"  Pipeline:   {result['pipeline']}")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
