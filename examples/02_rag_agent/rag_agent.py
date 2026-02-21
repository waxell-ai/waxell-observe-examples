#!/usr/bin/env python3
"""Multi-agent RAG pipeline demo.

Demonstrates modern SDK decorator patterns across 2 child agents:

  rag-orchestrator (parent)
  ├── @step: preprocess_query
  ├── @decision: classify_query
  ├── waxell.decide(): retrieval_strategy
  ├── rag-retriever (child)
  │   ├── @tool(tool_type="vector_db"): search_documents
  │   ├── @retrieval(source="rag"): retrieve_and_rank
  │   ├── @step: filter_documents
  │   └── @reasoning: evaluate_retrieval
  └── rag-synthesizer (child)
      ├── @reasoning: assess_quality
      ├── @decision: output_format
      └── LLM synthesis (auto-instrumented)

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

# CRITICAL: init() BEFORE importing openai so auto-instrumentor can patch it
from _common import setup_observe

setup_observe()

# NOW safe to import provider clients (auto-instrumentors have patched them)
import waxell_observe as waxell
from waxell_observe import generate_session_id
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


@waxell.decision(name="classify_query", options=["factual", "analytical", "creative"])
async def classify_query(query: str, openai_client) -> dict:
    """Classify a query to decide the retrieval strategy."""
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the following query as exactly one of: factual, analytical, creative. "
                    'Respond with JSON: {"chosen": "...", "reasoning": "..."}'
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
        return {"chosen": "analytical", "reasoning": content[:200]}


@waxell.decision(name="output_format", options=["brief", "detailed", "bullet_points"])
def choose_output_format(num_docs: int, context: str) -> dict:
    """Choose output format based on document count."""
    format_choice = "detailed" if num_docs > 2 else "brief"
    return {
        "chosen": format_choice,
        "reasoning": f"Query with {num_docs} docs -- {format_choice} format provides better coverage",
        "confidence": 0.85,
    }


# ---------------------------------------------------------------------------
# @tool decorator -- auto-record document search
# ---------------------------------------------------------------------------


@waxell.tool(tool_type="vector_db")
def search_documents(query: str) -> list:
    """Search the document store for relevant documents."""
    return retrieve_documents(query)


# ---------------------------------------------------------------------------
# @retrieval decorator -- auto-record retrieval operations
# ---------------------------------------------------------------------------


@waxell.retrieval(source="rag")
def retrieve_and_rank(query: str, documents: list) -> list[dict]:
    """Rank retrieved documents by relevance."""
    ranked = []
    for i, doc in enumerate(documents):
        score = round(0.95 - (i * 0.08), 2)
        ranked.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": score,
        })
    return ranked


# ---------------------------------------------------------------------------
# @reasoning decorators -- auto-record chain-of-thought
# ---------------------------------------------------------------------------


@waxell.reasoning_dec(step="evaluate_retrieval")
async def evaluate_retrieval(documents: list, query: str) -> dict:
    """Evaluate the quality of retrieved documents."""
    avg_score = sum(d.get("score", 0) for d in documents) / max(len(documents), 1)
    return {
        "thought": f"Retrieved {len(documents)} documents with average score {avg_score:.2f}. "
                   f"Checking coverage against query terms: {query[:60]}...",
        "evidence": [f"Doc '{d['title']}': score={d.get('score', 'N/A')}" for d in documents],
        "conclusion": "Good retrieval quality" if avg_score > 0.7 else "May need expanded search",
    }


@waxell.reasoning_dec(step="quality_assessment")
async def assess_answer_quality(answer: str, documents: list) -> dict:
    """Assess the quality of a generated answer against source documents."""
    doc_titles = [d.get("title", "unknown") for d in documents]
    coverage = len([t for t in doc_titles if t.lower().replace(" ", "") in answer.lower().replace(" ", "")])
    return {
        "thought": f"Generated answer references {coverage}/{len(documents)} source documents. "
                   f"Checking for completeness across: {', '.join(doc_titles)}.",
        "evidence": [f"Source: {t}" for t in doc_titles],
        "conclusion": "Answer adequately covers source material" if coverage > 0 else "Answer may need more grounding",
    }


# ---------------------------------------------------------------------------
# Agent 1: RAG Retriever -- document search, retrieval, and filtering
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="rag-retriever", workflow_name="document-retrieval")
async def run_retriever(query: str, openai_client, policy_triggers: bool = False, waxell_ctx=None):
    """Retrieve and filter documents for the query."""
    waxell.tag("agent_role", "retriever")
    waxell.tag("vector_db", "document_store")
    waxell.metadata("document_corpus_size", len(DEMO_DOCUMENTS))

    # Step 1: Query analysis (auto-instrumented LLM call)
    print("  [Retriever] Analyzing query...")
    analysis_content = LARGE_PROMPT if policy_triggers else query
    analysis_response = await openai_client.chat.completions.create(
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
    print(f"    Analysis: {analysis[:120]}...")

    # Step 2: @tool -- document retrieval
    print("  [Retriever] Searching document store...")
    raw_docs = search_documents(query=query)
    doc_ids = [d["id"] for d in raw_docs]
    print(f"    Found {len(raw_docs)} documents: {doc_ids}")

    # Step 3: @retrieval -- rank results
    print("  [Retriever] Ranking documents...")
    ranked_docs = retrieve_and_rank(query=query, documents=raw_docs)
    for doc in ranked_docs:
        print(f"    [{doc['id']}] {doc['title']} (score: {doc['score']})")

    # Step 4: @step -- filter documents
    @waxell.step_dec(name="filter_documents")
    async def filter_docs(docs: list, openai_cli) -> dict:
        doc_summaries = "\n".join(
            f"- Document {d['id']}: {d['title']} -- {d['content'][:100]}..."
            for d in docs
        )
        filter_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved documents:\n{doc_summaries}\n\n"
            "Select the most relevant documents and explain why."
        )
        filter_response = await openai_cli.chat.completions.create(
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
        return {"selected": len(docs), "filter_reasoning": filter_result[:200]}

    print("  [Retriever] Filtering for relevance...")
    filter_result = await filter_docs(docs=ranked_docs, openai_cli=openai_client)
    print(f"    Filtering: {filter_result['filter_reasoning'][:120]}...")

    # @reasoning -- evaluate retrieval quality
    print("  [Retriever] Evaluating retrieval quality...")
    evaluation = await evaluate_retrieval(documents=ranked_docs, query=query)
    print(f"    Conclusion: {evaluation.get('conclusion', 'N/A')}")

    # Policy trigger: extra tool steps
    if policy_triggers:
        print("  [Retriever] [Policy Trigger] Recording extra tool steps...")

        @waxell.step_dec(name="tool:web_scraper")
        async def web_scraper_step() -> dict:
            return {"url": "https://example.com"}

        @waxell.step_dec(name="tool:data_lookup")
        async def data_lookup_step() -> dict:
            return {"query": "internal"}

        await web_scraper_step()
        print("    Recorded tool:web_scraper (blocked_tools trigger)")
        await data_lookup_step()
        print("    Recorded tool:data_lookup (exceeds max_steps=3)")

    # Policy trigger: slow operation
    if policy_triggers:
        print("  [Retriever] [Policy Trigger] Simulating slow operation (4s > timeout 3s)...")
        await simulate_slow_operation(4.0)
        print("    Slow operation complete (latency trigger)")

    waxell.score("retrieval_quality", 0.88, comment="Good coverage of query terms")

    return {
        "documents": ranked_docs,
        "num_docs": len(ranked_docs),
        "doc_ids": doc_ids,
    }


# ---------------------------------------------------------------------------
# Agent 2: RAG Synthesizer -- answer synthesis + quality assessment
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="rag-synthesizer", workflow_name="answer-synthesis")
async def run_synthesizer(query: str, documents: list, openai_client, waxell_ctx=None):
    """Synthesize a final answer from retrieved documents."""
    waxell.tag("agent_role", "synthesizer")
    waxell.tag("provider", "openai")

    synthesis_context = "\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in documents
    )
    synthesis_prompt = (
        f"Original question: {query}\n\n"
        f"Reference documents:\n{synthesis_context}\n\n"
        "Synthesize a comprehensive answer based on these documents."
    )

    print("  [Synthesizer] Generating answer with OpenAI...")
    synthesis_response = await openai_client.chat.completions.create(
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
    answer = synthesis_response.choices[0].message.content

    # @reasoning -- quality assessment
    print("  [Synthesizer] Assessing answer quality...")
    quality = await assess_answer_quality(answer=answer, documents=documents)
    print(f"    Conclusion: {quality.get('conclusion', 'N/A')}")

    # @decision: output format
    format_result = choose_output_format(len(documents), f"Query with {len(documents)} docs")
    print(f"    Output format decision: {format_result['chosen']}")

    waxell.score("answer_quality", 0.91, comment="Good synthesis from retrieved documents")
    waxell.score("factual_grounding", True, data_type="boolean", comment="Answer grounded in source documents")

    print(f"  [Synthesizer] Generated answer ({len(answer)} chars)")
    return {"answer": answer, "model": synthesis_response.model, "answer_length": len(answer)}


# ---------------------------------------------------------------------------
# Orchestrator -- coordinates the full RAG pipeline
# ---------------------------------------------------------------------------


@waxell.observe(agent_name="rag-orchestrator", workflow_name="document-qa")
async def run_pipeline(query: str, dry_run: bool = False, policy_triggers: bool = False, waxell_ctx=None):
    """Coordinate the full RAG pipeline across 2 child agents.

    This is the parent agent. All child agents auto-link to this parent
    via WaxellContext lineage.
    """
    waxell.tag("demo", "rag")
    waxell.tag("query_type", "multi-step")
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
        chosen = classification.get("chosen", "analytical") if isinstance(classification, dict) else str(classification)
        print(f"  Classification: {chosen}")

        # Phase 3: waxell.decide() -- retrieval strategy
        print("[Orchestrator] Phase 3: Routing decision (waxell.decide())...")
        strategy_map = {"factual": "keyword_search", "analytical": "semantic_search", "creative": "hybrid_search"}
        strategy = strategy_map.get(chosen, "semantic_search")
        waxell.decide(
            "retrieval_strategy",
            chosen=strategy,
            options=["semantic_search", "keyword_search", "hybrid_search"],
            reasoning=f"Query classified as '{chosen}' -- {strategy} optimal for this type",
            confidence=0.88,
        )
        print(f"  Strategy: {strategy}")

        # Phase 4: rag-retriever child agent
        print("[Orchestrator] Phase 4: Retrieving documents (rag-retriever)...")
        retrieval_result = await run_retriever(
            query=query,
            openai_client=openai_client,
            policy_triggers=policy_triggers,
        )
        print(f"  Retrieved {retrieval_result['num_docs']} documents")

        # Phase 5: rag-synthesizer child agent
        print("[Orchestrator] Phase 5: Synthesizing answer (rag-synthesizer)...")
        synthesis_result = await run_synthesizer(
            query=query,
            documents=retrieval_result["documents"],
            openai_client=openai_client,
        )

        answer = synthesis_result["answer"]
        return {
            "answer": answer,
            "documents_used": retrieval_result["num_docs"],
            "steps": 5,
            "pipeline": "rag-orchestrator -> rag-retriever -> rag-synthesizer",
        }

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
    policy_triggers = args.policy_triggers

    if policy_triggers:
        print("[RAG Demo] POLICY TRIGGER MODE -- intentionally crossing policy thresholds")
        print("[RAG Demo] Expected triggers: budget (large prompt), safety (many steps), latency (slow operation)")
        print()

    print("[RAG Multi-Agent Pipeline Demo]")
    print(f"  Session:    {session}")
    print(f"  User:       {user_id} ({user_group})")
    print(f"  Query:      {query}")
    print(f"  Mode:       {'dry-run' if args.dry_run else 'LIVE (OpenAI)'}")
    print()

    result = await run_pipeline(
        query=query,
        dry_run=args.dry_run,
        policy_triggers=policy_triggers,
        session_id=session,
        user_id=user_id,
        user_group=user_group,
        enforce_policy=observe_active,
        mid_execution_governance=observe_active,
        client=get_observe_client(),
    )

    if "error" not in result:
        print()
        answer_display = result["answer"][:200] + ("..." if len(result["answer"]) > 200 else "")
        print(f"[Result]")
        print(f"  Answer:     {answer_display}")
        print(f"  Docs:       {result['documents_used']}")
        print(f"  Steps:      {result['steps']}")
        print(f"  Pipeline:   {result['pipeline']}")

    waxell.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
