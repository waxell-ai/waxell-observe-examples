# Waxell Observe Examples

Production-ready examples for [waxell-observe](https://github.com/waxell-ai/waxell-observe) — lightweight observability & governance for AI agents.

Every example works out of the box with `--dry-run` (no API keys needed).

## Quick Start

```bash
# 1. Clone
git clone https://github.com/waxell-ai/waxell-observe-examples.git
cd waxell-observe-examples

# 2. Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Run (no API keys needed)
python examples/16_basic_quickstart/quickstart.py --dry-run
```

That's it. You'll see mock LLM responses and the full agent pipeline running locally.

To send traces to the Waxell platform, copy `.env.example` to `.env` and add your `WAXELL_API_KEY`.

---

## Examples

### Start Here

| # | Example | What It Shows |
|---|---------|--------------|
| 16 | [**Basic Quickstart**](examples/16_basic_quickstart/) | The 5-step pattern every example follows. Start here. |

### Core Patterns

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 01 | [Anthropic Agent](examples/01_anthropic_agent/) | 3-step content analysis (classify, extract, summarize) | Anthropic |
| 02 | [RAG Agent](examples/02_rag_agent/) | Document retrieval, relevance filtering, answer synthesis | OpenAI |
| 03 | [Multi-Agent](examples/03_multi_agent/) | Coordinator + 3 sub-agents with `@waxell_agent` | OpenAI |

### Integrations

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 05 | [Streaming](examples/05_streaming_agent/) | OpenAI + Anthropic streaming side-by-side | OpenAI, Anthropic |
| 11 | [LiteLLM](examples/11_litellm_agent/) | 3 providers through LiteLLM's unified API | LiteLLM |
| 12 | [All Providers](examples/12_all_providers_agent/) | OpenAI + Anthropic + Groq in one trace | All |
| 13 | [Groq & Function Calling](examples/13_groq_agent/) | Groq Llama + OpenAI tool calls | Groq, OpenAI |
| 15 | [LangChain](examples/15_langchain_agent/) | `WaxellLangChainHandler` callback | LangChain |

### Governance & Security

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 04 | [Governance](examples/04_governance_agent/) | Policy checks, `check_policy()`, retry feedback, events | OpenAI |
| 06 | [Prompt Guard](examples/06_prompt_guard_agent/) | PII/credential/injection detection (block/warn/redact) | OpenAI |
| 14 | [Prompt Management](examples/14_prompt_management_agent/) | Prompt versioning, background collector, `get_context()` | OpenAI |

### Advanced Features

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 10 | [Enrichment](examples/10_enrichment_agent/) | `@observe`, scores, tags, metadata | OpenAI |

### Real-World Pipelines

| # | Example | What It Shows | Provider |
|---|---------|--------------|----------|
| 07 | [Customer Support](examples/07_support_agent/) | Intent classify, tool calls, decision tree, retry | OpenAI |
| 08 | [Research](examples/08_research_agent/) | Retrieval, web search, reasoning chain | OpenAI |
| 09 | [Code Review](examples/09_code_review_agent/) | Diff parsing, static analysis, review generation | Anthropic |

---

## How It Works

Every example follows the same pattern:

```python
from _common import setup_observe

# 1. Initialize BEFORE importing LLM SDKs (enables auto-instrumentation)
setup_observe()

from waxell_observe import WaxellContext

# 2. Wrap your agent logic
async with WaxellContext(agent_name="my-agent", ...) as ctx:

    # 3. Make LLM calls — they're automatically traced
    response = await client.chat.completions.create(...)

    # 4. Record steps and results
    ctx.record_step("classify", output={"category": "tech"})
    ctx.record_llm_call(model="gpt-4o-mini", tokens_in=50, tokens_out=120, task="classify")
    ctx.set_result({"answer": "..."})

# 5. Flush
waxell_observe.shutdown()
```

`setup_observe()` patches the OpenAI/Anthropic/Groq SDKs so every LLM call is captured automatically. `WaxellContext` groups everything into a single trace.

---

## Dry-Run Mode

Every example supports `--dry-run`:

```bash
python examples/02_rag_agent/rag_agent.py --dry-run
python examples/04_governance_agent/governance_agent.py --dry-run --policy-triggers
python examples/07_support_agent/support_agent.py --dry-run --query "Where is my order?"
```

Mock clients return realistic responses. No API keys, no costs, no network calls.

---

## Environment Variables

Copy `.env.example` to `.env`. All variables are optional for dry-run mode.

| Variable | Required For | Description |
|----------|-------------|-------------|
| `WAXELL_API_KEY` | Sending traces | Platform API key ([get one here](https://app.waxell.dev/settings/api-keys)) |
| `WAXELL_API_URL` | Custom endpoint | Default: `https://api.waxell.dev` |
| `OPENAI_API_KEY` | Real OpenAI calls | |
| `ANTHROPIC_API_KEY` | Real Anthropic calls | |
| `GROQ_API_KEY` | Real Groq calls | |
| `WAXELL_OBSERVE` | Kill switch | Set `false` to disable all telemetry |
| `WAXELL_DEBUG` | Debugging | Console span export |
| `WAXELL_CAPTURE_CONTENT` | Content capture | Include prompts/responses in traces (default: false) |

---

## Installation

```bash
# Recommended: core + OpenTelemetry
pip install "waxell-observe[otel] @ git+https://github.com/waxell-ai/waxell-observe.git"

# Everything (all provider instrumentors)
pip install "waxell-observe[all] @ git+https://github.com/waxell-ai/waxell-observe.git"
```

---

## Documentation

| Resource | Link |
|----------|------|
| Quickstart | [waxell.ai/docs/observe/quickstart](https://waxell.ai/docs/observe/quickstart) |
| Installation | [waxell.ai/docs/observe/installation](https://waxell.ai/docs/observe/installation) |
| Auto-instrumentation | [waxell.ai/docs/observe/integrations/auto-instrumentation](https://waxell.ai/docs/observe/integrations/auto-instrumentation) |
| Context Manager | [waxell.ai/docs/observe/integrations/context-manager](https://waxell.ai/docs/observe/integrations/context-manager) |
| Governance | [waxell.ai/docs/observe/features/governance](https://waxell.ai/docs/observe/features/governance) |
| API Reference | [waxell.ai/docs/observe/api/python-sdk](https://waxell.ai/docs/observe/api/python-sdk) |
| **For coding agents** | [CLAUDE.md](CLAUDE.md) |
