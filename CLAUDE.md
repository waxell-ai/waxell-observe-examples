# CLAUDE.md

Instructions for Claude Code and other coding agents working with this repository.

## Overview

16 standalone examples for [waxell-observe](https://github.com/waxell-ai/waxell-observe) — lightweight observability & governance for AI agents.

## Project Structure

```
waxell-observe-examples/
├── examples/
│   ├── _common.py                    # Shared: mock clients, helpers, observe setup
│   ├── 01_anthropic_agent/           # Each directory = one example
│   ├── ...
│   └── 16_basic_quickstart/          # START HERE
├── .env.example                      # All env vars
├── requirements.txt                  # pip install -r requirements.txt
└── pyproject.toml                    # Project metadata
```

## Running Examples

```bash
# From repo root — all examples support --dry-run
python examples/16_basic_quickstart/quickstart.py --dry-run
python examples/01_anthropic_agent/anthropic_agent.py --dry-run
python examples/02_rag_agent/rag_agent.py --dry-run --query "Your question"
```

## Standard Pattern

Every example follows this structure:

```python
# 1. Path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 2. Initialize BEFORE importing LLM SDKs
from _common import setup_observe
setup_observe()

# 3. Import waxell-observe and helpers
from waxell_observe import WaxellContext, generate_session_id
from _common import get_openai_client, get_observe_client, parse_args

# 4. Agent logic inside WaxellContext
async with WaxellContext(agent_name="my-agent", ...) as ctx:
    response = await client.chat.completions.create(...)
    ctx.record_step("step_name", output={...})
    ctx.record_llm_call(model=..., tokens_in=..., tokens_out=..., task=...)
    ctx.set_result({...})

# 5. Shutdown
waxell_observe.shutdown()
```

## Key Utilities in `_common.py`

| Function | Purpose |
|----------|---------|
| `setup_observe()` | Initialize waxell-observe (call BEFORE SDK imports) |
| `get_observe_client()` | Get configured WaxellObserveClient |
| `get_openai_client(dry_run)` | Mock or real OpenAI client |
| `get_anthropic_client(dry_run)` | Mock or real Anthropic client |
| `get_groq_client(dry_run)` | Mock or real Groq client |
| `get_streaming_openai_client(dry_run)` | Streaming OpenAI client |
| `get_streaming_anthropic_client(dry_run)` | Streaming Anthropic client |
| `parse_args()` | Standard CLI args (--dry-run, --query, --policy-triggers) |
| `pick_demo_user()` | Random demo user identity |
| `retrieve_documents(query)` | Keyword-based doc retrieval from demo corpus |

## Integration Patterns

**Auto-instrumentation** (examples: 01, 02, 05, 12, 13):
```python
setup_observe()  # Patches OpenAI/Anthropic/Groq SDKs automatically
```

**@waxell_agent decorator** (examples: 03, 10):
```python
@waxell_agent(agent_name="my-agent")
async def run(query: str, waxell_ctx=None):
    waxell_ctx.record_step(...)
```

**WaxellContext context manager** (examples: 01, 02, 04, 06-09, 16):
```python
async with WaxellContext(agent_name="my-agent", ...) as ctx:
    ctx.record_step(...)
```

**LangChain callback** (example: 15):
```python
handler = WaxellLangChainHandler(agent_name="my-agent", ...)
chain.invoke(input, config={"callbacks": [handler]})
handler.flush_sync(result={...})
```

## Documentation

- Full docs: https://waxell.ai/docs/observe/
- Quickstart: https://waxell.ai/docs/observe/quickstart
- Auto-instrumentation: https://waxell.ai/docs/observe/integrations/auto-instrumentation
- Context Manager: https://waxell.ai/docs/observe/integrations/context-manager
- Governance: https://waxell.ai/docs/observe/features/governance
- API Reference: https://waxell.ai/docs/observe/api/python-sdk

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAXELL_API_KEY` | (none) | Platform API key (`wax_sk_...`) |
| `WAXELL_API_URL` | `https://api.waxell.dev` | Platform API URL |
| `WAXELL_OBSERVE` | `true` | Set `false` to disable all telemetry |
| `WAXELL_DEBUG` | `false` | Console span export |
| `WAXELL_CAPTURE_CONTENT` | `false` | Include prompt/response in traces |

## Troubleshooting

**`ModuleNotFoundError: No module named '_common'`**
Run from the repo root, or check the sys.path setup at the top of the file.

**Auto-instrumentation not working**
`setup_observe()` must be called BEFORE importing `openai`/`anthropic`/`groq`.

**Telemetry not appearing**
Check: `WAXELL_API_KEY` is set, `WAXELL_OBSERVE` is not `false`, `waxell_observe.shutdown()` is called.
