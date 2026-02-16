# Basic Quickstart

**Start here.** The simplest possible waxell-observe example.

Initialize the SDK, make one LLM call, record the result. That's it.

## Quick Start

```bash
# Dry-run (no API keys needed)
python examples/16_basic_quickstart/quickstart.py --dry-run

# With real API calls
python examples/16_basic_quickstart/quickstart.py
```

## The 5 Steps

```python
# 1. Initialize BEFORE importing LLM SDKs
from _common import setup_observe
setup_observe()

# 2. Wrap your agent in a WaxellContext
async with WaxellContext(agent_name="my-agent", ...) as ctx:

    # 3. Make LLM calls as usual (auto-traced)
    response = await client.chat.completions.create(...)

    # 4. Record what happened
    ctx.record_step("answer", output={...})
    ctx.record_llm_call(model=..., tokens_in=..., tokens_out=..., task=...)
    ctx.set_result({...})

# 5. Flush telemetry
waxell_observe.shutdown()
```

## What to Look For

After running, check your [Waxell dashboard](https://app.waxell.dev) for the trace.
You'll see the LLM call, token counts, latency, and the recorded result.

## Learn More

- [waxell-observe Quickstart](https://waxell.ai/docs/observe/quickstart)
- [Installation](https://waxell.ai/docs/observe/installation)
- [Back to all examples](../../README.md)
