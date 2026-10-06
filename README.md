# memory-provider-evals

A DeepEval suite benchmarking Hermes durable-memory providers, layered on the
[`traced-harness`](../traced-harness) instrument.

**Separation of concerns:** `traced-harness` is the instrument — it drives the
agent, records OpenTelemetry spans, and writes JSONL traces. It knows nothing
about DeepEval. This repo is the *study*: it defines what "good memory" means,
supplies the providers under test, and scores them.

## What's measured

Five multi-turn conversational metrics, split between LLM-judge and
deterministic:

| Metric | Kind | Measures |
| --- | --- | --- |
| Temporal Invalidation | `ConversationalGEval` | superseded facts never resurface anywhere |
| Contradiction Rejection | `ConversationalGEval` | final answer + its supporting context are clean |
| Memory Recall F1 | deterministic | token-F1 of the recalled fact |
| Memory Token Overhead | deterministic | context cost of memory injection |
| Multi-Hop Retrieval | deterministic | entities joined across ≥2 sessions |

The deterministic metrics subclass `BaseConversationalMetric`, so they are
first-class DeepEval metrics (scored, reported, and pushed to Confident AI like
any built-in) while needing no judge model or network.

## Providers under test

`CashewAdapter`, `ChronicleAdapter`, `Memex8Adapter`, `NachosAdapter` — each
implementing the `MemoryPluginAdapter` lifecycle contract from the harness.
None of these backends are provisioned in CI; every adapter supports
`dry_run=True`, which records the exact command/URL/config it *would* execute.

## Setup

```bash
uv sync                      # pulls traced-harness from GitHub (pinned in uv.lock)
deepeval set-gemini --model <model-id>
export GOOGLE_API_KEY=...    # the judge and the agent under test both use this
```

`traced-harness` isn't on PyPI, so it's consumed as a git dependency:

```toml
[tool.uv.sources]
traced-harness = { git = "https://github.com/gavmor/traced-harness", branch = "main" }
```

`uv.lock` pins the exact resolved commit, so installs are reproducible. Pick up
harness changes with `uv lock --upgrade-package traced-harness`.

## Running

```bash
# Unit suite (no model, no backend) — the eval suite is deselected by marker
pytest

# The eval suite proper
deepeval test run tests/evals/test_memory_providers.py -m evals \
  --identifier "iterating-on-memory-recall-round-1" \
  --num-processes 5 --ignore-errors --skip-on-missing-params

# Deterministic metrics only (no judge model, no network)
MEMORY_METRICS_OFFLINE=1 deepeval test run tests/evals/test_memory_providers.py -m evals
```

Environment:

| Variable | Effect |
| --- | --- |
| `MEMORY_PROVIDER` | `cashew` (default) \| `chronicle` \| `memex8` \| `nachos` |
| `MEMORY_EVAL_WORKSPACE` | where provider stores and traces are written |
| `MEMORY_METRICS_OFFLINE` | `1` → assert only the deterministic metrics |
| `DEEPEVAL_GEMINI_MODEL` | override the judge model id |

## Why the dataset is committed, not generated

`deepeval generate --variation multi-turn` emits `ConversationalGolden` objects
carrying only `scenario`/`expected_outcome`/`context` — never `turns` or
`additional_metadata`. That is by design: a generated `scenario` seeds
`ConversationSimulator`, which invents the user turns at eval time.

These evals deliberately do **not** use the simulator. A memory benchmark
depends on a *scripted* belief-revision sequence (ingress fact → superseding
fact → probe); simulated turns would destroy the contradiction under test. So
the scripted turns and per-scenario expectations live in
`tests/evals/.dataset.json`, which is plain data — no generation pipeline to
maintain.

## Layout

```
src/memory_provider_evals/
  adapters.py     # the four providers under study
  memory_store.py # what answers a memory tool: a real lexical SQLite store
                  # (Nachos) or an explicit "not provisioned" failure
  chronicle_backend.py  # the real Chronicle engine, off a checkout
  mcp_server.py   # contract -> live MCP server exposing the provider's tools
  live.py         # the wired run: server -> client -> create_agent(client=...)
  trace.py        # TraceRecord + MemoryEvalSuite judges (metric internals)
  metrics.py      # deterministic BaseConversationalMetric classes
  benchmark.py    # comparative memorybench on BenchKit's measurement layer
  bridge.py       # golden -> Scenario -> run -> ConversationalTestCase
  ingest.py       # harness trace -> DeepEval LLMTestCase (single-turn)
tests/
  evals/
    metrics.py           # metric instances (judge model built lazily)
    .dataset.json        # committed scripted scenarios
    test_memory_providers.py
```

## Running a live benchmark

```
export GEMINI_API_KEY=...
uv run memorybench --provider nachos
uv run memorybench --provider chronicle
```

Two providers run end to end here. `nachos` is text-only and local-first, which
is what `LexicalMemoryStore` implements, so it needs no backend at all.
`chronicle` needs a checkout and nothing else — it is a stdlib-only Hermes
plugin, not a service:

```
git clone https://github.com/indigokarasu/chronicle-agent-context-and-memory.git vendor/chronicle
```

`vendor/` is gitignored; `$CHRONICLE_REPO` points at a checkout anywhere, and
one installed with `hermes plugins install
indigokarasu/chronicle-agent-context-and-memory` is found automatically.
`cashew` and `memex8` still raise `BackendNotProvisioned`, naming the
provisioning step, until their service is running; that lands in the
benchmark's `error` column rather than scoring as a provider that remembered
nothing.

`AGENT_MODEL_NAME` selects the agent model (default `gemini-flash-lite-latest`;
the harness default `gemini-3.1-flash-lite-preview` returns 503s). The free
Gemini tier allows 15 requests/minute, which a multi-scenario run can exceed.

### Chronicle specifics

Chronicle is the one provider here that does **not** ask the agent to remember.
`CaptureEngine.observe` appends every turn to an event log and extracts beliefs
from it, so its contract exposes one read tool (`chronicle_search`) and no
write tool; `live.py` drives capture after each turn, where Hermes calls
`sync_turn`. Consolidation is its curation queue plus maintenance scheduler —
`ChronicleCore.tick`'s two halves — run in process between sessions. The
`scripts/*.py` earlier specs named are a LongMemEval parameter sweep, a
session-exclusion vector prune and an off-box re-embedding repair; none of them
is a consolidation pass. See `ChronicleAdapter`'s docstring.

Embeddings default to upstream's offline `hashing` embedder. Chronicle's own
default (`auto`) opens TCP connections to LM Studio / Ollama / llama.cpp ports
inside the core constructor, which would bind a benchmark number to whatever
happens to be listening on the machine. The cost is real and belongs with any
result: feature-hashed vectors are a weaker semantic tier than a real embedding
model, so Chronicle's vector channel is measured at its offline floor.
`$CHRONICLE_EMBED_MODEL` opts back in.

## Why the agent needs an MCP server

`create_agent` registers a provider's tool *names* and injects its
system-prompt contract, but the implementations arrive over the MCP `client`.
Called without one, the agent is told it has `cashew_query` and then cannot
call it — so it answers from the context window, which the multi-session
design deliberately empties, and every provider scores identically badly for a
reason unrelated to its memory. `mcp_server.py` + `live.py` close that loop:
one server per provider contract, connected in process, passed as `client=`.

## What comes from the harness

`traced-harness` treats memory as a first-class peripheral alongside MCP and
skills, so this repo does **not** redefine it. From `traced_harness.memory`:

- `MemoryProviderAdapter` — the lifecycle ABC our four adapters subclass
- `MemoryToolContract` — tools, retrieval tools, context hooks, prompt contract
- `retrieval_span`, `record_memory_injection`, `consolidation_span` — telemetry
- `register_memory_tools`, `build_memory_instructions` — agent prompt wiring
- `make_memory_session_runner` — a `SessionRunner` pre-wired with the memory
  metadata key and span name
- `make_memory_tool_hook`, `make_memory_turn_executor` — per-turn memory
  telemetry: a timed span around each real recall call, and the turn's
  injection overhead

The harness *core* stays domain-blind: `SessionRunner` depends only on a
structural `PeripheralLifecycle` protocol and never imports the memory module.
