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
  adapters.py   # the four providers under study
  trace.py      # TraceRecord + MemoryEvalSuite judges (metric internals)
  metrics.py    # deterministic BaseConversationalMetric classes
  bridge.py     # golden -> Scenario -> run -> ConversationalTestCase
  ingest.py     # harness trace -> DeepEval LLMTestCase (single-turn)
tests/
  evals/
    metrics.py           # metric instances (judge model built lazily)
    .dataset.json        # committed scripted scenarios
    test_memory_providers.py
```

## What comes from the harness

`traced-harness` treats memory as a first-class peripheral alongside MCP and
skills, so this repo does **not** redefine it. From `traced_harness.memory`:

- `MemoryProviderAdapter` — the lifecycle ABC our four adapters subclass
- `MemoryToolContract` — tools, context hooks, system-prompt contract
- `retrieval_span`, `record_memory_injection`, `consolidation_span` — telemetry
- `register_memory_tools`, `build_memory_instructions` — agent prompt wiring
- `make_memory_session_runner` — a `SessionRunner` pre-wired with the memory
  metadata key and span name

The harness *core* stays domain-blind: `SessionRunner` depends only on a
structural `PeripheralLifecycle` protocol and never imports the memory module.
