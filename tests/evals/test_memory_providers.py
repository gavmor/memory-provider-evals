"""Multi-turn end-to-end evals for Hermes memory providers.

Run with DeepEval, not raw pytest. Bare ``pytest`` deselects this file via the
``evals`` marker (see pyproject), because it needs a live model and a
provisioned memory backend::

    deepeval test run tests/evals/test_memory_providers.py -m evals \\
      --identifier "iterating-on-memory-recall-round-1" \\
      --num-processes 5 --ignore-errors --skip-on-missing-params

Each golden is a scripted multi-session scenario. The suite replays it through
the real app (traced-harness ``SessionRunner`` -> memory adapter -> Agno agent,
with an inter-session consolidation pass), reads back the enriched JSONL trace,
and asserts the resulting ``ConversationalTestCase`` against the conversational
metric suite in ``metrics.py``.

Why ``.dataset.json`` is committed rather than generated
--------------------------------------------------------
``deepeval generate --variation multi-turn`` emits ``ConversationalGolden``
objects carrying only ``scenario``/``expected_outcome``/``context`` — never
``turns`` or ``additional_metadata``. That is by design: a generated
``scenario`` is a seed for ``ConversationSimulator``, which invents the user
turns at eval time.

These evals deliberately do NOT use the simulator: a memory benchmark depends
on a *scripted* belief-revision sequence (ingress fact -> superseding fact ->
probe), and simulated turns would destroy the contradiction under test. The
scripted turns and the per-scenario expectations therefore live in the
committed dataset, which is plain data — no generation pipeline to maintain.

Selecting what runs:

* ``MEMORY_PROVIDER``        — cashew (default) | chronicle | memex8 | nachos
* ``MEMORY_EVAL_WORKSPACE``  — where provider stores and traces are written
* ``MEMORY_METRICS_OFFLINE`` — set to 1 to assert only the deterministic
  metrics (no judge model, no network)
"""

import asyncio
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.dataset import ConversationalGolden, EvaluationDataset

from memory_provider_evals.adapters import build_adapter, default_provider
from memory_provider_evals.bridge import run_memory_scenario

# Deselected by bare `pytest` (addopts = -m "not evals"); run via deepeval.
pytestmark = pytest.mark.evals

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / ".dataset.json"

dataset = EvaluationDataset()
if DATASET_PATH.is_file():
    dataset.add_goldens_from_json_file(file_path=str(DATASET_PATH))

if not dataset.goldens:
    pytest.skip(
        f"No goldens found at {DATASET_PATH}. This dataset is committed, not "
        "generated — restore it from version control.",
        allow_module_level=True,
    )

OFFLINE = os.environ.get("MEMORY_METRICS_OFFLINE") == "1"

# These evals drive a live model and a live memory backend. Skip cleanly (rather
# than erroring at import) when credentials are absent, so an explicit run on an
# unconfigured machine reports an honest skip instead of an import traceback.
if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    pytest.skip(
        "GOOGLE_API_KEY / GEMINI_API_KEY unset — the memory evals need a live "
        "agent model, and the judge metrics need a Gemini judge. Configure "
        "with `deepeval set-gemini` and export the key.",
        allow_module_level=True,
    )

if OFFLINE:
    from metrics import DETERMINISTIC_METRICS as METRICS
else:
    from metrics import MULTI_TURN_METRICS as METRICS


def _workspace(golden: ConversationalGolden) -> Path:
    root = Path(os.environ.get("MEMORY_EVAL_WORKSPACE", EVAL_DIR / ".runs"))
    return root / (getattr(golden, "name", None) or "scenario")


@pytest.mark.parametrize(
    "golden", dataset.goldens, ids=lambda g: g.name or "golden"
)
def test_memory_provider(golden: ConversationalGolden):
    provider = default_provider()
    adapter = build_adapter(provider)

    from traced_harness.agent import create_agent
    from traced_harness.session_runner import make_agno_turn_executor

    async def _run():
        agent = await create_agent(memory_adapter=adapter)
        return await run_memory_scenario(
            golden=golden,
            adapter=adapter,
            turn_executor=make_agno_turn_executor(agent),
            workspace_dir=_workspace(golden),
        )

    test_case = asyncio.run(_run())
    assert_test(test_case=test_case, metrics=METRICS)
