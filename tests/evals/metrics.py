"""Metric instances for the memory-provider eval suite.

Per the DeepEval skill, metric *instances* live here so the eval file stays
focused on app execution. The custom metric *classes* are library code in
``memory_provider_evals.metrics``.

Multi-turn rule
---------------
These evals assert on ``ConversationalTestCase``, so every metric here is a
multi-turn conversational metric. Single-turn ``LLMTestCase`` metrics
(``AnswerRelevancyMetric``, ``FaithfulnessMetric``, ...) are invalid on this
suite and must not be added.

Five metrics, split between LLM-judge and deterministic:

| Metric                  | Kind                | Measures                       |
| ----------------------- | ------------------- | ------------------------------ |
| Temporal Invalidation   | ConversationalGEval | superseded facts never resurface |
| Contradiction Rejection | ConversationalGEval | final answer + its context are clean |
| Memory Recall F1        | deterministic       | token-F1 of the recalled fact  |
| Memory Token Overhead   | deterministic       | context cost of injection      |
| Multi-Hop Retrieval     | deterministic       | entities joined across >= 2 sessions |

Lazy judge model
----------------
``GeminiModel()`` raises at construction when no API key is set, so the two
``ConversationalGEval`` instances -- and ``MULTI_TURN_METRICS`` -- are built on
first access via module ``__getattr__``. That keeps ``DETERMINISTIC_METRICS``
importable (and the ``MEMORY_METRICS_OFFLINE=1`` path usable) on a machine with
no credentials, while still failing loudly if you ask for the judge metrics
without a key.
"""

import os
from typing import Any

from deepeval.metrics import ConversationalGEval
from deepeval.test_case import MultiTurnParams

from memory_provider_evals.metrics import (
    MemoryRecallF1Metric,
    MemoryTokenOverheadMetric,
    MultiHopRetrievalMetric,
)

# Params the judges may read. ConversationalGEval renders only a fixed set of
# MultiTurnParams — CONTEXT is NOT among them (it raises KeyError at scoring
# time), while METADATA is. The expectations therefore reach the judge through
# ConversationalTestCase.metadata["expectations"], which
# memory_provider_evals.bridge populates from the golden.
_JUDGE_PARAMS = [
    MultiTurnParams.ROLE,
    MultiTurnParams.CONTENT,
    MultiTurnParams.RETRIEVAL_CONTEXT,
    MultiTurnParams.METADATA,
    MultiTurnParams.SCENARIO,
    MultiTurnParams.EXPECTED_OUTCOME,
]

_TEMPORAL_INVALIDATION_STEPS = [
    (
        "Read METADATA['expectations'] to identify `superseded_facts` and "
        "`current_fact`."
    ),
    (
        "Scan every assistant turn and every retrieval_context passage "
        "across the whole conversation."
    ),
    (
        "Penalise heavily if any superseded fact is presented as currently "
        "true, either by the assistant or by a retrieved passage offered "
        "as present-tense context."
    ),
    (
        "Do NOT penalise a superseded fact that is explicitly framed as "
        "historical, corrected, or outdated — correctly dated history is "
        "the desired behaviour, not a failure."
    ),
    (
        "Award a high score only when the memory system has fully "
        "invalidated the stale belief while preserving the current one."
    ),
]

_CONTRADICTION_REJECTION_STEPS = [
    (
        "Read METADATA['expectations'] to identify `superseded_facts` and "
        "`current_fact`."
    ),
    (
        "Look only at the FINAL assistant turn and the retrieval_context "
        "that supported it."
    ),
    (
        "Penalise if the final answer asserts a superseded fact, hedges "
        "between the superseded and current fact, or if its supporting "
        "retrieval_context surfaced the superseded fact as current."
    ),
    "Penalise if the final answer contradicts the EXPECTED_OUTCOME.",
    (
        "Award a high score only when the final answer commits "
        "unambiguously to the current fact on clean supporting context."
    ),
]


# Deterministic metrics need no judge model, so they are built eagerly.
memory_recall_f1 = MemoryRecallF1Metric(threshold=0.5)
memory_token_overhead = MemoryTokenOverheadMetric(threshold=0.8)
multi_hop_retrieval = MultiHopRetrievalMetric(threshold=1.0)

#: Deterministic subset — runs with no judge model and no network.
#: CI smoke runs: `MEMORY_METRICS_OFFLINE=1 deepeval test run ...`
DETERMINISTIC_METRICS = [
    memory_recall_f1,
    memory_token_overhead,
    multi_hop_retrieval,
]


def judge_model():
    """Gemini judge, reusing the GOOGLE_API_KEY wired for the agent under test.

    Override the model id with ``DEEPEVAL_GEMINI_MODEL``.
    """
    from deepeval.models import GeminiModel

    model_id = os.environ.get("DEEPEVAL_GEMINI_MODEL")
    return GeminiModel(model=model_id) if model_id else GeminiModel()


_LAZY: dict[str, Any] = {}


def _build_judge_metrics() -> dict[str, Any]:
    model = judge_model()
    temporal_invalidation = ConversationalGEval(
        name="Temporal Invalidation",
        evaluation_params=_JUDGE_PARAMS,
        evaluation_steps=_TEMPORAL_INVALIDATION_STEPS,
        model=model,
        threshold=0.7,
    )
    contradiction_rejection = ConversationalGEval(
        name="Contradiction Rejection",
        evaluation_params=_JUDGE_PARAMS,
        evaluation_steps=_CONTRADICTION_REJECTION_STEPS,
        model=model,
        threshold=0.7,
    )
    return {
        "JUDGE_MODEL": model,
        "temporal_invalidation": temporal_invalidation,
        "contradiction_rejection": contradiction_rejection,
        "MULTI_TURN_METRICS": [
            temporal_invalidation,
            contradiction_rejection,
            *DETERMINISTIC_METRICS,
        ],
    }


_LAZY_NAMES = frozenset(
    {
        "JUDGE_MODEL",
        "temporal_invalidation",
        "contradiction_rejection",
        "MULTI_TURN_METRICS",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        if not _LAZY:
            _LAZY.update(_build_judge_metrics())
        return _LAZY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
