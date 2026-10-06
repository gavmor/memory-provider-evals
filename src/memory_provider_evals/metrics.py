"""Deterministic DeepEval conversational metrics for memory-provider evals.

These are the *non-LLM* half of the memory metric suite. Each subclasses
:class:`deepeval.metrics.BaseConversationalMetric` so it is a first-class
DeepEval metric -- usable in ``assert_test(test_case=..., metrics=[...])``,
reported by ``deepeval test run``, and pushed to Confident AI like any built-in
metric. The scoring bodies reuse the judges in
:mod:`memory_provider_evals.trace` so trace-level analysis and eval-suite
scoring can never drift apart.

The semantic criteria (temporal invalidation, contradiction rejection) are
``ConversationalGEval`` instances instead and live in ``tests/evals/metrics.py``.

Test-case contract
------------------
All three metrics read :attr:`ConversationalTestCase.metadata`, which
:mod:`memory_provider_evals.bridge` populates as::

    {
      "memory": {
        "provider": "cashew",
        "injected_tokens": 40,
        "generated_tokens": 120,
        "token_overhead_ratio": 0.33,
        "retrievals": [{"query": ..., "passages": [...], "session_id": "s2"}],
      },
      "expectations": {              # mirrored from the golden
        "current_fact": "Tesla Model 3",
        "superseded_facts": ["Honda Civic"],
        "multi_hop_entities": ["Tesla", "Berlin"],
        "max_token_overhead_ratio": 0.5,
      },
    }

A metric whose required expectation is absent raises
:class:`deepeval.errors.MissingTestCaseParamsError`, so
``deepeval test run --skip-on-missing-params`` skips it rather than failing the
whole run -- the idiomatic DeepEval behaviour for partially-specified datasets.
"""

from __future__ import annotations

from typing import Any

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import BaseConversationalMetric
from deepeval.metrics.utils import check_conversational_test_case_params
from deepeval.test_case import ConversationalTestCase, MultiTurnParams

from memory_provider_evals.trace import (
    connects_entities_across_sessions,
    token_f1,
)

__all__ = [
    "EXPECTATIONS_METADATA_KEY",
    "MEMORY_METADATA_KEY",
    "MemoryRecallF1Metric",
    "MemoryTokenOverheadMetric",
    "MultiHopRetrievalMetric",
]

MEMORY_METADATA_KEY = "memory"
EXPECTATIONS_METADATA_KEY = "expectations"


def _section(test_case: ConversationalTestCase, key: str) -> dict[str, Any]:
    meta = test_case.metadata or {}
    section = meta.get(key)
    return section if isinstance(section, dict) else {}


def _require(
    test_case: ConversationalTestCase,
    section: str,
    field: str,
    metric: BaseConversationalMetric,
) -> Any:
    value = _section(test_case, section).get(field)
    if value is None or value == [] or value == "":
        raise MissingTestCaseParamsError(
            f"{metric.__name__} requires "
            f"`metadata['{section}']['{field}']` on the ConversationalTestCase. "
            "Populate it from the golden's `additional_metadata` via "
            "memory_provider_evals.bridge.conversational_test_case_from_trace()."
        )
    return value


def _last_assistant_content(test_case: ConversationalTestCase) -> str:
    for turn in reversed(test_case.turns or []):
        if turn.role == "assistant":
            return str(turn.content or "")
    return ""


class _DeterministicConversationalMetric(BaseConversationalMetric):
    """Shared plumbing: no judge model, sync scoring, async delegates to sync."""

    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ) -> None:
        self.threshold = 1.0 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        # Deterministic: nothing to parallelise, and no judge model to call.
        self.async_mode = False
        self.evaluation_model = "deterministic"

    async def a_measure(
        self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any
    ) -> float:
        return self.measure(test_case, *args, **kwargs)

    def _finalize(self, score: float, reason: str) -> float:
        self.score = score
        self.reason = reason if self.include_reason else None
        self.success = self.is_successful()
        if self.verbose_mode:
            self.verbose_logs = f"{self.__name__}: score={score:.3f} — {reason}"
        return self.score


class MemoryRecallF1Metric(_DeterministicConversationalMetric):
    """Token-F1 between the golden's current fact and the final assistant turn.

    Rewards recalling the ground-truth fact (recall) while penalising padding
    the answer with unrelated or hallucinated content (precision). Deterministic
    by design: this is the reproducible accuracy axis, and the one plotted
    against token overhead on the cost/accuracy frontier.
    """

    @property
    def __name__(self):
        return "Memory Recall F1"

    def measure(
        self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any
    ) -> float:
        check_conversational_test_case_params(
            test_case, [MultiTurnParams.ROLE, MultiTurnParams.CONTENT], self
        )
        current_fact = str(
            _require(test_case, EXPECTATIONS_METADATA_KEY, "current_fact", self)
        )
        answer = _last_assistant_content(test_case)
        score = token_f1(current_fact, answer)
        reason = (
            f"Token-F1 {score:.2f} between expected current fact "
            f"{current_fact!r} and the final assistant turn "
            f"{answer[:120]!r}."
        )
        return self._finalize(score, reason)


class MemoryTokenOverheadMetric(_DeterministicConversationalMetric):
    """Penalises memory injection that costs more context than it earns.

    Score is ``1.0`` while the observed overhead ratio (injected memory tokens /
    generated tokens) stays within ``max_token_overhead_ratio``, then decays as
    ``budget / observed`` so a provider that doubles the budget scores 0.5.
    """

    @property
    def __name__(self):
        return "Memory Token Overhead"

    def measure(
        self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any
    ) -> float:
        memory = _section(test_case, MEMORY_METADATA_KEY)
        budget = float(
            _require(
                test_case,
                EXPECTATIONS_METADATA_KEY,
                "max_token_overhead_ratio",
                self,
            )
        )
        if budget <= 0:
            raise MissingTestCaseParamsError(
                f"{self.__name__} requires a positive "
                "`max_token_overhead_ratio`."
            )
        generated = int(memory.get("generated_tokens", 0) or 0)
        injected = int(memory.get("injected_tokens", 0) or 0)
        observed = memory.get("token_overhead_ratio")
        if observed is None:
            observed = (injected / generated) if generated else 0.0
        observed = float(observed)

        score = 1.0 if observed <= budget else budget / observed
        reason = (
            f"Memory injected {injected} tokens against {generated} generated "
            f"({observed:.2f} overhead ratio) versus a {budget:.2f} budget."
        )
        return self._finalize(score, reason)


class MultiHopRetrievalMetric(_DeterministicConversationalMetric):
    """Binary: did retrieval connect the golden's entities across >= 2 sessions?

    Guards against a provider that "passes" a multi-hop probe purely from
    single-session recall or from context still sitting in the window. Scores
    1.0 only when every expected entity is surfaced by retrieved passages *and*
    those retrievals span at least two distinct ``session_id``s.
    """

    @property
    def __name__(self):
        return "Multi-Hop Retrieval"

    def __init__(self, match_threshold: float = 0.6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.match_threshold = match_threshold

    def measure(
        self, test_case: ConversationalTestCase, *args: Any, **kwargs: Any
    ) -> float:
        entities = list(
            _require(
                test_case, EXPECTATIONS_METADATA_KEY, "multi_hop_entities", self
            )
        )
        records = _section(test_case, MEMORY_METADATA_KEY).get("retrievals") or []
        connected = connects_entities_across_sessions(
            list(records), entities, self.match_threshold
        )
        sessions = sorted(
            {str(r.get("session_id", "")) for r in records if isinstance(r, dict)}
        )
        reason = (
            f"Entities {entities} {'were' if connected else 'were NOT'} all "
            f"surfaced by retrievals spanning >= 2 sessions "
            f"(observed sessions: {sessions or 'none'}, "
            f"{len(records)} retrieval record(s))."
        )
        return self._finalize(1.0 if connected else 0.0, reason)
