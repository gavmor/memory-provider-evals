"""Unit tests for the DeepEval metrics and the trace -> test-case bridge.

These run under plain pytest with no model and no network: they exercise the
deterministic metric classes and the ``ConversationalTestCase`` construction
that the committed eval suite in ``tests/evals/`` depends on.
"""

from __future__ import annotations

import json

import pytest
from deepeval.dataset import ConversationalGolden
from deepeval.errors import MissingTestCaseParamsError
from deepeval.test_case import ConversationalTestCase, Turn

from memory_provider_evals.adapters import build_adapter
from memory_provider_evals.bridge import (
    conversational_test_case_from_trace,
    scenario_from_golden,
)
from memory_provider_evals.metrics import (
    MemoryRecallF1Metric,
    MemoryTokenOverheadMetric,
    MultiHopRetrievalMetric,
)
from memory_provider_evals.trace import TraceRecord

BELIEF_REVISION_GOLDEN = ConversationalGolden(
    name="belief_revision_vehicle",
    scenario="User states a vehicle, corrects it, then probes for it.",
    expected_outcome="The assistant answers Tesla Model 3.",
    turns=[
        Turn(
            role="user",
            content="I drive a Honda Civic.",
            metadata={"session_id": "s1"},
        ),
        Turn(
            role="user",
            content="Actually, I drive a Tesla Model 3 now.",
            metadata={"session_id": "s2"},
        ),
        Turn(
            role="user", content="What do I drive?", metadata={"session_id": "s3"}
        ),
    ],
    additional_metadata={
        "current_fact": "Tesla Model 3",
        "superseded_facts": ["Honda Civic"],
        "multi_hop_entities": ["Tesla", "Berlin"],
        "max_token_overhead_ratio": 0.5,
    },
)


def _write_trace(tmp_path, final_answer: str) -> TraceRecord:
    """Honda (s1) -> Tesla (s2) -> probe (s3), matching the golden."""
    rows = [
        {
            "input": "I drive a Honda Civic.",
            "actual_output": "Noted.",
            "tools_called": [],
            "additional_metadata": {
                "memory": {
                    "session_id": "s1",
                    "injection": {"overhead_tokens": 0},
                    "generated_tokens": 2,
                }
            },
        },
        {
            "input": "Actually, I drive a Tesla Model 3 now.",
            "actual_output": "Updated.",
            "tools_called": [],
            "additional_metadata": {
                "memory": {
                    "session_id": "s2",
                    "injection": {"overhead_tokens": 20},
                    "generated_tokens": 2,
                    "retrievals": [
                        {
                            "query": "what vehicle",
                            "passages": ["user previously drove a Honda Civic"],
                            "passage_count": 1,
                        }
                    ],
                }
            },
        },
        {
            "input": "What do I drive?",
            "actual_output": final_answer,
            "tools_called": [
                {
                    "name": "cashew_query",
                    "input_parameters": {"q": "vehicle"},
                    "output": "user drives a Tesla Model 3",
                }
            ],
            "additional_metadata": {
                "memory": {
                    "session_id": "s3",
                    "injection": {"overhead_tokens": 20},
                    "generated_tokens": 36,
                    "retrievals": [
                        {
                            "query": "current vehicle",
                            "passages": ["Alice moved to Berlin"],
                            "passage_count": 1,
                        }
                    ],
                }
            },
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return TraceRecord.from_file(path)


@pytest.fixture
def passing_case(tmp_path) -> ConversationalTestCase:
    trace = _write_trace(tmp_path, "You drive a Tesla Model 3.")
    return conversational_test_case_from_trace(
        trace, BELIEF_REVISION_GOLDEN, provider="cashew"
    )


# -- bridge -----------------------------------------------------------------
def test_scenario_from_golden_groups_sessions():
    scenario = scenario_from_golden(BELIEF_REVISION_GOLDEN)
    assert [s.session_id for s in scenario.sessions] == ["s1", "s2", "s3"]
    assert scenario.sessions[0].turns == ["I drive a Honda Civic."]


def test_scenario_from_golden_merges_contiguous_same_session():
    golden = ConversationalGolden(
        scenario="two turns in one session",
        turns=[
            Turn(role="user", content="a", metadata={"session_id": "s1"}),
            Turn(role="user", content="b", metadata={"session_id": "s1"}),
            Turn(role="user", content="c", metadata={"session_id": "s2"}),
        ],
    )
    scenario = scenario_from_golden(golden)
    assert [s.session_id for s in scenario.sessions] == ["s1", "s2"]
    assert scenario.sessions[0].turns == ["a", "b"]


def test_scenario_from_golden_isolates_unlabelled_turns():
    """No session_id => one session per turn, forcing cross-session recall."""
    golden = ConversationalGolden(
        scenario="unlabelled",
        turns=[Turn(role="user", content="a"), Turn(role="user", content="b")],
    )
    assert len(scenario_from_golden(golden).sessions) == 2


def test_scenario_from_golden_rejects_turnless_golden():
    """A generated golden with no turns cannot drive a scripted replay."""
    golden = ConversationalGolden(scenario="generated scenario only", turns=[])
    with pytest.raises(ValueError, match="no user turns"):
        scenario_from_golden(golden)


def test_test_case_is_conversational_with_retrieval_context(passing_case):
    assert isinstance(passing_case, ConversationalTestCase)
    assert len(passing_case.turns) == 6  # 3 trace turns -> user+assistant pairs
    assert [t.role for t in passing_case.turns[:2]] == ["user", "assistant"]

    final_assistant = passing_case.turns[-1]
    assert final_assistant.role == "assistant"
    # Retrieval span passages AND memory-tool output both land as context.
    assert "Alice moved to Berlin" in final_assistant.retrieval_context
    assert "user drives a Tesla Model 3" in final_assistant.retrieval_context
    assert final_assistant.tools_called[0].name == "cashew_query"


def test_expectations_are_surfaced_to_judges_via_context(passing_case):
    blob = " ".join(passing_case.context)
    assert "Tesla Model 3" in blob
    assert "Honda Civic" in blob
    assert "SUPERSEDED FACTS" in blob


def test_memory_metadata_contract(passing_case):
    mem = passing_case.metadata["memory"]
    assert mem["provider"] == "cashew"
    assert mem["injected_tokens"] == 40
    assert mem["generated_tokens"] == 40
    assert mem["token_overhead_ratio"] == pytest.approx(1.0)
    assert {r["session_id"] for r in mem["retrievals"]} == {"s2", "s3"}


# -- deterministic metrics --------------------------------------------------
def test_recall_f1_rewards_correct_fact(passing_case):
    metric = MemoryRecallF1Metric(threshold=0.5)
    assert metric.measure(passing_case) > 0.5
    assert metric.is_successful() is True
    assert "Tesla Model 3" in metric.reason


def test_recall_f1_fails_on_unrelated_answer(tmp_path):
    trace = _write_trace(tmp_path, "I have no idea what you drive.")
    case = conversational_test_case_from_trace(trace, BELIEF_REVISION_GOLDEN)
    metric = MemoryRecallF1Metric(threshold=0.5)
    assert metric.measure(case) == 0.0
    assert metric.is_successful() is False


def test_token_overhead_scores_within_budget(passing_case):
    # observed ratio 1.0 against a 0.5 budget -> 0.5
    metric = MemoryTokenOverheadMetric(threshold=0.8)
    assert metric.measure(passing_case) == pytest.approx(0.5)
    assert metric.is_successful() is False


def test_token_overhead_full_score_under_budget(passing_case):
    passing_case.metadata["memory"]["token_overhead_ratio"] = 0.25
    metric = MemoryTokenOverheadMetric(threshold=0.8)
    assert metric.measure(passing_case) == 1.0
    assert metric.is_successful() is True


def test_multi_hop_requires_two_sessions(passing_case):
    # Both entities resolve within s3 only -> no cross-session hop.
    metric = MultiHopRetrievalMetric(threshold=1.0)
    assert metric.measure(passing_case) == 0.0
    assert metric.is_successful() is False

    passing_case.metadata["memory"]["retrievals"] = [
        {
            "query": "q",
            "passages": ["Tesla was bought in Toronto"],
            "session_id": "s2",
        },
        {"query": "q", "passages": ["Alice moved to Berlin"], "session_id": "s3"},
    ]
    metric = MultiHopRetrievalMetric(threshold=1.0)
    assert metric.measure(passing_case) == 1.0
    assert metric.is_successful() is True


def test_metric_raises_missing_params_when_expectation_absent(passing_case):
    passing_case.metadata["expectations"] = {}
    with pytest.raises(MissingTestCaseParamsError):
        MemoryRecallF1Metric().measure(passing_case)


def test_metrics_expose_deepeval_names():
    assert MemoryRecallF1Metric().__name__ == "Memory Recall F1"
    assert MemoryTokenOverheadMetric().__name__ == "Memory Token Overhead"
    assert MultiHopRetrievalMetric().__name__ == "Multi-Hop Retrieval"


@pytest.mark.asyncio
async def test_a_measure_matches_sync(passing_case):
    metric = MemoryRecallF1Metric()
    assert await metric.a_measure(passing_case) == metric.measure(passing_case)


def test_build_adapter_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown memory provider"):
        build_adapter("not-a-provider")
    assert build_adapter("cashew").name == "cashew"


def test_judge_params_are_all_renderable_by_conversational_geval():
    """ConversationalGEval renders only a fixed set of MultiTurnParams.

    Passing an unsupported one (CONTEXT, USER_DESCRIPTION, CHATBOT_ROLE, ...)
    raises KeyError deep inside scoring, long after the suite looks healthy.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "evals"))
    import metrics as eval_metrics
    from deepeval.metrics.g_eval.utils import CONVERSATIONAL_G_EVAL_PARAMS

    unsupported = [
        p.name
        for p in eval_metrics._JUDGE_PARAMS
        if p not in CONVERSATIONAL_G_EVAL_PARAMS
    ]
    assert not unsupported, (
        f"ConversationalGEval cannot render {unsupported}; "
        "scoring would fail with KeyError at runtime."
    )
