"""Trace ingestion into DeepEval test cases, with cascade pruning.

Validates that:
1. Recorded harness traces are ingested directly into DeepEval LLMTestCase
   objects without re-running model inference.
2. The Husain & Shankar first-failure evaluation principle (Ch. 3 & 8) is
   enforced: ingestion stops at the first observed peripheral failure and
   prunes cascaded downstream turns.
3. Ingested test cases are evaluable with DeepEval metrics
   (e.g. ToolCorrectnessMetric).

The non-DeepEval half of this behaviour (``evaluate_trace`` reporting, agent
context retention) stays in the traced-harness repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

from memory_provider_evals.ingest import to_deepeval_test_cases

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "traces"


class LocalDeterministicEvaluationLLM(DeepEvalBaseLLM):
    """Local, deterministic stub so metric execution needs no network."""

    def load_model(self) -> Any:
        return None

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return "Deterministic evaluation check passed."

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return self.generate(prompt, *args, **kwargs)

    def get_model_name(self) -> str:
        return "local-deterministic-evaluator"


def test_trace_ingestion_into_deepeval_test_cases() -> None:
    """Recorded traces convert into LLMTestCase instances without inference."""
    trace_path = FIXTURES_DIR / "clean_run.jsonl"
    test_cases = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)

    assert len(test_cases) == 2
    for tc in test_cases:
        assert isinstance(tc, LLMTestCase)
        assert tc.input
        assert tc.actual_output
        assert len(tc.tools_called) >= 1
        for tool in tc.tools_called:
            assert isinstance(tool, ToolCall)
            assert tool.name
            assert isinstance(tool.input_parameters, dict)
            assert tool.output

    turn0 = test_cases[0]
    assert turn0.input == "What is the health and armor of the Colonial Spatha tank?"
    assert "3650 HP" in turn0.actual_output
    assert turn0.tools_called[0].name == "get_vehicle_stats"
    assert turn0.tools_called[0].input_parameters == {"vehicle_name": "Spatha"}
    meta = getattr(turn0, "metadata", None) or getattr(
        turn0, "additional_metadata", {}
    )
    assert meta.get("session_id") == "clean_session"
    assert meta.get("agent") == "traced_agno"


def test_first_failure_cascade_pruning_on_ingest() -> None:
    """Ingestion stops at the first failure, omitting polluted turns."""
    trace_path = FIXTURES_DIR / "multi_turn_cascade.jsonl"

    pruned = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)
    assert len(pruned) == 2
    assert pruned[0].tools_called[0].name == "get_vehicle_stats"
    assert pruned[1].tools_called[0].name == "get_map_intel"

    unpruned = to_deepeval_test_cases(trace_path, stop_at_first_failure=False)
    assert len(unpruned) == 4


def test_failing_peripheral_is_ingested() -> None:
    trace_path = FIXTURES_DIR / "failing_peripheral.jsonl"
    test_cases = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)
    assert len(test_cases) == 1
    assert test_cases[0].tools_called[0].name == "get_map_intel"


def test_deepeval_tool_correctness_metric_evaluation() -> None:
    """Ingested test cases are evaluable with DeepEval's ToolCorrectnessMetric."""
    trace_path = FIXTURES_DIR / "clean_run.jsonl"
    test_cases = to_deepeval_test_cases(trace_path, stop_at_first_failure=True)
    local_model = LocalDeterministicEvaluationLLM()

    case0 = test_cases[0]
    case0.expected_tools = [
        ToolCall(
            name="get_vehicle_stats",
            input_parameters={"vehicle_name": "Spatha"},
        )
    ]
    metric = ToolCorrectnessMetric(
        model=local_model,
        should_exact_match=True,
        async_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )
    metric.measure(case0)
    assert metric.score == 1.0
    assert metric.is_successful() is True

    case1 = test_cases[1]
    case1.expected_tools = [
        ToolCall(
            name="get_production_cost",
            input_parameters={"item_name": "DifferentTank"},  # intentional mismatch
        )
    ]
    metric_mismatch = ToolCorrectnessMetric(
        model=local_model,
        should_exact_match=True,
        async_mode=False,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )
    metric_mismatch.measure(case1)
    assert metric_mismatch.score == 0.0
    assert metric_mismatch.is_successful() is False
