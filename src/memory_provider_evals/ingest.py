"""Import harness traces into DeepEval single-turn test cases.

Moved out of ``traced_harness`` so the instrument stays DeepEval-free: the
harness produces JSONL traces, and this study repo decides how to evaluate
them.

Implements the first-failure evaluation principle from 'Evals for AI Engineers'
(Husain & Shankar, Ch. 3 & Ch. 8): multi-turn agent errors cascade forward.
Subsequent turns after the first observed failure are polluted by the upstream
failure and should be pruned to avoid cataloging symptoms or blaming downstream
components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall
from traced_harness.eval import TraceTurn, load_trace

__all__ = ["to_deepeval_test_cases", "turn_to_llm_test_case"]


def turn_to_llm_test_case(turn: TraceTurn) -> LLMTestCase:
    """Convert one harness trace turn into a DeepEval ``LLMTestCase``."""
    return LLMTestCase(
        input=turn.input,
        actual_output=turn.actual_output,
        tools_called=[
            ToolCall(
                name=t.name,
                input_parameters=t.input_parameters,
                output=t.output,
            )
            for t in turn.tools_called
        ],
        metadata=turn.additional_metadata,
    )


def to_deepeval_test_cases(
    trace_file: str | Path,
    stop_at_first_failure: bool = True,
) -> list[Any]:
    """Import an exact session trace directly as DeepEval ``LLMTestCase``s.

    If ``stop_at_first_failure`` is True, evaluation stops at the first failed
    turn, omitting subsequent polluted turns from the test dataset.
    """
    test_cases: list[Any] = []
    for turn in load_trace(trace_file):
        test_cases.append(turn_to_llm_test_case(turn))
        if stop_at_first_failure and turn.has_peripheral_errors:
            break
    return test_cases
