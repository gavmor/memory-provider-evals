"""Tests for MemoryEvalSuite judges over synthetic memory traces.

These judges are the deterministic internals of the DeepEval metrics in
``memory_provider_evals.metrics``.
"""

from __future__ import annotations

from traced_harness.eval import TraceTurn

from memory_provider_evals.trace import MemoryEvalSuite, TraceRecord


def _turn(index, prompt, output, memory=None, tools=None):
    return TraceTurn(
        index=index,
        input=prompt,
        actual_output=output,
        tools_called=tools or [],
        additional_metadata={"memory": memory or {}},
    )


def _belief_revision_trace(final_answer: str) -> TraceRecord:
    """Honda (s1) -> Tesla (s2) -> probe (s3)."""
    return TraceRecord(
        turns=[
            _turn(
                0,
                "I drive a Honda Civic",
                "Noted.",
                memory={"session_id": "s1", "injection": {"overhead_tokens": 0}},
            ),
            _turn(
                1,
                "Actually I drive a Tesla Model 3 now",
                "Updated.",
                memory={
                    "session_id": "s2",
                    "injection": {"overhead_tokens": 20},
                    "retrievals": [
                        {
                            "query": "what vehicle",
                            "passages": ["user previously drove a Honda Civic"],
                            "passage_count": 1,
                        }
                    ],
                },
            ),
            _turn(
                2,
                "What do I drive?",
                final_answer,
                memory={
                    "session_id": "s3",
                    "injection": {"overhead_tokens": 20},
                    "generated_tokens": 5,
                    "retrievals": [
                        {
                            "query": "current vehicle",
                            "passages": ["user drives a Tesla Model 3"],
                            "passage_count": 1,
                        }
                    ],
                },
            ),
        ]
    )


def test_precision_recall_rewards_correct_fact():
    trace = _belief_revision_trace("You drive a Tesla Model 3")
    suite = MemoryEvalSuite()
    score = suite.eval_precision_recall(trace, "Tesla Model 3")
    assert score > 0.5


def test_precision_recall_zero_when_unrelated():
    trace = _belief_revision_trace("I have no idea what you drive")
    suite = MemoryEvalSuite()
    assert suite.eval_precision_recall(trace, "Tesla Model 3") == 0.0


def test_temporal_invalidation_true_when_stale_absent():
    trace = _belief_revision_trace("You drive a Tesla Model 3")
    suite = MemoryEvalSuite()
    # 'Honda Civic' only appears in an earlier turn's retrieval about the past;
    # here the final answer is clean but the s2 retrieval DID surface Honda.
    assert suite.eval_temporal_invalidation(trace, "Honda Civic") is False


def test_temporal_invalidation_true_when_fully_clean():
    trace = TraceRecord(
        turns=[
            _turn(
                0,
                "What do I drive?",
                "You drive a Tesla Model 3",
                memory={"session_id": "s3"},
            )
        ]
    )
    suite = MemoryEvalSuite()
    assert suite.eval_temporal_invalidation(trace, "Honda Civic") is True


def test_token_overhead_ratio():
    trace = _belief_revision_trace("You drive a Tesla Model 3")
    suite = MemoryEvalSuite()
    # overhead tokens = 0 + 20 + 20 = 40; generated = heuristic + heuristic + 5
    ratio = suite.eval_token_overhead_ratio(trace)
    assert ratio > 0.0


def test_contradiction_rejection():
    suite = MemoryEvalSuite()
    clean = _belief_revision_trace("You drive a Tesla Model 3")
    assert suite.eval_contradiction_rejection(clean, ["Honda Civic"]) is True

    polluted = _belief_revision_trace("You drive a Honda Civic")
    assert suite.eval_contradiction_rejection(polluted, ["Honda Civic"]) is False


def test_multi_hop_requires_cross_session_entities():
    suite = MemoryEvalSuite()
    trace = TraceRecord(
        turns=[
            _turn(
                0,
                "intro",
                "ok",
                memory={
                    "session_id": "s1",
                    "retrievals": [
                        {"query": "q", "passages": ["Tesla was bought in Toronto"]}
                    ],
                },
            ),
            _turn(
                1,
                "probe",
                "ok",
                memory={
                    "session_id": "s2",
                    "retrievals": [
                        {"query": "q", "passages": ["Alice moved to Berlin"]}
                    ],
                },
            ),
        ]
    )
    # Tesla (s1) + Berlin (s2): connected across two sessions.
    assert suite.eval_multi_hop(trace, ["Tesla", "Berlin"]) is True
    # Entity never retrieved -> False.
    assert suite.eval_multi_hop(trace, ["Tesla", "Saturn"]) is False
    # Both in the same session -> no cross-session hop.
    assert suite.eval_multi_hop(trace, ["Tesla", "Toronto"]) is False


def test_cost_accuracy_point():
    trace = _belief_revision_trace("You drive a Tesla Model 3")
    suite = MemoryEvalSuite()
    overhead, acc = suite.cost_accuracy_point(trace, 0.9)
    assert overhead == 40.0
    assert acc == 0.9


def test_from_file_roundtrip(tmp_path):
    import json

    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": "q",
                "actual_output": "You drive a Tesla",
                "tools_called": [
                    {
                        "name": "cashew_query",
                        "input_parameters": {"q": "vehicle"},
                        "output": "user drives a Tesla",
                    }
                ],
                "additional_metadata": {"memory": {"session_id": "s1"}},
            }
        )
        + "\n"
    )
    trace = TraceRecord.from_file(path)
    assert len(trace.turns) == 1
    # Memory-tool output counts as retrieved context.
    assert any("Tesla" in t for t in trace.retrieved_texts())
