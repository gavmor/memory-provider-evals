"""Bridge between memory-provider scenario runs and DeepEval multi-turn evals.

This is the *app entry point* the committed eval suite in ``tests/evals/``
calls. It closes the loop:

    ConversationalGolden
        -> Scenario (ordered sessions of scripted user turns)
        -> traced_harness SessionRunner drives the real provider + agent,
           writing a JSONL trace
        -> TraceRecord
        -> ConversationalTestCase (turns carry retrieval_context + tools_called)
        -> assert_test(test_case=..., metrics=MULTI_TURN_METRICS)

Why multi-turn and not ``ConversationSimulator``
------------------------------------------------
DeepEval's simulator generates its own user turns. A memory benchmark depends
on a *scripted* belief-revision sequence (ingress fact -> superseding fact ->
probe); letting a simulator invent turns would destroy the contradiction under
test. So goldens carry their turns explicitly and we build the
``ConversationalTestCase`` by hand from the observed run.

Session boundaries
------------------
Each golden turn may carry ``metadata={"session_id": "s1"}``. Contiguous turns
sharing a ``session_id`` form one session. Turns with no ``session_id`` each get
their own session -- the strictest reading, because the harness keys agent
history by ``session_id``, so a new session means cross-session recall *must*
come from the memory provider rather than from context still in the window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepeval.dataset import ConversationalGolden
from deepeval.test_case import ConversationalTestCase, ToolCall, Turn
from deepeval.test_case.llm_test_case import RetrievedContextData
from traced_harness.eval import TraceTurn
from traced_harness.plugins import MemoryPluginAdapter
from traced_harness.session_runner import (
    Scenario,
    Session,
    SessionRunner,
    TurnExecutor,
)

from memory_provider_evals.metrics import (
    EXPECTATIONS_METADATA_KEY,
    MEMORY_METADATA_KEY,
)
from memory_provider_evals.trace import MemoryEvalSuite, TraceRecord

__all__ = [
    "conversational_test_case_from_trace",
    "run_memory_scenario",
    "scenario_from_golden",
]

#: Expectation keys a golden may declare under ``additional_metadata``.
EXPECTATION_KEYS = (
    "current_fact",
    "superseded_facts",
    "multi_hop_entities",
    "max_token_overhead_ratio",
)


def _golden_name(golden: ConversationalGolden, fallback: str) -> str:
    raw = getattr(golden, "name", None) or fallback
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(raw))
    return safe[:60] or fallback


def scenario_from_golden(
    golden: ConversationalGolden, name: str | None = None
) -> Scenario:
    """Group a golden's scripted user turns into ordered memory sessions."""
    turns = list(golden.turns or [])
    user_turns = [t for t in turns if getattr(t, "role", "user") == "user"]
    if not user_turns:
        raise ValueError(
            "ConversationalGolden has no user turns to replay; a memory "
            "scenario needs an explicit scripted turn sequence."
        )

    sessions: list[Session] = []
    for i, turn in enumerate(user_turns):
        meta = getattr(turn, "metadata", None) or {}
        sid = str(meta.get("session_id") or "")
        if sid and sessions and sessions[-1].session_id == sid:
            sessions[-1].turns.append(str(turn.content))
            continue
        sessions.append(
            Session(session_id=sid or f"s{i + 1}", turns=[str(turn.content)])
        )

    return Scenario(
        name=name or _golden_name(golden, "memory_scenario"),
        sessions=sessions,
    )


def _expectations(golden: ConversationalGolden) -> dict[str, Any]:
    meta = golden.additional_metadata or {}
    return {k: meta[k] for k in EXPECTATION_KEYS if meta.get(k) is not None}


def _expectation_context(expectations: dict[str, Any]) -> list[str]:
    """Render expectations as context lines the GEval judges can read.

    ``ConversationalGEval`` scores over declared ``MultiTurnParams``; test-case
    ``metadata`` is not a reliable judge input, so the facts under test are
    surfaced through ``context``, which is. Keeping this derived from
    ``additional_metadata`` means the golden stays the single source of truth
    for both the LLM-judge and the deterministic metrics.
    """
    lines: list[str] = []
    current = expectations.get("current_fact")
    if current:
        lines.append(
            f"CURRENT FACT (true as of the final turn): {current}. "
            "The assistant's final answer must assert this."
        )
    superseded = expectations.get("superseded_facts") or []
    if superseded:
        joined = "; ".join(str(f) for f in superseded)
        lines.append(
            f"SUPERSEDED FACTS (were true earlier, are now false): {joined}. "
            "The assistant must not assert any of these as current, and "
            "memory must not resurface them as current context."
        )
    entities = expectations.get("multi_hop_entities") or []
    if entities:
        lines.append(
            "MULTI-HOP ENTITIES that must be connected across sessions: "
            + ", ".join(str(e) for e in entities)
            + "."
        )
    return lines


def _turn_retrieval_context(
    trace: TraceRecord, turn: TraceTurn
) -> list[str | RetrievedContextData]:
    """Passages the memory provider surfaced for this specific turn."""
    texts: list[str | RetrievedContextData] = []
    for rec in TraceRecord._mem(turn).get("retrievals", []) or []:
        texts.extend(str(p) for p in rec.get("passages", []) or [])
    for tool in turn.tools_called:
        if tool.name in trace.memory_tool_names and tool.output:
            texts.append(str(tool.output))
    return texts


def conversational_test_case_from_trace(
    trace: TraceRecord,
    golden: ConversationalGolden,
    provider: str = "",
) -> ConversationalTestCase:
    """Build the DeepEval multi-turn test case from an observed memory trace.

    Each trace turn becomes a ``user`` / ``assistant`` :class:`Turn` pair. The
    assistant turn carries the memory provider's retrieved passages as
    ``retrieval_context`` and its tool calls as ``tools_called``, so the
    ``ConversationalGEval`` metrics can judge *what memory actually supplied*
    rather than only the final prose.
    """
    suite = MemoryEvalSuite()
    expectations = _expectations(golden)
    turns: list[Turn] = []

    for t in trace.turns:
        sid = trace.session_id(t)
        turns.append(
            Turn(role="user", content=t.input, metadata={"session_id": sid})
        )
        turns.append(
            Turn(
                role="assistant",
                content=t.actual_output,
                retrieval_context=_turn_retrieval_context(trace, t) or None,
                tools_called=[
                    ToolCall(
                        name=tool.name,
                        input_parameters=tool.input_parameters,
                        output=tool.output,
                    )
                    for tool in t.tools_called
                ]
                or None,
                metadata={
                    "session_id": sid,
                    MEMORY_METADATA_KEY: TraceRecord._mem(t),
                },
            )
        )

    metadata = {
        MEMORY_METADATA_KEY: {
            "provider": provider,
            "injected_tokens": trace.injected_tokens_total(),
            "generated_tokens": trace.generated_tokens_total(),
            "token_overhead_ratio": suite.eval_token_overhead_ratio(trace),
            "retrievals": trace.retrieval_records(),
            "consolidations": trace.consolidation_records(),
            "trace_file": str(trace.file_path) if trace.file_path else None,
        },
        EXPECTATIONS_METADATA_KEY: expectations,
    }

    return ConversationalTestCase(
        turns=turns,
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        context=(list(golden.context or []) + _expectation_context(expectations))
        or None,
        user_description=golden.user_description,
        name=getattr(golden, "name", None) or None,
        metadata=metadata,
    )


async def run_memory_scenario(
    golden: ConversationalGolden,
    adapter: MemoryPluginAdapter,
    turn_executor: TurnExecutor,
    workspace_dir: str | Path,
    trace_dir: str | Path | None = None,
    reset_after: bool = True,
) -> ConversationalTestCase:
    """Replay a golden against a live memory provider and return its test case.

    This is the real app execution path: it drives the harness
    :class:`SessionRunner`, which performs setup, runs every session in order
    with an inter-session consolidation pass, and writes an enriched JSONL
    trace.
    """
    scenario = scenario_from_golden(golden)
    runner = SessionRunner(
        adapter=adapter,
        turn_executor=turn_executor,
        workspace_dir=workspace_dir,
        trace_dir=trace_dir,
    )
    try:
        result = await runner.run_scenario(scenario)
    finally:
        if reset_after:
            runner.reset_suite()

    trace = TraceRecord.from_file(result.trace_file)
    return conversational_test_case_from_trace(
        trace, golden, provider=adapter.name
    )
