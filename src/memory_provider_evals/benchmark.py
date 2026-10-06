"""Memory-provider benchmark: comparative measurement, not a pass/fail gate.

This is the benchmark half of the repo. The DeepEval suite in ``tests/evals``
answers "does this provider meet a bar?"; this answers "which provider is
best?" — the question you actually rank on.

Built on BenchKit's measurement substrate, consumed as a library:

* :class:`benchkit_for_harnesses.archive.ArchiveWriter` — streaming JSONL with
  content-hashed, self-certifying filenames. A crash at scenario 9 preserves
  0–8.
* ``benchkit_for_harnesses.brackets`` — the answer-bracket protocol
  ``{[{[ … ]}]}``. Deterministic scoring, no judge needed for the headline
  accuracy number.
* ``benchkit_for_harnesses.ledger`` — one line per benchmark invocation under
  ``$BENCHKIT_HOME``, so runs are comparable across time.

Why not ``benchkit.core.run_items``
-----------------------------------
BenchKit's loop is one prompt per item, and every built-in harness adapter
deliberately kills session state (``--no-session``, ``--ephemeral``). A memory
benchmark needs the inverse: continuity across sessions plus a consolidation
pass between them. So this module keeps ``traced_harness.SessionRunner`` as the
execution engine and borrows BenchKit's *measurement* layer — the same split
``ifeval`` and ``bundled_bench`` use as sub-experiments.

Scoring columns per scenario
----------------------------
================  ===========  ==========================================
column            kind         meaning
================  ===========  ==========================================
``correct``       bracket      final answer matched the target via
                               ``eval_bracketed(strict=True)`` — the headline
                               accuracy. Strict because loose mode falls back
                               to substring containment, which credits a hedge
                               ("you used to drive a Honda Civic, but you might
                               drive a Tesla Model 3 now") — precisely the
                               belief-revision failure under test.
``recall_f1``     det.         token-F1 of the recalled fact
``overhead``      det.         injected memory tokens / generated tokens
``multi_hop``     det.         entities joined across >= 2 sessions
``stale_leak``    det.         a superseded fact resurfaced anywhere
``latency_ms``    measured     wall time for the whole scenario
``consolidation`` measured     between-session wall/CPU/bytes
================  ===========  ==========================================
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchkit_for_harnesses.archive import ArchiveWriter
from benchkit_for_harnesses.brackets import (
    ANSWER_INSTRUCTION,
    eval_bracketed,
    extract_bracketed_answer,
)
from benchkit_for_harnesses.ledger import append_entry, runs_dir
from deepeval.dataset import ConversationalGolden
from traced_harness.memory import MemoryProviderAdapter, make_memory_session_runner
from traced_harness.session_runner import TurnExecutor

from memory_provider_evals.bridge import (
    conversational_test_case_from_trace,
    scenario_from_golden,
)
from memory_provider_evals.trace import MemoryEvalSuite, TraceRecord, token_f1

__all__ = [
    "BenchmarkReport",
    "ExecutorFactory",
    "ScenarioResult",
    "bracket_instructed",
    "rank",
    "render_table",
    "run_provider_benchmark",
]

#: Builds the turn executor for one scenario, scoped to that scenario's
#: workspace. An async context manager so a live provider can hold an MCP
#: connection open for the scenario and close it afterwards.
ExecutorFactory = Callable[[Path], AbstractAsyncContextManager[TurnExecutor]]


@asynccontextmanager
async def _scenario_executor(
    turn_executor: TurnExecutor | None,
    executor_factory: ExecutorFactory | None,
    scenario_workspace: Path,
) -> AsyncIterator[TurnExecutor]:
    """Yield the executor for one scenario, from whichever source was given."""
    if executor_factory is None:
        assert turn_executor is not None  # guarded by the caller
        yield turn_executor
        return
    async with executor_factory(scenario_workspace) as executor:
        yield executor


def bracket_instructed(golden: ConversationalGolden) -> ConversationalGolden:
    """Append BenchKit's answer-bracket instruction to the final probe turn.

    Deterministic scoring needs the model to delimit its answer. Only the last
    user turn is touched — the ingress and revision turns must stay verbatim,
    because they are the belief-revision setup under test.
    """
    turns = list(golden.turns or [])
    user_idx = [i for i, t in enumerate(turns) if getattr(t, "role", "user") == "user"]
    if not user_idx:
        return golden
    last = user_idx[-1]
    probe = turns[last]
    patched = probe.model_copy(
        update={"content": f"{probe.content}\n\n{ANSWER_INSTRUCTION}"}
    )
    turns[last] = patched
    return golden.model_copy(update={"turns": turns})


@dataclass
class ScenarioResult:
    """One scenario against one provider — a BenchKit-shaped archive record."""

    idx: int
    benchmark: str
    scenario: str
    provider: str
    model: str
    target: str
    response: str
    correct: bool
    latency_ms: int
    recall_f1: float
    overhead_ratio: float
    multi_hop: bool | None
    stale_leak: bool | None
    n_sessions: int
    n_retrievals: int
    consolidation_wall_s: float
    consolidation_bytes: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    provider: str
    model: str
    results: list[ScenarioResult] = field(default_factory=list)
    archive_path: Path | None = None

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        return (
            sum(r.correct for r in self.results) / self.n if self.n else 0.0
        )

    @property
    def mean_recall_f1(self) -> float:
        return (
            sum(r.recall_f1 for r in self.results) / self.n if self.n else 0.0
        )

    @property
    def mean_overhead(self) -> float:
        return (
            sum(r.overhead_ratio for r in self.results) / self.n if self.n else 0.0
        )

    @property
    def stale_leak_rate(self) -> float:
        seen = [r for r in self.results if r.stale_leak is not None]
        return sum(bool(r.stale_leak) for r in seen) / len(seen) if seen else 0.0

    @property
    def mean_latency_ms(self) -> int:
        return (
            round(sum(r.latency_ms for r in self.results) / self.n) if self.n else 0
        )

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if r.error)


def _score(
    trace: TraceRecord,
    golden: ConversationalGolden,
    provider: str,
) -> dict[str, Any]:
    """Deterministic memory columns, reusing the shared judges."""
    suite = MemoryEvalSuite()
    meta = golden.additional_metadata or {}
    tc = conversational_test_case_from_trace(trace, golden, provider=provider)
    mem = tc.metadata["memory"]

    current = str(meta.get("current_fact") or "")
    superseded = list(meta.get("superseded_facts") or [])
    entities = list(meta.get("multi_hop_entities") or [])

    final_answer = tc.turns[-1].content if tc.turns else ""
    # Score token-F1 against the *extracted* answer: the bracket delimiters are
    # protocol scaffolding, and counting them as tokens deflates precision on a
    # perfectly correct response.
    scored_answer = extract_bracketed_answer(str(final_answer)) or str(final_answer)

    return {
        "recall_f1": token_f1(current, scored_answer) if current else 0.0,
        "overhead_ratio": float(mem["token_overhead_ratio"]),
        "multi_hop": (
            suite.eval_multi_hop(trace, entities) if len(entities) >= 2 else None
        ),
        # eval_temporal_invalidation returns True when CLEAN; we report the leak.
        "stale_leak": (
            any(not suite.eval_temporal_invalidation(trace, f) for f in superseded)
            if superseded
            else None
        ),
        "n_retrievals": len(mem["retrievals"]),
        "consolidation_wall_s": sum(
            float(c.get("wall_seconds", 0.0)) for c in mem["consolidations"]
        ),
        "consolidation_bytes": sum(
            int(c.get("bytes_growth", 0)) for c in mem["consolidations"]
        ),
        "_final_answer": str(final_answer),
    }


async def run_provider_benchmark(
    goldens: list[ConversationalGolden],
    adapter: MemoryProviderAdapter,
    turn_executor: TurnExecutor | None = None,
    workspace_dir: str | Path = ".",
    model: str = "",
    benchmark: str = "memorybench",
    output_dir: str | Path | None = None,
    reset_between: bool = True,
    executor_factory: ExecutorFactory | None = None,
) -> BenchmarkReport:
    """Run every scenario against one provider, streaming a BenchKit archive.

    Supply either ``turn_executor`` (one executor for the whole run — right
    for a fake, or for a provider holding no per-scenario resources) or
    ``executor_factory`` (one per scenario, scoped to that scenario's
    workspace). A live provider needs the latter: its MCP connection and its
    store belong to the scenario's workspace, and sharing them would let
    scenario 2 recall what scenario 1 stored, which is exactly the leakage
    the benchmark is meant to detect.

    Never raises on a scenario failure: a failed scenario becomes a record
    with ``error`` set and ``correct=False``, so n-counts stay honest and one
    broken provider cannot void the comparison.
    """
    if (turn_executor is None) == (executor_factory is None):
        raise ValueError(
            "Pass exactly one of turn_executor or executor_factory."
        )

    out = Path(output_dir) if output_dir else runs_dir()
    report = BenchmarkReport(provider=adapter.name, model=model)
    desc = f"{benchmark}_{adapter.name}_{model or 'default'}"
    started = time.perf_counter()

    with ArchiveWriter(out, desc) as writer:
        for idx, raw_golden in enumerate(goldens):
            golden = bracket_instructed(raw_golden)
            meta = raw_golden.additional_metadata or {}
            target = str(meta.get("current_fact") or "")
            name = getattr(raw_golden, "name", None) or f"scenario_{idx}"
            scenario_workspace = Path(workspace_dir) / name
            t0 = time.perf_counter()
            runner: Any = None
            try:
                async with _scenario_executor(
                    turn_executor, executor_factory, scenario_workspace
                ) as executor:
                    runner = make_memory_session_runner(
                        adapter, executor, scenario_workspace
                    )
                    run = await runner.run_scenario(scenario_from_golden(golden))
                trace = TraceRecord.from_file(run.trace_file)
                scored = _score(trace, raw_golden, adapter.name)
                answer = str(scored.pop("_final_answer"))
                result = ScenarioResult(
                    idx=idx,
                    benchmark=benchmark,
                    scenario=name,
                    provider=adapter.name,
                    model=model,
                    target=target,
                    response=answer,
                    correct=bool(target)
                    and eval_bracketed(answer, target, strict=True),
                    latency_ms=round((time.perf_counter() - t0) * 1000),
                    n_sessions=len(run.turn_results),
                    **scored,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                result = ScenarioResult(
                    idx=idx,
                    benchmark=benchmark,
                    scenario=name,
                    provider=adapter.name,
                    model=model,
                    target=target,
                    response=f"[ERROR {type(exc).__name__}] {exc}",
                    correct=False,
                    latency_ms=round((time.perf_counter() - t0) * 1000),
                    recall_f1=0.0,
                    overhead_ratio=0.0,
                    multi_hop=None,
                    stale_leak=None,
                    n_sessions=0,
                    n_retrievals=0,
                    consolidation_wall_s=0.0,
                    consolidation_bytes=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if reset_between and runner is not None:
                    runner.reset_suite()

            report.results.append(result)
            writer.write_record(result.to_dict())

    report.archive_path = writer.final_path
    append_entry(
        {
            "cmd": "memorybench",
            "benchmark": benchmark,
            "harness": "traced-harness",
            "provider": adapter.name,
            "model": model,
            "n_items": report.n,
            "accuracy": report.accuracy,
            "n_errors": report.n_errors,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "exit_code": 0,
            "output_path": str(report.archive_path) if report.archive_path else None,
        }
    )
    return report


def rank(reports: list[BenchmarkReport]) -> list[BenchmarkReport]:
    """Best first: accuracy desc, then stale-leak asc, then overhead asc.

    Accuracy is the headline. Ties break on *not* resurfacing stale facts —
    a provider that recalls well but leaks superseded beliefs is worse than one
    that does neither — and then on context cost.
    """
    return sorted(
        reports,
        key=lambda r: (-r.accuracy, r.stale_leak_rate, r.mean_overhead),
    )


def render_table(reports: list[BenchmarkReport]) -> str:
    """Comparison table, best first. Measurement, not a verdict."""
    ranked = rank(reports)
    head = (
        f"{'provider':<12} {'acc':>7} {'recallF1':>9} {'stale':>7} "
        f"{'overhead':>9} {'lat(ms)':>9} {'n':>4} {'err':>4}"
    )
    lines = [head, "-" * len(head)]
    for r in ranked:
        lines.append(
            f"{r.provider:<12} {r.accuracy:>6.1%} {r.mean_recall_f1:>9.2f} "
            f"{r.stale_leak_rate:>6.1%} {r.mean_overhead:>9.2f} "
            f"{r.mean_latency_ms:>9} {r.n:>4} {r.n_errors:>4}"
        )
    if any(r.n_errors for r in ranked):
        lines.append("")
        lines.append(
            "NOTE: providers with err>0 did not complete every scenario; "
            "their accuracy is not comparable."
        )
    return "\n".join(lines)
