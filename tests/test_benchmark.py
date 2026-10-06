"""Benchmark substrate tests — ranking, archives, ledger, failure discipline.

These run with no model, no network, and no memory backend: a fake turn
executor replays canned provider behaviour, so the ranking logic itself is
what's under test.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchkit_for_harnesses.brackets import ANSWER_INSTRUCTION
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import Turn
from traced_harness.memory import MemoryProviderAdapter, MemoryToolContract

from memory_provider_evals.benchmark import (
    BenchmarkReport,
    ScenarioResult,
    bracket_instructed,
    rank,
    render_table,
    run_provider_benchmark,
)

GOLDEN = ConversationalGolden(
    name="belief_revision_vehicle",
    scenario="User states a vehicle, corrects it, then probes for it.",
    expected_outcome="The assistant answers Tesla Model 3.",
    turns=[
        Turn(role="user", content="I drive a Honda Civic.", metadata={"session_id": "s1"}),
        Turn(
            role="user",
            content="Actually, I drive a Tesla Model 3 now.",
            metadata={"session_id": "s2"},
        ),
        Turn(role="user", content="What do I drive?", metadata={"session_id": "s3"}),
    ],
    additional_metadata={
        "current_fact": "Tesla Model 3",
        "superseded_facts": ["Honda Civic"],
        "max_token_overhead_ratio": 0.5,
    },
)


class _FakeProvider(MemoryProviderAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(dry_run=True)
        self.name = name

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        return {"env": {}, "contract": self.contract(), "store_paths": []}

    def trigger_consolidation(self) -> None:
        self._record("consolidate", self.name)

    def teardown(self) -> None:
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(provider=self.name, tools=[f"{self.name}_recall"])


@dataclass
class _Turn:
    prompt: str
    output: str
    session_id: str
    tools_called: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _executor(final_answer: str, overhead: int = 10, passage: str = ""):
    """Replay a provider: only the probe turn's answer varies."""

    async def _exec(prompt: str, session_id: str, peripheral: str) -> _Turn:
        is_probe = "What do I drive" in prompt
        mem: dict[str, Any] = {
            "injection": {"overhead_tokens": overhead},
            "generated_tokens": 20,
        }
        if is_probe and passage:
            mem["retrievals"] = [
                {"query": "vehicle", "passages": [passage], "session_id": session_id}
            ]
        return _Turn(
            prompt=prompt,
            output=final_answer if is_probe else "Noted.",
            session_id=session_id,
            metadata=mem,
        )

    return _exec


def _run(name, answer, tmp_path, **kw) -> BenchmarkReport:
    return asyncio.run(
        run_provider_benchmark(
            goldens=[GOLDEN],
            adapter=_FakeProvider(name),
            turn_executor=_executor(answer, **kw),
            workspace_dir=tmp_path / name,
            model="fake-model",
            output_dir=tmp_path / "archives",
        )
    )


# -- bracket instruction ----------------------------------------------------
def test_bracket_instruction_only_patches_the_probe_turn():
    patched = bracket_instructed(GOLDEN)
    contents = [t.content for t in patched.turns]
    assert ANSWER_INSTRUCTION in contents[-1]
    # The belief-revision setup must stay verbatim.
    assert contents[0] == "I drive a Honda Civic."
    assert contents[1] == "Actually, I drive a Tesla Model 3 now."
    # Original golden is not mutated.
    assert ANSWER_INSTRUCTION not in (GOLDEN.turns or [])[-1].content


def test_bracket_instruction_tolerates_turnless_golden():
    g = ConversationalGolden(scenario="no turns", turns=[])
    assert bracket_instructed(g).turns == []


# -- scoring ----------------------------------------------------------------
def test_bracketed_correct_answer_scores(tmp_path):
    rep = _run("good", "{[{[Tesla Model 3]}]}", tmp_path, passage="drives a Tesla Model 3")
    assert rep.n == 1
    assert rep.accuracy == 1.0
    assert rep.results[0].correct is True
    assert rep.results[0].stale_leak is False


def test_wrong_answer_scores_zero(tmp_path):
    rep = _run("bad", "{[{[Honda Civic]}]}", tmp_path, passage="drives a Honda Civic")
    assert rep.accuracy == 0.0
    assert rep.results[0].correct is False
    # The superseded fact resurfaced — that is the leak column.
    assert rep.results[0].stale_leak is True


def test_unbracketed_answer_is_not_credited(tmp_path):
    """Deterministic scoring requires the bracket protocol."""
    rep = _run("sloppy", "You drive a Tesla Model 3", tmp_path)
    assert rep.results[0].correct is False
    # ...but recall F1 still records that the fact was present.
    assert rep.results[0].recall_f1 > 0.5


def test_overhead_is_measured(tmp_path):
    cheap = _run("cheap", "{[{[Tesla Model 3]}]}", tmp_path, overhead=5)
    dear = _run("dear", "{[{[Tesla Model 3]}]}", tmp_path, overhead=40)
    assert dear.mean_overhead > cheap.mean_overhead


# -- archive + ledger -------------------------------------------------------
def test_archive_is_written_and_content_hashed(tmp_path):
    rep = _run("arch", "{[{[Tesla Model 3]}]}", tmp_path)
    assert rep.archive_path is not None and rep.archive_path.exists()
    name = rep.archive_path.name
    assert "memorybench_arch_fake-model" in name
    assert "NYC_" in name and name.endswith(".jsonl")
    # Placeholder hash must have been replaced at finalization.
    assert "_000000000000." not in name

    rows = [json.loads(ln) for ln in rep.archive_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["provider"] == "arch"
    assert rows[0]["correct"] is True


def test_ledger_entry_appended(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHKIT_HOME", str(tmp_path / "bkhome"))
    from benchkit_for_harnesses.ledger import ledger_path, tail_entries

    _run("led", "{[{[Tesla Model 3]}]}", tmp_path)
    assert ledger_path().exists()
    entry = tail_entries(1)[-1]
    assert entry["cmd"] == "memorybench"
    assert entry["provider"] == "led"
    assert entry["n_items"] == 1
    assert entry["accuracy"] == 1.0


# -- failure discipline -----------------------------------------------------
def test_scenario_failure_becomes_a_record_not_an_exception(tmp_path):
    """One broken provider must not void the comparison."""

    async def _boom(prompt, session_id, peripheral):
        raise RuntimeError("backend not provisioned")

    rep = asyncio.run(
        run_provider_benchmark(
            goldens=[GOLDEN],
            adapter=_FakeProvider("broken"),
            turn_executor=_boom,
            workspace_dir=tmp_path / "broken",
            model="fake-model",
            output_dir=tmp_path / "archives",
        )
    )
    assert rep.n == 1  # honest n-count
    assert rep.n_errors == 1
    assert rep.accuracy == 0.0
    r = rep.results[0]
    assert r.correct is False
    assert r.response.startswith("[ERROR RuntimeError]")
    assert "backend not provisioned" in (r.error or "")


# -- ranking ----------------------------------------------------------------
def _report(provider, acc, leak, overhead) -> BenchmarkReport:
    rep = BenchmarkReport(provider=provider, model="m")
    rep.results = [
        ScenarioResult(
            idx=0, benchmark="b", scenario="s", provider=provider, model="m",
            target="t", response="r", correct=acc >= 1.0, latency_ms=1,
            recall_f1=acc, overhead_ratio=overhead, multi_hop=None,
            stale_leak=leak, n_sessions=3, n_retrievals=1,
            consolidation_wall_s=0.0, consolidation_bytes=0,
        )
    ]
    return rep


def test_rank_orders_by_accuracy_then_leak_then_overhead():
    a = _report("a", 1.0, False, 0.9)   # accurate, clean, expensive
    b = _report("b", 1.0, True, 0.1)    # accurate, leaks, cheap
    c = _report("c", 0.0, False, 0.1)   # inaccurate
    assert [r.provider for r in rank([c, b, a])] == ["a", "b", "c"]


def test_render_table_flags_incomparable_providers():
    good = _report("good", 1.0, False, 0.1)
    broken = _report("broken", 0.0, False, 0.0)
    broken.results[0].error = "boom"
    table = render_table([good, broken])
    assert "provider" in table and "acc" in table
    assert table.index("good") < table.index("broken")
    assert "not comparable" in table


def test_render_table_omits_note_when_all_clean():
    assert "not comparable" not in render_table([_report("a", 1.0, False, 0.1)])


def test_hedging_answer_is_rejected(tmp_path):
    """The belief-revision failure mode must not score as correct.

    BenchKit's loose mode falls back to substring containment when no brackets
    are present, so an answer naming BOTH the superseded and the current fact
    would be credited. Strict mode requires the model to commit.
    """
    hedge = "You used to drive a Honda Civic, but you might drive a Tesla Model 3 now."
    rep = _run("hedger", hedge, tmp_path, passage="previously drove a Honda Civic")
    assert rep.results[0].correct is False
    assert rep.accuracy == 0.0
    # The hedge did resurface the superseded fact.
    assert rep.results[0].stale_leak is True
