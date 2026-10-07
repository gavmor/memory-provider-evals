"""Tests for the concrete memory-provider adapters in dry_run mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from traced_harness.memory import MemoryProviderAdapter

from memory_provider_evals.adapters import (
    CashewAdapter,
    ChronicleAdapter,
    Memex8Adapter,
    NachosAdapter,
)


def _kinds(adapter):
    return [a[0] for a in adapter.actions]


def test_abstract_adapter_cannot_instantiate():
    with pytest.raises(TypeError):
        MemoryProviderAdapter()  # type: ignore[abstract]


def test_cashew_sandboxes_a_hermes_home_and_runs_its_own_consolidation(tmp_path):
    """Upstream has no CASHEW_CONFIG and no runnable sleep_cron_script: the
    sandbox knob is HERMES_HOME (cashew.json is read from it, and
    cashew_db_path is resolved relative to it) and the pass is
    sleep_adapter.run_sleep_cycle in process. See CashewAdapter's docstring."""
    a = CashewAdapter(dry_run=True)
    info = a.setup(tmp_path)
    assert info["env"] == {"HERMES_HOME": str(tmp_path / "cashew_home")}
    assert info["contract"].tools == ["cashew_query"]
    assert any(str(p).endswith("brain.db") for p in info["store_paths"])
    # dry_run records the config it would write, and writes nothing.
    assert "init_store" in _kinds(a)
    assert not (tmp_path / "cashew_home" / "cashew.json").exists()

    a.trigger_consolidation()
    assert "consolidate" in _kinds(a)
    # No subprocess at all: `python -m plugins.memory.cashew.sleep_cron_script`
    # is a template that refuses to run outside the copy initialize() stages.
    assert not any(k == "exec" for k, _ in a.actions)

    a.teardown()
    assert "teardown" in _kinds(a)


def test_cashew_measures_its_store_but_not_its_model_cache(tmp_path):
    """The embedding child's HF_HOME is pinned inside the database's own
    directory, so store_paths names the SQLite files rather than the
    directory: otherwise a first run books a 90MB model download as
    consolidation growth."""
    info = CashewAdapter(dry_run=True).setup(tmp_path)
    paths = info["store_paths"]
    assert not any("model-cache" in p for p in paths)
    assert not any(p.endswith("/cashew") for p in paths)
    assert {Path(p).name for p in paths} == {
        "brain.db",
        "brain.db-wal",
        "brain.db-shm",
        "embedding-cache.db",
        "embedding-cache.db-wal",
        "embedding-cache.db-shm",
    }


def test_cashew_captures_turns_without_asking_the_agent_to_store(tmp_path):
    """Cashew's write path is sync_turn, not a tool the agent calls — so its
    contract names a read tool and nothing else, and every tool it names is a
    retrieval. cashew_extract is withheld deliberately: it runs the identical
    end_session call the sync worker runs, so exposing it alongside implicit
    capture would store every turn twice."""
    a = CashewAdapter(dry_run=True)
    contract = a.setup(tmp_path)["contract"]
    assert contract.recall_tools() == contract.tools
    assert "cashew_extract" not in contract.tools
    a.observe_turn("I drive a Tesla Model 3.", "Noted.", session_id="s1")
    assert "observe_turn" in _kinds(a)


def test_chronicle_sandboxes_a_hermes_home_and_runs_its_own_consolidation(tmp_path):
    """Upstream has no CHRONICLE_DB/CHRONICLE_VECTORS, and no consolidation
    script: the sandbox knob is HERMES_HOME and the pass is the curation
    queue. See ChronicleAdapter's docstring for what the three `scripts/*.py`
    an earlier spec named actually do."""
    a = ChronicleAdapter(dry_run=True)
    info = a.setup(tmp_path)
    assert info["contract"].tools == ["chronicle_search"]
    assert info["env"] == {"HERMES_HOME": str(tmp_path / "chronicle")}
    # Vectors live inside the database, so there is one store path, and it is
    # the directory holding the db plus its sidecars.
    assert info["store_paths"] == [
        str(tmp_path / "chronicle" / "commons" / "db" / "chronicle")
    ]

    a.trigger_consolidation()
    assert "consolidate" in _kinds(a)
    # No subprocess at all: the pass runs in process against this run's store.
    assert not any(k == "exec" for k, _ in a.actions)


def test_chronicle_captures_turns_without_asking_the_agent_to_store(tmp_path):
    """Chronicle's write path is capture, not a tool — so its contract names a
    read tool and nothing else, and every tool it names is a retrieval."""
    a = ChronicleAdapter(dry_run=True)
    contract = a.setup(tmp_path)["contract"]
    assert contract.recall_tools() == contract.tools
    a.observe_turn("I drive a Tesla Model 3.", "Noted.", session_id="s1")
    assert "observe_turn" in _kinds(a)


def test_memex8_compose_and_slumber(tmp_path):
    a = Memex8Adapter(dry_run=True, api_key="k")
    a.setup(tmp_path)
    up = next(d for k, d in a.actions if k == "exec")
    assert up[:2] == ["docker", "compose"] and "up" in up

    a.trigger_consolidation()
    posts = [d for k, d in a.actions if k == "http_post"]
    assert posts[-1]["url"].endswith("/api/v1/slumber")

    a.teardown()
    down = [d for k, d in a.actions if k == "exec"][-1]
    assert "down" in down and "-v" in down


def test_nachos_is_text_only_inline_consolidation(tmp_path):
    a = NachosAdapter(dry_run=True)
    info = a.setup(tmp_path)
    assert "nachos_memory_recall" in info["contract"].tools
    # No visual/image hooks — text-only.
    assert set(info["contract"].context_hooks) == {
        "nachos_manifest",
        "nachos_prefetch",
    }
    # Default has no offline sleep routine: consolidation is inline.
    a.trigger_consolidation()
    assert "noop_inline_compaction" in _kinds(a)
    assert not any(k == "exec" for k, _ in a.actions)


def test_nachos_explicit_compaction_cmd_runs(tmp_path):
    a = NachosAdapter(dry_run=True, compaction_cmd=["python", "-m", "nachos_core"])
    a.setup(tmp_path)
    a.trigger_consolidation()
    assert any(k == "exec" for k, _ in a.actions)


def test_cashew_live_mode_writes_config(tmp_path):
    """Non-dry_run path materializes the sandboxed config, through upstream's
    own validating writer — so a key this repo invents fails here instead of
    being silently dropped at load time, which is how `offline: true` survived
    in the spec for as long as it did."""
    a = CashewAdapter(dry_run=False)
    info = a.setup(tmp_path)
    cfg = tmp_path / "cashew_home" / "cashew.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    # Relative to hermes_home, by upstream's profile-isolation rule.
    assert data["cashew_db_path"] == "cashew/brain.db"
    assert "database_path" not in data and "offline" not in data
    # Heuristic-only: the default role would bind the run to whatever
    # auxiliary model the box's Hermes profile happens to name.
    assert data["llm_aux_role"] is None
    # Sleep is enabled but not *scheduled* — consolidation is driven between
    # sessions so it lands inside the measured span.
    assert data["sleep_cycles"] is True and data["sleep_schedule"] == ""
    a.teardown()
    assert not cfg.exists()  # teardown purges the sandbox
    assert info["contract"].provider == "cashew"
