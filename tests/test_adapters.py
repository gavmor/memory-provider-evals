"""Tests for the concrete memory-provider adapters in dry_run mode."""

from __future__ import annotations

import json

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


def test_cashew_setup_consolidate_teardown(tmp_path):
    a = CashewAdapter(dry_run=True)
    info = a.setup(tmp_path)
    assert info["env"]["CASHEW_CONFIG"].endswith("cashew.json")
    assert info["contract"].tools == ["cashew_query"]
    assert any(str(p).endswith("brain.db") for p in info["store_paths"])
    # dry_run records the config but does not write it to disk
    assert "write_config" in _kinds(a)
    assert not (tmp_path / "cashew" / "cashew.json").exists()

    a.trigger_consolidation()
    exec_cmds = [d for k, d in a.actions if k == "exec"]
    assert exec_cmds[-1][-1].endswith("sleep_cron_script")

    a.teardown()
    assert "teardown" in _kinds(a)


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
    """Non-dry_run path actually materializes the sandboxed config file."""
    a = CashewAdapter(dry_run=False)
    info = a.setup(tmp_path)
    cfg = tmp_path / "cashew" / "cashew.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert data["database_path"].endswith("brain.db")
    a.teardown()
    assert not cfg.exists()  # teardown purges the sandbox
    assert info["contract"].provider == "cashew"
