"""Cashew, driven against the real upstream engine — no fakes.

These tests are the point of the provisioning work: a mocked Cashew would
re-assert this repo's assumptions about Cashew rather than check them, and
every assumption in the adapter that turned out to be wrong (``CASHEW_CONFIG``,
an absolute ``database_path``, an ``offline`` flag, a runnable
``sleep_cron_script``) was wrong in exactly the way a mock would have
preserved.

They skip when the extra is not installed — see
:func:`memory_provider_evals.cashew_backend.cashew_installed` and
:data:`CASHEW_INSTALL_HINT`.

A note on runtime: the first test to initialize a provider downloads the
embedding model, into the shared cache
(:func:`memory_provider_evals.cashew_backend.shared_model_cache`) so it is paid
once per box rather than once per scenario. Each ``store`` forks an embedding
child, so these are seconds-per-test, not milliseconds.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.client import Client

from memory_provider_evals.adapters import CashewAdapter
from memory_provider_evals.cashew_backend import (
    CashewStore,
    cashew_config,
    cashew_db_path,
    cashew_home,
    cashew_installed,
    shared_model_cache,
)
from memory_provider_evals.mcp_server import memory_server_for

pytestmark = pytest.mark.skipif(
    not cashew_installed(),
    reason="cashew extra not installed (see cashew_backend.CASHEW_INSTALL_HINT)",
)


@pytest.fixture
def store(tmp_path):
    """A live Cashew under a throwaway home, shut down afterwards.

    Closing matters beyond tidiness: ``embedding_process`` keeps a module-global
    ``_OWNER`` and admits one supervisor per interpreter, so a provider left
    running would make the *next* test fail to start.
    """
    s = CashewStore.get(cashew_home(tmp_path))
    try:
        yield s
    finally:
        s.close()


# -- configuration ----------------------------------------------------------
def test_the_benchmark_pins_a_small_offline_model(monkeypatch):
    """Upstream's default is thenlper/gte-large — 1024 dimensions and ~1.3GB.
    Both are in UPSTREAM_KNOWN_DIMS, so either passes the dimension handshake;
    the small one is chosen so a cold box pays 90MB once."""
    monkeypatch.delenv("CASHEW_EMBEDDING_MODEL", raising=False)
    assert cashew_config()["embedding_model"] == "all-MiniLM-L6-v2"
    monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-base")
    assert cashew_config()["embedding_model"] == "thenlper/gte-base"


def test_the_config_this_repo_writes_is_one_upstream_accepts(tmp_path):
    """The old adapter wrote `database_path` (not a key) as an absolute path
    (rejected outright) plus `offline` (not a key). Round-tripping through
    upstream's own loader is what proves the replacement is not more of the
    same: load_config keeps only keys in DEFAULTS."""
    from plugins.memory.cashew.config import load_config

    home = cashew_home(tmp_path)
    CashewStore(home)  # construction alone writes nothing
    from memory_provider_evals.cashew_backend import write_cashew_config

    write_cashew_config(home)
    config = load_config(home)
    assert config.embedding_model == cashew_config()["embedding_model"]
    assert config.llm_aux_role is None
    assert config.sleep_schedule == ""
    # Relative, inside the home — resolve_db_path raises on anything else.
    assert config.cashew_db_path == "cashew/brain.db"
    assert cashew_db_path(home) == home / "cashew" / "brain.db"


def test_the_store_lands_where_the_adapter_declares_it(tmp_path, store):
    """store_paths is what consolidation byte-growth samples, so the engine's
    own path resolution has to be the one the adapter reports — and the model
    cache, which lives in the same directory, must not be in it."""
    declared = CashewAdapter().setup(tmp_path)["store_paths"]
    store.put("Gavin drives a Tesla Model 3.")
    db = cashew_db_path(cashew_home(tmp_path))
    assert db.exists()
    assert str(db) in declared
    assert not any("model-cache" in p for p in declared)


def test_the_model_cache_is_shared_rather_than_per_scenario(tmp_path, store):
    """EmbeddingSupervisor overwrites HF_HOME with <db dir>/model-cache and
    offers no knob, so without the symlink every scenario re-downloads the
    model into its own workspace."""
    store.put("Anything, to force initialization.")
    link = cashew_db_path(cashew_home(tmp_path)).parent / "model-cache"
    assert link.is_symlink()
    assert link.resolve() == shared_model_cache().resolve()


# -- the write path ---------------------------------------------------------
def test_capture_extracts_a_fact_from_a_turn_nobody_asked_to_store(store):
    """Cashew's headline behaviour, like Chronicle's: the agent never calls a
    write tool, and the fact is in memory anyway."""
    store.observe("My dog is called Biscuit.", "Noted — Biscuit it is.", "s1")
    recalled = " ".join(store.recall("dog name"))
    assert "Biscuit" in recalled


def test_observe_returns_only_after_the_turn_has_landed(store):
    """sync_turn is a <10ms enqueue onto a background worker. A benchmark
    cannot leave it there: the next session queries immediately, and a turn
    still in the queue would be scored as one the provider failed to
    remember."""
    store.observe("I moved to Lisbon last March.", "Got it.", "s1")
    # No sleep, no retry: the fact is readable on the very next call.
    assert any("Lisbon" in p for p in store.recall("where do I live"))
    work = store.provider.health_status()["work"]
    assert work["pending"] == 0 and work["in_flight"] == 0
    assert work["completed"] == work["accepted"]


def test_recall_splits_the_context_blob_into_one_passage_per_node(store):
    """Upstream renders retrieved nodes into a single `context` string. The
    harness counts retrievals, so handing it one passage would score a
    multi-node recall as a single memory."""
    store.observe("I row at Thames RC.", "Noted.", "s1")
    store.observe("My boat is a single scull.", "Noted.", "s1")
    passages = store.recall("rowing", limit=5)
    assert len(passages) > 1
    # Each keeps its node label, so the model can tell an extracted insight
    # from an observation of something said.
    assert all(p.startswith("[") for p in passages)
    assert not any(p.startswith("=== RELEVANT CONTEXT") for p in passages)


def test_recall_on_an_empty_store_is_empty_not_an_error(store):
    """A provider that found nothing must not look like one that failed."""
    assert store.recall("anything at all") == []


def test_put_stores_through_cashews_own_synchronous_tool(store):
    """cashew_extract runs the identical end_session call the sync worker
    runs — which is also why it is not exposed to the agent alongside implicit
    capture."""
    assert store.put("Gavin's cat is called Pepper.", tags=["Gavin"])
    assert any("Pepper" in p for p in store.recall("cat name"))


def test_removal_is_refused_rather_than_silently_reported_as_absent(store):
    """Cashew has no delete: a node leaves the graph by decaying through the
    sleep cycle's GC. Returning False would read as 'it was not there'."""
    with pytest.raises(NotImplementedError):
        store.remove("whatever")


# -- consolidation ----------------------------------------------------------
def test_consolidation_runs_the_real_sleep_cycle_in_process(store):
    """Cashew's offline pass is sleep_adapter.run_sleep_cycle — not
    `python -m plugins.memory.cashew.sleep_cron_script`, which is a template
    whose installation marker is only filled in by initialize()."""
    # Upstream's vectorized pipeline needs at least two embedded nodes: with
    # fewer it logs "too few valid embeddings" and returns `unavailable`. One
    # turn is genuinely not a consolidatable graph, so seed a few.
    store.observe("I work at Nous Research.", "Noted.", "s1")
    store.observe("My dog is called Biscuit, a corgi.", "Noted.", "s1")
    store.observe("I row at Thames RC on the tideway.", "Got it.", "s2")

    result = store.consolidate()
    assert result["status"] == "completed", result.get("error")
    assert result["nodes_selected"] > 1
    # The live provider's supervisor is handed over, so orphan repair is
    # available rather than skipped as it is with embedding_client=None.
    assert result["nodes_with_embeddings"] == result["nodes_selected"]
    assert result["orphan_write_failed"] == 0


def test_consolidating_a_single_turn_reports_unavailable_not_success(store):
    """The floor above, asserted directly: a one-node graph cannot be
    consolidated, and upstream says so rather than claiming a no-op cycle
    succeeded. Worth pinning — it is the difference between a provider with
    nothing to do and a provider that silently did nothing."""
    store.observe("I work at Nous Research.", "Noted.", "s1")
    assert store.consolidate()["status"] == "unavailable"


def test_the_sleep_cron_script_is_a_template_not_an_entry_point():
    """Guards the correction this task turned on: the module imports fine and
    still cannot run, installed or not, because its _INSTALLATION_MARKER is
    None until initialize() stages a copy with one."""
    from plugins.memory.cashew import sleep_cron_script

    assert sleep_cron_script._INSTALLATION_MARKER is None
    with pytest.raises(RuntimeError, match="installation marker"):
        sleep_cron_script._load_profile_modules(cashew_home("/nonexistent"))


# -- lifecycle --------------------------------------------------------------
def test_teardown_purges_the_home_the_harness_will_not(tmp_path):
    """SessionRunner.reset_suite only calls teardown; the provider owns its
    bytes. And the shut-down provider must leave the process-wide embedding
    owner free, or the next scenario cannot start at all."""
    adapter = CashewAdapter()
    adapter.setup(tmp_path)
    adapter.observe_turn("I drive a Honda Civic.", "Noted.", session_id="s1")
    adapter.trigger_consolidation()
    assert cashew_db_path(cashew_home(tmp_path)).exists()

    adapter.teardown()
    assert not cashew_home(tmp_path).exists()

    revived = CashewAdapter()
    revived.setup(tmp_path)
    try:
        revived.observe_turn("I drive a Tesla Model 3.", "Noted.", session_id="s2")
        recalled = " ".join(revived._cashew().recall("what do I drive"))
        assert "Tesla" in recalled
        assert "Civic" not in recalled
    finally:
        revived.teardown()


# -- what the agent can actually call ---------------------------------------
def test_the_agent_gets_cashew_query_over_mcp(tmp_path):
    """Through a real in-process MCP client — what create_agent receives."""
    adapter = CashewAdapter()
    adapter.setup(tmp_path)
    server, store = memory_server_for(adapter, tmp_path)
    assert isinstance(store, CashewStore)
    store.observe("My rowing club is Thames RC.", "Noted.", "s1")

    async def _run():
        async with Client(server) as client:
            listed = await client.list_tools()
            result = await client.call_tool("cashew_query", {"query": "rowing club"})
            return sorted(t.name for t in listed.tools), result

    try:
        names, result = asyncio.run(_run())
        # cashew_extract is upstream's second tool and is deliberately withheld.
        assert names == ["cashew_query"]
        text = "".join(getattr(b, "text", "") for b in result.content)
        # A JSON array, so several memories are counted as several retrievals
        # rather than flattened into one by the agent framework.
        passages = json.loads(text)
        assert isinstance(passages, list)
        assert any("Thames" in p for p in passages)
    finally:
        adapter.teardown()
