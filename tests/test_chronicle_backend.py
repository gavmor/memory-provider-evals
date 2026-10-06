"""Chronicle, driven against the real upstream engine — no fakes.

These tests are the point of the provisioning work: a mocked Chronicle would
re-assert this repo's assumptions about Chronicle rather than check them, and
every assumption in the adapter that turned out to be wrong (``CHRONICLE_DB``,
a vectors directory, three consolidation scripts) was wrong in exactly the way
a mock would have preserved.

They skip when no checkout is on the box — see
:func:`memory_provider_evals.chronicle_backend.chronicle_repo_root` for where
one is looked for and :data:`CHRONICLE_CLONE_HINT` for how to make one.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.client import Client

from memory_provider_evals.adapters import ChronicleAdapter
from memory_provider_evals.chronicle_backend import (
    ChronicleStore,
    chronicle_config,
    chronicle_db_path,
    chronicle_home,
    chronicle_repo_root,
)
from memory_provider_evals.mcp_server import memory_server_for

pytestmark = pytest.mark.skipif(
    chronicle_repo_root() is None,
    reason="no Chronicle checkout (see chronicle_backend.CHRONICLE_CLONE_HINT)",
)


@pytest.fixture
def store(tmp_path):
    """A live Chronicle under a throwaway home, closed afterwards.

    Closing matters beyond tidiness: ``ChronicleCore`` caches one instance per
    home for the process, so a core left open would be handed to the next
    caller for the same path.
    """
    s = ChronicleStore(chronicle_home(tmp_path))
    try:
        yield s
    finally:
        s.close()


# -- configuration ----------------------------------------------------------
def test_the_benchmark_pins_an_offline_embedder(monkeypatch):
    """Chronicle's default (`auto`) probes LM Studio / Ollama / llama.cpp ports
    inside the core constructor, which would bind the measurement to whatever
    happens to be listening on the machine."""
    monkeypatch.delenv("CHRONICLE_EMBED_MODEL", raising=False)
    assert chronicle_config()["embeddings"]["model"] == "hashing"
    monkeypatch.setenv("CHRONICLE_EMBED_MODEL", "nomic-embed-text")
    assert chronicle_config()["embeddings"]["model"] == "nomic-embed-text"


def test_the_store_lands_where_the_adapter_declares_it(tmp_path):
    """store_paths is what consolidation byte-growth samples, so the engine's
    own path resolution has to be the one the adapter reports."""
    adapter = ChronicleAdapter()
    declared = adapter.setup(tmp_path)["store_paths"]
    store = ChronicleStore(chronicle_home(tmp_path))
    try:
        store.put("Gavin drives a Tesla Model 3.")
        db = chronicle_db_path(store.core.hermes_home)
        assert db.exists()
        assert str(db.parent) in declared
    finally:
        store.close()


# -- the two write paths ----------------------------------------------------
def test_capture_extracts_a_fact_from_a_turn_nobody_asked_to_store(store):
    """Chronicle's headline behaviour: the agent never calls a write tool, and
    the fact is in memory anyway."""
    store.observe("My dog is called Biscuit.", "Noted — Biscuit it is.", "s1")
    store.consolidate()
    recalled = " ".join(store.recall("dog name"))
    assert "Biscuit" in recalled


def test_recall_returns_both_beliefs_and_what_was_said(store):
    """On a store built from conversation the transcript is most of the
    memory; returning beliefs alone would measure half the provider."""
    store.observe("I moved to Lisbon last March.", "Got it.", "s1")
    store.consolidate()
    passages = store.recall("Lisbon", limit=5)
    assert any(p.startswith("[belief]") for p in passages)
    assert any(p.startswith("[said") for p in passages)
    assert len(set(passages)) == len(passages)  # deduped


def test_put_stores_through_chronicles_own_tool(store):
    assert store.put("Gavin's cat is called Pepper.", tags=["Gavin"])
    assert any("Pepper" in p for p in store.recall("cat name"))


def test_recall_on_an_empty_store_is_empty_not_an_error(store):
    """A provider that found nothing must not look like one that failed."""
    assert store.recall("anything at all") == []


# -- consolidation ----------------------------------------------------------
def test_consolidation_drains_the_curation_queue(store):
    """Chronicle's offline pass is the curation queue plus the maintenance
    scheduler — not the three `scripts/*.py` an earlier spec named."""
    store.observe("I work at Nous Research.", "Noted.", "s1")
    assert store.consolidate() > 0
    # Idempotent: a second pass with nothing queued does nothing.
    assert store.consolidate() == 0


# -- lifecycle --------------------------------------------------------------
def test_teardown_purges_the_home_the_harness_will_not(tmp_path):
    """SessionRunner.reset_suite only calls teardown; the provider owns its
    bytes."""
    adapter = ChronicleAdapter()
    adapter.setup(tmp_path)
    adapter.observe_turn("I drive a Honda Civic.", "Noted.", session_id="s1")
    adapter.trigger_consolidation()
    assert chronicle_db_path(chronicle_home(tmp_path)).exists()

    adapter.teardown()
    assert not chronicle_home(tmp_path).exists()

    # And the closed core is gone from the process cache, so a later run under
    # the same path gets a fresh store rather than a handle to a deleted file.
    revived = ChronicleAdapter()
    revived.setup(tmp_path)
    try:
        revived.observe_turn("I drive a Tesla Model 3.", "Noted.", session_id="s2")
        assert any("Tesla" in p for p in revived._chronicle().recall("what do I drive"))
        assert not any("Civic" in p for p in revived._chronicle().recall("what do I drive"))
    finally:
        revived.teardown()


# -- what the agent can actually call ---------------------------------------
def test_the_agent_gets_chronicle_search_over_mcp(tmp_path):
    """Through a real in-process MCP client — what create_agent receives."""
    adapter = ChronicleAdapter()
    adapter.setup(tmp_path)
    server, store = memory_server_for(adapter, tmp_path)
    assert isinstance(store, ChronicleStore)
    store.observe("My rowing club is Thames RC.", "Noted.", "s1")
    store.consolidate()

    async def _run():
        async with Client(server) as client:
            listed = await client.list_tools()
            result = await client.call_tool(
                "chronicle_search", {"query": "rowing club"}
            )
            return sorted(t.name for t in listed.tools), result

    try:
        names, result = asyncio.run(_run())
        assert names == ["chronicle_search"]
        text = "".join(getattr(b, "text", "") for b in result.content)
        # A JSON array, so several memories are counted as several retrievals
        # rather than flattened into one by the agent framework.
        passages = json.loads(text)
        assert isinstance(passages, list)
        assert any("Thames" in p for p in passages)
    finally:
        store.close()
        adapter.teardown()
