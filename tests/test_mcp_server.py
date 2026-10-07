"""The MCP server that gives the agent working memory tools.

These tests drive the server through a real in-process ``mcp.client.Client``
— the same object ``create_agent(client=...)`` receives — so what is asserted
is what the agent would actually be able to call, not a direct call to the
store behind it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from traced_harness.memory import MemoryProviderAdapter, MemoryToolContract

from memory_provider_evals import chronicle_backend, mcp_server
from memory_provider_evals.adapters import (
    CashewAdapter,
    ChronicleAdapter,
    Memex8Adapter,
    NachosAdapter,
)
from memory_provider_evals.cashew_backend import CashewStore, cashew_installed
from memory_provider_evals.chronicle_backend import (
    ChronicleStore,
    chronicle_repo_root,
)
from memory_provider_evals.mcp_server import (
    TOOL_OPERATIONS,
    build_memory_server,
    memory_server_for,
    store_for,
)
from memory_provider_evals.memory_store import (
    LexicalMemoryStore,
    UnprovisionedStore,
)

ALL_ADAPTERS = [CashewAdapter, ChronicleAdapter, Memex8Adapter, NachosAdapter]


def _text(result: Any) -> str:
    """Flatten an MCP call result's content blocks to plain text."""
    return "".join(getattr(block, "text", "") for block in result.content)


def _call(server, tool: str, args: dict[str, Any]) -> Any:
    async def _run():
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    return asyncio.run(_run())


def _tool_names(server) -> list[str]:
    async def _run():
        async with Client(server) as client:
            listed = await client.list_tools()
            return sorted(t.name for t in listed.tools)

    return asyncio.run(_run())


# -- contract coverage ------------------------------------------------------
@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS)
def test_every_adapters_tools_have_a_declared_operation(adapter_cls):
    """A tool whose behaviour is guessed from its name is worse than none."""
    contract = adapter_cls(dry_run=True).contract()
    assert contract.tools
    for tool in contract.tools:
        assert tool in TOOL_OPERATIONS, tool


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS)
def test_the_server_exposes_exactly_the_contract_tools(adapter_cls, tmp_path):
    adapter = adapter_cls(dry_run=True)
    server, _store = memory_server_for(adapter, tmp_path)
    assert _tool_names(server) == sorted(adapter.contract().tools)


def test_an_undeclared_tool_name_is_refused_at_build_time():
    contract = MemoryToolContract(provider="x", tools=["telepathy"])
    with pytest.raises(ValueError, match="telepathy"):
        build_memory_server(contract, LexicalMemoryStore("unused.db"))


@pytest.mark.parametrize("adapter_cls", ALL_ADAPTERS)
def test_only_reading_tools_are_declared_as_retrievals(adapter_cls):
    """`n_retrievals` proves the provider was consulted; a write is not that."""
    contract = adapter_cls(dry_run=True).contract()
    for tool in contract.recall_tools():
        assert TOOL_OPERATIONS[tool] == "recall", tool
    # ...and every reading tool the provider offers is declared.
    readers = {t for t in contract.tools if TOOL_OPERATIONS[t] == "recall"}
    assert set(contract.recall_tools()) == readers


# -- store selection --------------------------------------------------------
def test_nachos_gets_a_real_local_store_under_its_own_workspace(tmp_path):
    store = store_for(NachosAdapter(dry_run=True), tmp_path)
    assert isinstance(store, LexicalMemoryStore)
    # Inside the dir NachosAdapter.setup() creates and declares in
    # store_paths, so consolidation byte-growth measures these bytes.
    assert store.db_path.parent == tmp_path / "nachos_home" / "nachos"


def test_service_backed_providers_get_an_unprovisioned_store(tmp_path):
    store = store_for(Memex8Adapter(dry_run=True), tmp_path)
    assert isinstance(store, UnprovisionedStore)
    assert store.provisioning_hint


def test_cashew_gets_the_real_provider_when_the_extra_is_installed(tmp_path):
    """Cashew is a distribution, not a service: installing the extra is the
    whole provisioning step, so a provisioned run talks to the real engine."""
    if not cashew_installed():
        pytest.skip("cashew extra not installed on this machine")
    store = store_for(CashewAdapter(dry_run=True), tmp_path)
    assert isinstance(store, CashewStore)
    # The same ephemeral home the adapter's setup() creates, so the bytes this
    # store writes land inside the declared store_paths.
    assert store.hermes_home == (tmp_path / "cashew_home").resolve()


def test_cashew_without_the_extra_names_the_install(tmp_path, monkeypatch):
    """An absent backend has to say which absent backend, and how to get it."""
    monkeypatch.setattr(mcp_server, "cashew_installed", lambda: False)
    store = store_for(CashewAdapter(dry_run=True), tmp_path)
    assert isinstance(store, UnprovisionedStore)
    assert "--extra cashew" in store.provisioning_hint


def test_chronicle_gets_the_real_engine_when_a_checkout_is_present(tmp_path):
    """Chronicle is a local-first plugin, not a service: a clone is the whole
    provisioning step, so a provisioned run talks to the real engine."""
    if chronicle_repo_root() is None:
        pytest.skip("no Chronicle checkout on this machine")
    store = store_for(ChronicleAdapter(dry_run=True), tmp_path)
    assert isinstance(store, ChronicleStore)
    # The same ephemeral home the adapter's setup() creates, so the bytes this
    # store writes land inside the declared store_paths.
    assert store.hermes_home == tmp_path / "chronicle"


def test_chronicle_without_a_checkout_names_the_clone(tmp_path, monkeypatch):
    """An absent backend has to say which absent backend, and how to get it."""
    monkeypatch.setenv("CHRONICLE_REPO", str(tmp_path / "nowhere"))
    monkeypatch.setattr(
        chronicle_backend, "DEFAULT_CHECKOUTS", (tmp_path / "also-nowhere",)
    )
    store = store_for(ChronicleAdapter(dry_run=True), tmp_path)
    assert isinstance(store, UnprovisionedStore)
    assert "git clone" in store.provisioning_hint


def test_calling_an_unprovisioned_tool_surfaces_the_provisioning_step(tmp_path):
    adapter = Memex8Adapter(dry_run=True)
    server, _store = memory_server_for(adapter, tmp_path)
    result = _call(server, "memex8_search", {"query": "what do I drive"})
    assert result.is_error
    assert "not provisioned" in _text(result)
    assert "docker compose" in _text(result)


# -- the tools the agent actually calls -------------------------------------
def test_put_then_recall_over_mcp(tmp_path):
    adapter = NachosAdapter(dry_run=True)
    server, store = memory_server_for(adapter, tmp_path)

    async def _run():
        async with Client(server) as client:
            await client.call_tool(
                "nachos_memory_put", {"text": "Gavin drives a Honda Civic."}
            )
            return await client.call_tool(
                "nachos_memory_recall", {"query": "what does Gavin drive"}
            )

    recalled = _text(asyncio.run(_run()))
    assert "Honda Civic" in recalled
    # ...and it really went to the provider's store, not just an echo.
    assert store.all_entries() == ["Gavin drives a Honda Civic."]


def test_recall_returns_a_json_list_the_harness_can_count(tmp_path):
    """`traced_harness.memory._passages` parses the text back into passages.

    A result that flattened several memories into one blob would be counted
    as a single retrieval regardless of how much was recalled.
    """
    adapter = NachosAdapter(dry_run=True)
    server, store = memory_server_for(adapter, tmp_path)
    store.put("Gavin drives a Honda Civic.")
    store.put("Gavin drives to Berlin every Friday.")

    payload = json.loads(
        _text(_call(server, "nachos_memory_recall", {"query": "Gavin drives"}))
    )
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_remove_over_mcp(tmp_path):
    adapter = NachosAdapter(dry_run=True)
    server, store = memory_server_for(adapter, tmp_path)
    stale = store.put("Gavin drives a Honda Civic.")
    store.put("Gavin drives a Tesla Model 3.")

    _call(server, "nachos_memory_remove", {"entry_id": stale})
    assert store.all_entries() == ["Gavin drives a Tesla Model 3."]


def test_recalling_nothing_is_an_empty_list_not_a_fabrication(tmp_path):
    adapter = NachosAdapter(dry_run=True)
    server, _store = memory_server_for(adapter, tmp_path)
    assert json.loads(
        _text(_call(server, "nachos_memory_recall", {"query": "anything at all"}))
    ) == []


# -- the seam into the harness ----------------------------------------------
def test_the_server_satisfies_the_harness_mcp_target_contract(tmp_path):
    """`connect_mcp(server=...)` must accept what this module builds."""
    from traced_harness.client import parse_mcp_target

    server, _store = memory_server_for(NachosAdapter(dry_run=True), tmp_path)
    target, label = parse_mcp_target(server=server)
    assert target is server
    assert label.startswith("in-process:")


def test_a_contract_with_no_tools_yields_a_server_with_no_tools(tmp_path):
    class _Toolless(MemoryProviderAdapter):
        name = "toolless"

        def setup(self, workspace_dir: Path) -> dict[str, Any]:
            return {}

        def trigger_consolidation(self) -> None: ...

        def teardown(self) -> None: ...

        def contract(self) -> MemoryToolContract:
            return MemoryToolContract(provider=self.name, tools=[])

    server, _store = memory_server_for(_Toolless(dry_run=True), tmp_path)
    assert _tool_names(server) == []
