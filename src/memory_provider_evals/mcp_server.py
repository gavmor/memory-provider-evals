"""The MCP server that gives the agent working memory tools.

``traced_harness.agent.create_agent`` registers a provider's tool *names* and
injects its system-prompt contract, but the implementations arrive over the
MCP ``client``. Without a server there is no ``cashew_query`` to call, so the
agent can only answer from the context window — which the multi-session design
deliberately empties between sessions. Every provider then scores identically
badly, for a reason that has nothing to do with its memory.

This module closes that gap: it turns a
:class:`~traced_harness.memory.MemoryToolContract` into a live
:class:`~mcp.server.mcpserver.MCPServer`, registering one tool per contract
name, backed by a :class:`~memory_provider_evals.memory_store.MemoryStore`.

The server is consumed **in process**: ``mcp.client.Client`` accepts a server
object directly, which is also what ``traced_harness.client.parse_mcp_target``
reaches via ``--server 'module:attr'``. No subprocess, no port, and the store
stays inspectable from the test that drove it.

What each tool name does
------------------------
Tool names come from the provider, so the operation each one performs is
declared explicitly in :data:`TOOL_OPERATIONS` rather than guessed from the
string. An unmapped name is an error at build time, not a tool that silently
does the wrong thing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from traced_harness.memory import MemoryProviderAdapter, MemoryToolContract

from memory_provider_evals.cashew_backend import (
    CASHEW_INSTALL_HINT,
    CashewStore,
    cashew_home,
    cashew_installed,
)
from memory_provider_evals.chronicle_backend import (
    CHRONICLE_CLONE_HINT,
    ChronicleStore,
    chronicle_home,
    chronicle_repo_root,
)
from memory_provider_evals.memory_store import (
    BackendNotProvisioned,
    LexicalMemoryStore,
    MemoryStore,
    UnprovisionedStore,
)

__all__ = [
    "LOCAL_STORE_BUILDERS",
    "LOCAL_STORE_PROVIDERS",
    "PROVISIONING_HINTS",
    "TOOL_OPERATIONS",
    "build_memory_server",
    "memory_server_for",
    "store_for",
]

#: Contract tool name -> the operation it performs. Every tool named by an
#: adapter in :mod:`memory_provider_evals.adapters` appears here.
TOOL_OPERATIONS: dict[str, str] = {
    # Cashew — magnus919/hermes-cashew
    "cashew_query": "recall",
    # Chronicle — indigokarasu/chronicle-agent-context-and-memory
    "chronicle_search": "recall",
    # Memex8 — Ex8-ca/memex8
    "memex8_search": "recall",
    # Nachos — Nacho-Labs-LLC/hermes-plugin-nachos
    "nachos_memory_recall": "recall",
    "nachos_memory_put": "put",
    "nachos_memory_remove": "remove",
}

#: How to provision each provider's real backend, quoted back to the operator
#: when a tool is called without one.
PROVISIONING_HINTS: dict[str, str] = {
    "cashew": CASHEW_INSTALL_HINT,
    "chronicle": CHRONICLE_CLONE_HINT,
    "memex8": (
        "Run `docker compose up -d qdrant memex8` from Ex8-ca/memex8 and set "
        "MEMEX8_URL / MEMEX8_API_KEY."
    ),
}


def _nachos_store(
    adapter: MemoryProviderAdapter, workspace_dir: str | Path
) -> MemoryStore:
    """Nachos is text-only, local-first, ``scorer="lexical"`` — so a SQLite
    corpus with a lexical scorer *is* the provider, not a stand-in for it."""
    root = Path(workspace_dir) / "nachos_home" / "nachos"
    return LexicalMemoryStore(root / "memories.db")


def _cashew_store(
    adapter: MemoryProviderAdapter, workspace_dir: str | Path
) -> MemoryStore:
    """The real Cashew provider, under an ephemeral ``HERMES_HOME``.

    Cashew is a distribution rather than a service, so "provisioned" means the
    ``cashew`` extra is installed; without it the store is unprovisioned and
    says how to get it. The store is fetched from the per-home registry rather
    than constructed, because the adapter driving capture and consolidation
    must share this one provider — upstream's embedding supervisor admits
    exactly one owner per interpreter.
    """
    if not cashew_installed():
        return UnprovisionedStore(adapter.name, PROVISIONING_HINTS["cashew"])
    return CashewStore.get(cashew_home(workspace_dir))


def _chronicle_store(
    adapter: MemoryProviderAdapter, workspace_dir: str | Path
) -> MemoryStore:
    """The real Chronicle engine, off a checkout, under an ephemeral home.

    Chronicle is a stdlib-only local-first plugin rather than a service, so
    "provisioned" means a clone is on disk; without one the store is
    unprovisioned and says how to get it.
    """
    root = chronicle_repo_root(getattr(adapter, "repo_root", None))
    if root is None:
        return UnprovisionedStore(adapter.name, PROVISIONING_HINTS["chronicle"])
    return ChronicleStore(chronicle_home(workspace_dir), repo_root=root)


#: Providers this repo can answer locally, and what answers them. Everything
#: absent from here is a separate service that must be running.
LOCAL_STORE_BUILDERS: dict[str, Callable[[MemoryProviderAdapter, str | Path], MemoryStore]] = {
    "nachos": _nachos_store,
    "cashew": _cashew_store,
    "chronicle": _chronicle_store,
}

#: Provider ids with a local store. Kept as a name because it reads better at
#: call sites than ``in LOCAL_STORE_BUILDERS``.
LOCAL_STORE_PROVIDERS = frozenset(LOCAL_STORE_BUILDERS)


def store_for(
    adapter: MemoryProviderAdapter,
    workspace_dir: str | Path,
) -> MemoryStore:
    """The store backing ``adapter``'s tools for a run under ``workspace_dir``.

    The path mirrors where the adapter's own ``setup()`` puts its state, so
    the bytes this store writes are inside the paths the harness samples for
    consolidation growth. It is resolved rather than created here: ``setup()``
    runs later, when the session runner starts the scenario.
    """
    builder = LOCAL_STORE_BUILDERS.get(adapter.name)
    if builder is not None:
        return builder(adapter, workspace_dir)
    return UnprovisionedStore(
        adapter.name, PROVISIONING_HINTS.get(adapter.name, "")
    )


@contextmanager
def _reported(provider: str) -> Iterator[None]:
    """Turn an absent backend into a tool error the caller can read.

    MCP wraps an unexpected exception as a bare "Error executing tool X",
    losing the reason. A missing backend is not a crash — it is a fact about
    this machine that the operator needs stated, and that the benchmark
    records in its ``error`` column.
    """
    try:
        yield
    except BackendNotProvisioned as exc:
        raise ToolError(str(exc)) from exc


def build_memory_server(
    contract: MemoryToolContract,
    store: MemoryStore,
    name: str | None = None,
) -> MCPServer:
    """An MCP server exposing exactly the tools ``contract`` names.

    Tool docstrings are what the model reads when deciding whether to call,
    so they state the provider and the operation rather than describing the
    store's implementation. Recall answers with a JSON array rather than a
    list of content blocks: the harness parses the text back into passages,
    and several memories returned as separate blocks would be flattened into
    one by the agent framework and counted as a single retrieval.
    """

    unknown = [t for t in contract.tools if t not in TOOL_OPERATIONS]
    if unknown:
        raise ValueError(
            f"{contract.provider}: no operation declared for tool(s) "
            f"{sorted(unknown)}. Add them to TOOL_OPERATIONS — a memory tool "
            "whose behaviour is guessed from its name is worse than no tool."
        )

    provider = contract.provider
    server = MCPServer(name or f"{provider}-memory")

    def _register_recall(tool_name: str) -> None:
        async def _recall(query: str, limit: int = 5) -> str:
            with _reported(provider):
                return json.dumps(list(store.recall(query, limit)))

        _recall.__name__ = tool_name
        _recall.__doc__ = (
            f"Recall durable memories from {provider} that are relevant to "
            "`query`. Returns a JSON array of the stored entries, most "
            "relevant first; `[]` means nothing on this topic was ever "
            "stored."
        )
        server.tool(name=tool_name)(_recall)

    def _register_put(tool_name: str) -> None:
        async def _put(text: str, tags: list[str] | None = None) -> str:
            with _reported(provider):
                return store.put(text, tags)

        _put.__name__ = tool_name
        _put.__doc__ = (
            f"Store a durable memory in {provider}. Pass one self-contained "
            "fact as `text` — it must still make sense in a later session "
            "with no conversation history. Returns the new entry's id."
        )
        server.tool(name=tool_name)(_put)

    def _register_remove(tool_name: str) -> None:
        async def _remove(entry_id: str) -> bool:
            with _reported(provider):
                return store.remove(entry_id)

        _remove.__name__ = tool_name
        _remove.__doc__ = (
            f"Delete one durable memory from {provider} by its entry id. "
            "Returns whether an entry was deleted."
        )
        server.tool(name=tool_name)(_remove)

    registrars: dict[str, Any] = {
        "recall": _register_recall,
        "put": _register_put,
        "remove": _register_remove,
    }
    for tool_name in contract.tools:
        registrars[TOOL_OPERATIONS[tool_name]](tool_name)

    return server


def memory_server_for(
    adapter: MemoryProviderAdapter,
    workspace_dir: str | Path,
) -> tuple[MCPServer, MemoryStore]:
    """Build the server and store for one provider under one workspace.

    Returns both so a caller can assert against what the agent actually
    stored, rather than inferring it from the answer text.
    """
    store = store_for(adapter, workspace_dir)
    return build_memory_server(adapter.contract(), store), store
