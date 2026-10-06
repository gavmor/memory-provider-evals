"""Chronicle's real backend, driven through Chronicle's own entry points.

Upstream: `indigokarasu/chronicle-agent-context-and-memory
<https://github.com/indigokarasu/chronicle-agent-context-and-memory>`_ — an
event-sourced memory provider for Hermes: SQLite event log, belief extraction
with provenance, FTS + vector retrieval, and a curation queue that does the
offline work between turns.

It is a Hermes *plugin*, not a package on PyPI, and its maintenance scripts
live in ``scripts/`` rather than in the installed distribution. So this module
works from a checkout (see :func:`chronicle_repo_root`) and imports
``engine.core`` off it, which is exactly how upstream's own scripts bootstrap
(``sys.path.insert(0, $CHRONICLE_DIR)``).

What is wired, and why
----------------------
Chronicle's headline behaviour is that the *agent is not responsible for
remembering*: ``CaptureEngine.observe(user, assistant)`` appends every turn to
the event log and extracts beliefs from it. That is the write path wired here
(:meth:`ChronicleStore.observe`), driven per turn by
:mod:`memory_provider_evals.live`.

The agent therefore gets one tool, ``chronicle_search`` — Chronicle's real
both-tiers read (beliefs *and* what was said). ``chronicle_remember`` is
deliberately **not** exposed: the generic MCP put signature carries only
``(text, tags)`` and would land every agent write as ``kind="note"``, routing
around the fact/supersession path that capture exercises — a worse write path
than the one Chronicle actually ships, offered to the agent as if it were the
provider.

Offline consolidation is ``Scheduler.on_hook`` + ``CurationWorker.drain``
(:meth:`ChronicleStore.consolidate`), i.e. what ``ChronicleCore.tick`` does per
turn. See :meth:`memory_provider_evals.adapters.ChronicleAdapter.
trigger_consolidation` for why the three ``scripts/*.py`` an earlier spec named
are not that.

Embeddings
----------
Default ``hashing``: upstream's deterministic offline embedder. The default
(``auto``) opens TCP connections to LM Studio / Ollama / llama.cpp ports
*inside the core constructor* and binds the run to whatever happens to be
listening — a benchmark number that depends on an unrelated process on the box
is not comparable to one taken anywhere else. ``$CHRONICLE_EMBED_MODEL`` opts
back in. The cost is honest and worth stating in any result: feature-hashed
vectors are a weaker semantic tier than a real embedding model, so Chronicle's
vector channel is measured at its offline floor, not its ceiling.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from memory_provider_evals.memory_store import BackendNotProvisioned

__all__ = [
    "CHRONICLE_CLONE_HINT",
    "CHRONICLE_REPO_ENV",
    "ChronicleStore",
    "chronicle_config",
    "chronicle_db_path",
    "chronicle_home",
    "chronicle_repo_root",
]

#: Points at a Chronicle checkout anywhere on the box.
CHRONICLE_REPO_ENV = "CHRONICLE_REPO"

#: Overrides the embedder. Unset means :data:`OFFLINE_EMBEDDER`.
CHRONICLE_EMBED_MODEL_ENV = "CHRONICLE_EMBED_MODEL"

OFFLINE_EMBEDDER = "hashing"
OFFLINE_EMBEDDER_DIMENSIONS = 256

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Checkout locations tried in order after ``$CHRONICLE_REPO``: this repo's
#: gitignored ``vendor/`` (what the hint below creates) and the directory
#: ``hermes plugins install`` clones into.
DEFAULT_CHECKOUTS = (
    _REPO_ROOT / "vendor" / "chronicle",
    Path.home() / ".hermes" / "plugins" / "chronicle-agent-context-and-memory",
)

CHRONICLE_CLONE_HINT = (
    "Provision it with: git clone "
    "https://github.com/indigokarasu/chronicle-agent-context-and-memory.git "
    "vendor/chronicle (or point $CHRONICLE_REPO at an existing checkout, or "
    "install it as a Hermes plugin: hermes plugins install "
    "indigokarasu/chronicle-agent-context-and-memory)."
)


def chronicle_repo_root(explicit: str | Path | None = None) -> Path | None:
    """Locate a Chronicle checkout, or ``None`` if this box has none.

    A directory counts only when ``engine/core.py`` is in it: Chronicle is
    imported off the checkout, so an empty or half-cloned directory must read
    as "not provisioned" rather than fail later with an ImportError whose text
    says nothing about provisioning.
    """
    candidates = [explicit, os.environ.get(CHRONICLE_REPO_ENV), *DEFAULT_CHECKOUTS]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if (root / "engine" / "core.py").is_file():
            return root.resolve()
    return None


def chronicle_home(workspace_dir: str | Path) -> Path:
    """The ephemeral ``HERMES_HOME`` Chronicle runs under for one scenario.

    Chronicle has no ``CHRONICLE_DB`` / ``CHRONICLE_VECTORS``: ``ChronicleCore``
    derives its database from ``hermes_home`` and keeps vectors in that same
    database (``observed_vectors`` plus an optional sqlite-vec index). So the
    one knob that sandboxes a run is the home directory, and that is what the
    adapter sets and tears down.
    """
    return Path(workspace_dir) / "chronicle"


def chronicle_db_path(hermes_home: str | Path) -> Path:
    """Where ``ChronicleCore`` puts its store under ``hermes_home``.

    Mirrors the constructor's own resolution, which ignores a configured
    ``db_path`` unless it is written relative to ``~/.hermes``.
    """
    return Path(hermes_home) / "commons" / "db" / "chronicle" / "chronicle.db"


def chronicle_config() -> dict[str, Any]:
    """Engine config for a sandboxed, reproducible benchmark run."""
    return {
        "embeddings": {
            "model": os.environ.get(CHRONICLE_EMBED_MODEL_ENV, OFFLINE_EMBEDDER),
            "dimensions": OFFLINE_EMBEDDER_DIMENSIONS,
        },
        # The git mirror is a second, out-of-store copy of the event log that
        # shells out to git. Nothing here reads it, and its writes would land
        # in the byte-growth consolidation samples as noise.
        "git": {"enabled": False},
        # Consolidation is measured between sessions. A background drain thread
        # would move that work off the measured span and could still be running
        # when teardown closes the store.
        "curation": {"drain": {"background": False}},
    }


def _import_chronicle_core(repo_root: Path) -> Any:
    """Import ``ChronicleCore`` off ``repo_root``.

    The checkout goes on ``sys.path`` the way upstream's own scripts do. This
    publishes a top-level ``engine`` package for the process; nothing else in
    this repo claims that name, and the alternative (installing the
    distribution) publishes ``provider``, ``context`` and ``_base`` too.
    """
    path = str(repo_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    from engine.core import ChronicleCore

    return ChronicleCore


class ChronicleStore:
    """A :class:`~memory_provider_evals.memory_store.MemoryStore` over Chronicle.

    One store per scenario workspace. The underlying ``ChronicleCore`` is a
    process-wide singleton keyed by ``hermes_home``, so the adapter (which
    drives capture and consolidation) and this store (which answers the
    agent's tool calls) share one engine without being handed to each other,
    while two scenarios under different workspaces stay isolated.

    The core is built lazily: the MCP server is wired before
    ``SessionRunner.run_scenario`` calls the adapter's ``setup()``, and
    constructing a core creates its database.
    """

    #: Chronicle's agent-facing read tool, and the one this store answers with.
    RECALL_TOOL = "chronicle_search"

    def __init__(
        self,
        hermes_home: str | Path,
        repo_root: str | Path | None = None,
        principal: str = "default",
        session_id: str = "memorybench",
    ) -> None:
        self.hermes_home = Path(hermes_home)
        self.repo_root = repo_root
        self.principal = principal
        self.session_id = session_id
        self._core: Any = None

    # -- engine -----------------------------------------------------------
    @property
    def core(self) -> Any:
        """The live ``ChronicleCore``, built on first use."""
        if self._core is None:
            root = chronicle_repo_root(self.repo_root)
            if root is None:
                raise BackendNotProvisioned(
                    "chronicle: no checkout found. " + CHRONICLE_CLONE_HINT
                )
            core_cls = _import_chronicle_core(root)
            self._core = core_cls.get(str(self.hermes_home), chronicle_config())
            self._core.initialize(
                self.session_id,
                hermes_home=str(self.hermes_home),
                principal_id=self.principal,
            )
        return self._core

    def _dispatch(self, tool: str, args: dict[str, Any]) -> Any:
        """Call one Chronicle tool exactly as a Hermes host would.

        ``Tools.dispatch`` is what ``ChronicleMemoryProvider.handle_tool_call``
        calls, so this is the provider's real tool surface rather than a
        shortcut into the engine behind it. It answers with a JSON string and
        reports failure as ``{"error": ...}`` instead of raising.
        """
        payload = json.loads(self.core.tools.dispatch(self.principal, tool, args))
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"chronicle {tool}: {payload['error']}")
        return payload

    # -- the MemoryStore protocol -----------------------------------------
    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Both of ``chronicle_search``'s tiers, flattened for the agent.

        Beliefs first, then the transcript lines they were extracted from.
        Both are returned because both are what Chronicle holds: on a store
        built from conversation, the transcript is most of the memory, and
        dropping it would measure half the provider. Each entry is labelled so
        the model can tell a consolidated belief from a thing someone said.

        ``limit`` is passed through unchanged; upstream bounds each tier by it.
        """
        payload = self._dispatch(self.RECALL_TOOL, {"query": query, "limit": limit})
        passages: list[str] = []
        for belief in payload.get("results") or []:
            value = str(belief.get("value") or "").strip()
            if not value:
                continue
            entity = str(belief.get("entity_id") or "").strip()
            attribute = str(belief.get("attribute") or "").strip()
            subject = f"{entity} {attribute}: " if entity and attribute else ""
            passages.append(f"[belief] {subject}{value}")
        for said in payload.get("said") or []:
            excerpt = str(said.get("excerpt") or "").strip()
            if not excerpt:
                continue
            when = str(said.get("when") or "").strip()
            label = f"[said {when}]" if when else "[said]"
            passages.append(f"{label} {excerpt}")
        # Order-preserving dedupe: a belief and the turn it came from can
        # render to the same text, and two passages would be counted as two
        # retrieved memories when there is one.
        seen: set[str] = set()
        unique = []
        for passage in passages:
            if passage in seen:
                continue
            seen.add(passage)
            unique.append(passage)
        return unique

    def put(self, text: str, tags: list[str] | None = None) -> str:
        """Store one memory via ``chronicle_remember``; returns the event id.

        Present because the store protocol has a write, and used by tests to
        seed a store. It is *not* exposed to the agent — see the module
        docstring. ``tags`` names the subject, which is what Chronicle's
        ``entity`` argument means.
        """
        entry = str(text).strip()
        if not entry:
            raise ValueError("refusing to store an empty memory")
        args: dict[str, Any] = {"kind": "note", "content": entry}
        if tags:
            args["entity"] = str(tags[0])
        return str(self._dispatch("chronicle_remember", args).get("event", ""))

    def remove(self, entry_id: str) -> bool:
        """Retract a belief by id via ``chronicle_forget``.

        Not exposed to the agent either: Chronicle's answer to a superseded
        fact is a *new* assertion that supersedes it, recorded with both
        values and a contradiction linking them — forgetting the old one would
        destroy the provenance the benchmark's stale-leak column reads.
        """
        result = self._dispatch("chronicle_forget", {"belief_id": entry_id})
        return bool(result.get("status"))

    # -- Chronicle's own write and consolidation paths ---------------------
    def observe(self, user_content: str, assistant_content: str, session_id: str = "") -> str:
        """Append one turn to the event log and extract beliefs from it.

        This is Chronicle's actual write path — ``CaptureEngine.observe``, what
        ``ChronicleMemoryProvider.sync_turn`` calls on every Hermes turn. The
        agent is never asked to store anything.
        """
        return str(
            self.core.capture.observe(
                str(user_content),
                str(assistant_content),
                session_id=session_id or self.session_id,
            )
        )

    def consolidate(self, max_jobs: int = 1000) -> int:
        """Chronicle's offline pass: schedule what is due, then drain the queue.

        ``ChronicleCore.tick`` drains *before* scheduling, so a job it enqueues
        waits for a later turn — deliberately, to keep maintenance out of the
        user's turn. Between sessions there is no user's turn to protect and
        nothing after this call to pick the job up, so the order is inverted:
        schedule, then drain, so a sweep that came due actually runs inside the
        measured consolidation span.

        Returns the number of curation jobs executed.
        """
        self.core.scheduler.on_hook("session_end")
        return int(self.core.process_pending(max_jobs))

    def close(self) -> None:
        """Release the SQLite store and drop the process singleton.

        Teardown deletes the home directory; a cached core holding an open
        connection to a deleted database would be handed to the next
        ``ChronicleCore.get()`` for the same path.
        """
        if self._core is not None:
            self._core.close()
            self._core = None
