"""Cashew's real backend, driven through Cashew's own entry points.

Upstream: `magnus919/hermes-cashew
<https://github.com/magnus919/hermes-cashew>`_ — a Hermes memory provider over
the ``cashew-brain`` thought graph: a SQLite node/edge store, sentence-
transformers embeddings behind an owned child process, ``sqlite-vec`` recall,
and a "sleep" cycle that cross-links and consolidates the graph offline.

Unlike Chronicle, this one is a *distribution*: ``pip install hermes-cashew``
publishes the ``plugins.memory.cashew`` namespace package and pulls in
``cashew-brain`` (which publishes top-level ``core``, ``extractors``,
``integration`` and ``scripts``), ``torch`` and ``sentence-transformers``.
That is several GB and a namespace landgrab, so it is an **optional extra**
here (``uv sync --extra cashew``) and this module reports a clean
"not provisioned" when it is absent — see :func:`cashew_installed`.

Spec corrections, all verified against the upstream checkout
------------------------------------------------------------
Every one of these was wrong in the adapter this module replaces, and none of
them would have been caught by a mocked backend.

``CASHEW_CONFIG`` does not exist.
    Nothing upstream reads that name. ``config.resolve_config_path`` is
    ``$HERMES_HOME/cashew.json``, full stop, so the knob that sandboxes a run
    is ``HERMES_HOME`` — the same knob Chronicle and Nachos use. Upstream does
    define per-key overrides, but they are ``CASHEW_<FIELD>``
    (``CASHEW_DB_PATH``, ``CASHEW_EMBEDDING_MODEL``, ...) applied *on top of*
    the JSON by ``load_config``.

``database_path`` is not a config key, and may not be absolute.
    The key is ``cashew_db_path`` and ``config.resolve_db_path`` **raises** on
    an absolute path, deliberately, to keep a profile's store inside its home.
    The adapter used to write an absolute ``database_path``: an unknown key
    pointing at a path upstream would have rejected.

``offline: true`` is not a config key and never was.
    ``load_config`` keeps only the keys in ``config.DEFAULTS``, so it was
    silently dropped. The real lever is the standard ``HF_HUB_OFFLINE``, which
    ``EmbeddingSupervisor._child_environment`` does forward to the embedding
    child — but see :func:`shared_model_cache`: it would only *help* if the
    model were already cached, and the supervisor pins ``HF_HOME`` *inside the
    scenario's own ephemeral home*, where nothing ever is. So an offline flag
    could not have saved the download; a shared cache can, and does.

``python -m plugins.memory.cashew.sleep_cron_script`` always fails.
    Not for want of installing it — that module exists. It is a *template*:
    its ``_INSTALLATION_MARKER`` is ``None`` in the source tree and is
    substituted by ``cron_reconcile.render_script`` when ``initialize()``
    stages a copy into ``$HERMES_HOME/scripts/``. Run from the package it
    raises "Cashew cron script has no installation marker" before it does any
    work. The consolidation pass itself is ``sleep_adapter.run_sleep_cycle``,
    which is what :meth:`CashewStore.consolidate` calls in process.

Capture is implicit, as it is for Chronicle.
    ``sync_turn(user, assistant, session_id)`` is what a Hermes host calls
    after every turn; with ``auto_extraction`` on it hands the turn to
    ``core.session.end_session``. The agent is not asked to remember.

Two tools, one exposed
----------------------
``get_tool_schemas`` offers ``cashew_query`` (recall) and ``cashew_extract``
(store one turn, synchronously). Only ``cashew_query`` is given to the agent.
``cashew_extract`` runs *the identical* ``end_session`` call the sync worker
runs, so exposing it alongside implicit capture would store every turn twice
and inflate the graph the recall score is computed over. It is still this
module's write path for seeding (:meth:`CashewStore.put`) precisely because it
is the synchronous one.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar

from memory_provider_evals.memory_store import BackendNotProvisioned

__all__ = [
    "CASHEW_INSTALL_HINT",
    "CASHEW_MODEL_CACHE_ENV",
    "OFFLINE_EMBEDDING_MODEL",
    "CashewStore",
    "cashew_config",
    "cashew_db_path",
    "cashew_home",
    "cashew_installed",
    "cashew_store_paths",
    "shared_model_cache",
]

CASHEW_INSTALL_HINT = (
    "Provision it with: uv sync --extra cashew (installs magnus919/"
    "hermes-cashew, cashew-brain, torch and sentence-transformers — several "
    "GB, which is why it is not a default dependency)."
)

#: Points the per-scenario embedding model cache at one shared directory.
CASHEW_MODEL_CACHE_ENV = "CASHEW_MODEL_CACHE"

#: 384 dimensions, ~90MB. Upstream's default is ``thenlper/gte-large`` (1024d,
#: ~1.3GB); both are in ``embedding_compat.UPSTREAM_KNOWN_DIMS``, so either
#: passes the dimension handshake. The small one is chosen so a cold box pays
#: 90MB once instead of 1.3GB, and ``$CASHEW_EMBEDDING_MODEL`` — upstream's own
#: override, applied by ``load_config`` — opts back up. The cost is honest and
#: belongs in any result: MiniLM is a weaker semantic tier than gte-large, so
#: Cashew's vector channel is measured below its ceiling.
OFFLINE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: How long to wait for the sync worker to finish the turns it accepted.
#: Extraction is heuristic (no LLM round-trip) but does embed, so it is
#: seconds, not milliseconds.
CAPTURE_TIMEOUT_SECONDS = 180.0


def cashew_installed() -> bool:
    """Whether this box can actually run Cashew.

    Both halves have to be importable: ``plugins.memory.cashew`` is the Hermes
    provider and ``core.context`` is the ``cashew-brain`` engine underneath it.
    The provider module deliberately imports with either one missing (so that
    plugin discovery works on a box that has not installed the backend), and
    its own ``is_available()`` makes the same distinction.
    """
    from importlib.util import find_spec

    try:
        return (
            find_spec("plugins.memory.cashew") is not None
            and find_spec("core.context") is not None
        )
    except (ImportError, ValueError):
        return False


def cashew_home(workspace_dir: str | Path) -> Path:
    """The ephemeral ``HERMES_HOME`` Cashew runs under for one scenario.

    There is no ``CASHEW_CONFIG``: ``cashew.json`` is read from this directory
    and ``cashew_db_path`` is resolved *relative to* it, with absolute paths
    rejected outright. So this one directory is the whole sandbox, and the
    operator's real ``~/.hermes`` is never touched.
    """
    return Path(workspace_dir) / "cashew_home"


def cashew_db_path(hermes_home: str | Path) -> Path:
    """Where the thought graph lands under ``hermes_home``.

    Mirrors ``config.resolve_db_path(home, DEFAULTS["cashew_db_path"])``; the
    default is kept so the layout on disk is the one a real profile has.
    """
    return Path(hermes_home) / "cashew" / "brain.db"


def cashew_store_paths(hermes_home: str | Path) -> list[str]:
    """Cashew's store, as the files consolidation growth should be sampled over.

    Named file by file rather than as a directory, because the directory holds
    one thing that is emphatically not the store: ``model-cache/`` is where the
    embedding child's ``HF_HOME`` points, and a first run would otherwise
    record a 90MB model download as consolidation writing 90MB of memory.

    Both SQLite databases are listed with their write-ahead sidecars. Paths
    that do not exist contribute zero bytes, so listing all six is safe before
    anything has been written.
    """
    db = cashew_db_path(hermes_home)
    cache = db.parent / "embedding-cache.db"
    return [
        str(p)
        for base in (db, cache)
        for p in (base, base.with_name(base.name + "-wal"), base.with_name(base.name + "-shm"))
    ]


def shared_model_cache() -> Path:
    """One embedding-model cache for every scenario on this box.

    ``EmbeddingSupervisor._child_environment`` *overwrites* ``HF_HOME`` with
    ``<db dir>/model-cache/huggingface`` and offers no knob for it, so each
    scenario's fresh home would re-download the model — 90MB and a couple of
    minutes, per scenario, inside the measured workspace. :meth:`CashewStore.
    _link_model_cache` symlinks that directory here instead, which is the only
    intervention point short of patching upstream.
    """
    override = os.environ.get(CASHEW_MODEL_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    cache_root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(cache_root) / "memory-provider-evals" / "cashew-models"


def cashew_config() -> dict[str, Any]:
    """Provider config for a sandboxed, reproducible benchmark run.

    Only the three keys that differ from ``config.DEFAULTS`` are set; the rest
    are left alone so the run is upstream's own behaviour.

    ``llm_aux_role: None``
        The default, ``"memory"``, resolves a *Hermes host* auxiliary model
        through ``hermes_cli.config`` and uses it for LLM extraction, think
        cycles and dream synthesis. On a box with a Hermes profile configured
        that would bind the benchmark to whatever model that profile happens to
        name — an unrelated process deciding the number. ``None`` selects
        upstream's heuristic extractor, which is deterministic and free.

    ``embedding_model``
        See :data:`OFFLINE_EMBEDDING_MODEL`.

    ``sleep_schedule: ""``
        Empty disables cron registration: ``initialize()`` would otherwise
        stage a generated script into ``$HERMES_HOME/scripts`` and register a
        recurring Hermes cron job for it. Consolidation here is driven
        explicitly between sessions so that it lands inside the measured span;
        a scheduler doing the same work on its own clock would land outside it.
        ``sleep_cycles`` stays ``True`` — sleep is enabled, it just is not
        *scheduled*.
    """
    return {
        "embedding_model": os.environ.get(
            "CASHEW_EMBEDDING_MODEL", OFFLINE_EMBEDDING_MODEL
        ),
        "llm_aux_role": None,
        "sleep_schedule": "",
    }


def write_cashew_config(hermes_home: str | Path) -> Path:
    """Write ``$HERMES_HOME/cashew.json`` through upstream's own writer.

    ``config.save_config`` validates every field and fills the rest from
    ``DEFAULTS``, so a key this repo gets wrong fails here rather than being
    silently dropped at load time — which is exactly how ``offline: true``
    survived in the spec for as long as it did.
    """
    home = Path(hermes_home)
    home.mkdir(parents=True, exist_ok=True)
    from plugins.memory.cashew.config import save_config

    return Path(save_config(cashew_config(), home))


class CashewStore:
    """A :class:`~memory_provider_evals.memory_store.MemoryStore` over Cashew.

    One store per scenario home, and *shared* between the adapter (which drives
    capture and consolidation) and the MCP server (which answers the agent's
    ``cashew_query``) via :meth:`get`. The sharing is not a convenience:
    ``embedding_process`` keeps a module-global ``_OWNER`` and a second live
    ``EmbeddingSupervisor`` in one interpreter raises
    ``EmbeddingUnavailable(OWNED)``. One provider per process is upstream's
    rule, so one provider per home is this module's.

    The provider is built lazily. The MCP server is wired before
    ``SessionRunner.run_scenario`` calls the adapter's ``setup()``, and
    ``initialize()`` creates databases, downloads a model and forks a child.
    """

    #: Cashew's agent-facing read tool, and the one this store answers with.
    RECALL_TOOL = "cashew_query"

    #: Upstream's synchronous write tool. Not exposed to the agent (see the
    #: module docstring); used by :meth:`put` to seed a store in tests.
    EXTRACT_TOOL = "cashew_extract"

    #: One live provider per home, shared between the adapter and the MCP
    #: server. ClassVar deliberately: the registry is the process's, not an
    #: instance's — see the class docstring on upstream's single-owner rule.
    _live: ClassVar[dict[Path, CashewStore]] = {}

    def __init__(
        self,
        hermes_home: str | Path,
        session_id: str = "memorybench",
        capture_timeout: float = CAPTURE_TIMEOUT_SECONDS,
    ) -> None:
        self.hermes_home = Path(hermes_home)
        self.session_id = session_id
        self.capture_timeout = capture_timeout
        self._provider: Any = None

    @classmethod
    def get(cls, hermes_home: str | Path, **kwargs: Any) -> CashewStore:
        """The store for ``hermes_home``, created once per process."""
        key = Path(hermes_home).resolve()
        store = cls._live.get(key)
        if store is None:
            store = cls(key, **kwargs)
            cls._live[key] = store
        return store

    # -- engine -----------------------------------------------------------
    def _link_model_cache(self) -> None:
        """Point this scenario's forced ``HF_HOME`` at the shared cache.

        A symlink rather than a copy: it is the model cache's *location* the
        supervisor fixes, not its contents, and :func:`cashew_store_paths`
        excludes it from the measured bytes either way.
        """
        shared = shared_model_cache()
        shared.mkdir(parents=True, exist_ok=True)
        link = cashew_db_path(self.hermes_home).parent / "model-cache"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            return
        link.symlink_to(shared, target_is_directory=True)

    @property
    def provider(self) -> Any:
        """The live ``CashewMemoryProvider``, initialized on first use."""
        if self._provider is None:
            if not cashew_installed():
                raise BackendNotProvisioned(
                    "cashew: hermes-cashew / cashew-brain are not importable. "
                    + CASHEW_INSTALL_HINT
                )
            from plugins.memory.cashew import CashewMemoryProvider

            write_cashew_config(self.hermes_home)
            self._link_model_cache()
            provider = CashewMemoryProvider()
            provider.initialize(self.session_id, hermes_home=str(self.hermes_home))
            # initialize() degrades silently by contract: a failed model load or
            # a lost lease leaves a provider that accepts calls and quietly
            # stores nothing. That would be scored as a provider that remembers
            # badly, when the truth is that it never started.
            health = provider.health_status()
            if health.get("state") != "ready":
                reason = health.get("reason_code") or health.get("state")
                raise BackendNotProvisioned(
                    f"cashew: provider initialized to state {health.get('state')!r} "
                    f"(reason {reason!r}) instead of 'ready'; it would accept "
                    "calls and store nothing."
                )
            self._provider = provider
        return self._provider

    def _call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call one Cashew tool exactly as a Hermes host would.

        ``handle_tool_call`` is the provider's real tool surface, and by
        contract it never raises and never returns ``None`` — failure comes
        back as ``{"ok": false, "error": ...}``, which this turns into an
        exception so a broken backend cannot read as an empty memory.
        """
        payload = json.loads(self.provider.handle_tool_call(tool, args))
        if not payload.get("ok"):
            raise RuntimeError(
                f"cashew {tool}: {payload.get('error', 'unknown error')}"
            )
        return dict(payload)

    # -- the MemoryStore protocol -----------------------------------------
    def recall(self, query: str, limit: int = 5) -> list[str]:
        """``cashew_query``, split back into one passage per graph node.

        Upstream renders the retrieved nodes into a single ``context`` blob —
        a ``=== RELEVANT CONTEXT ===`` header and then one
        ``[domain: … | type: …] content`` line per node
        (``CashewMemoryProvider._format_context``). The harness counts
        retrievals, so handing it one passage would score a five-node recall as
        a single memory. Each node's label is kept: it is how the model tells a
        consolidated insight from an observation of something said.

        A line that does not open a new label is a continuation of the node
        above it, since a node's content may itself contain newlines.
        """
        payload = self._call(self.RECALL_TOOL, {"query": query, "max_nodes": limit})
        passages: list[str] = []
        for line in str(payload.get("context") or "").splitlines():
            if not line.strip() or line.startswith("==="):
                continue
            if line.startswith("[") or not passages:
                passages.append(line)
            else:
                passages[-1] += "\n" + line
        return passages

    def put(self, text: str, tags: list[str] | None = None) -> str:
        """Store one memory via ``cashew_extract``; returns the node count.

        Present because the store protocol has a write, and used by tests to
        seed a store — it is not exposed to the agent (see the module
        docstring). Cashew has no "store this fact" entry point at all: every
        write goes through ``end_session`` over a turn, so ``text`` is passed
        as what the user said, which is what it is.
        """
        entry = str(text).strip()
        if not entry:
            raise ValueError("refusing to store an empty memory")
        if tags:
            entry = f"{entry} ({', '.join(str(t) for t in tags)})"
        payload = self._call(
            self.EXTRACT_TOOL,
            {"user_content": entry, "assistant_content": "Noted."},
        )
        return str(payload.get("new_nodes", 0))

    def remove(self, entry_id: str) -> bool:
        """Unsupported, and that is a fact about Cashew rather than about this.

        There is no delete in ``get_tool_schemas`` and none in the engine: a
        Cashew node leaves the graph by decaying through the sleep cycle's GC,
        never by being asked to. Raising here keeps that visible; returning
        ``False`` would read as "the entry was not there".
        """
        raise NotImplementedError(
            "cashew exposes no removal entry point — nodes leave the graph via "
            "sleep-cycle decay, not deletion."
        )

    # -- Cashew's own write and consolidation paths ------------------------
    def observe(
        self, user_content: str, assistant_content: str, session_id: str = ""
    ) -> None:
        """Hand one finished turn to Cashew, and wait for it to land.

        ``sync_turn`` is Cashew's real write path — what a Hermes host calls
        after every turn — but it is deliberately a <10ms enqueue onto a
        background worker. A benchmark cannot leave it there: the next session
        queries immediately, and a turn still in the queue would be scored as a
        turn the provider failed to remember.

        So the enqueue is followed by a barrier on the provider's own public
        work ledger (``health_status()["work"]``), which reports queued and
        in-flight turns separately from terminal ones. Anything that is not a
        clean completion — a drop, a failure, a timeout — is raised rather than
        absorbed, for the same reason: a provider that silently stopped
        recording must not be reported as one that remembers nothing.
        """
        before = self.provider.health_status()["work"]
        self.provider.sync_turn(
            str(user_content), str(assistant_content), session_id or self.session_id
        )
        after = self.provider.health_status()["work"]
        if after["accepted"] == before["accepted"]:
            raise RuntimeError(
                "cashew sync_turn did not accept the turn — the provider is in "
                f"a half-state (work={after})."
            )
        self._await_capture(expect_completed=before["completed"] + 1)

    def _await_capture(self, expect_completed: int) -> None:
        """Block until the sync worker has drained what it accepted."""
        deadline = time.monotonic() + self.capture_timeout
        work = self.provider.health_status()["work"]
        while work["pending"] or work["in_flight"]:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"cashew capture did not drain within {self.capture_timeout}s "
                    f"(work={work})."
                )
            time.sleep(0.05)
            work = self.provider.health_status()["work"]
        if work["completed"] < expect_completed:
            raise RuntimeError(
                "cashew dropped or failed a turn instead of capturing it "
                f"(work={work})."
            )

    def consolidate(self) -> dict[str, Any]:
        """Cashew's offline pass: ``sleep_adapter.run_sleep_cycle``, in process.

        Not ``python -m plugins.memory.cashew.sleep_cron_script`` — see the
        module docstring for why that module cannot run outside the copy
        ``initialize()`` stages.

        The live provider's embedding supervisor is handed over rather than a
        fresh one. That reaches past a private attribute, which is a cost worth
        naming, but ``embedding_process`` admits exactly one supervisor per
        interpreter and the provider holds it; the only alternative is
        ``embedding_client=None``, which makes upstream skip orphan embedding
        and measures a weaker cycle than the one Cashew ships. If the attribute
        ever goes away the cycle still runs, just without orphan repair.
        """
        from plugins.memory.cashew.sleep_adapter import run_sleep_cycle

        provider = self.provider
        config = provider._config
        return dict(
            run_sleep_cycle(
                db_path=str(cashew_db_path(self.hermes_home)),
                limit=config.sleep_max_nodes,
                embedding_model=config.embedding_model,
                embedding_client=getattr(provider, "_embedding_supervisor", None),
            )
        )

    def close(self) -> None:
        """Shut the provider down and release the process-wide embedding owner.

        Teardown deletes the home; a provider left running would hold open
        SQLite handles on a deleted database and, worse, keep
        ``embedding_process._OWNER``, so the *next* scenario's provider would
        fail to start with ``EmbeddingUnavailable(OWNED)``.
        """
        if self._provider is not None:
            self._provider.shutdown()
            self._provider = None
        self._live.pop(self.hermes_home.resolve(), None)
