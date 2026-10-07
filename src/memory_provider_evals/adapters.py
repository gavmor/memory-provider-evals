"""Concrete Hermes memory-provider adapters: the subjects under study.

Each adapter wraps a real, third-party Hermes memory plugin and implements the
``MemoryProviderAdapter`` lifecycle contract supplied by ``traced_harness``:

* ``CashewAdapter``    -> magnus919/hermes-cashew (Cashew thought-graph, SQLite)
* ``ChronicleAdapter`` -> indigokarasu/chronicle-agent-context-and-memory
* ``Memex8Adapter``    -> Ex8-ca/memex8 (Rust daemon + Qdrant via docker compose)
* ``NachosAdapter``    -> Nacho-Labs-LLC/hermes-plugin-nachos (text context engine)

These live in the study repo rather than the harness: the harness defines what
a memory peripheral *is* (the ABC, telemetry, and prompt wiring), while *which*
providers are being benchmarked is a property of this particular study.

Fidelity / verification note
----------------------------
The adapters are written against each project's *documented, real* entry points
(verified against the upstream repositories, not invented):

* Cashew's consolidation is ``sleep_adapter.run_sleep_cycle``, run in process.
  ``plugins/memory/cashew/sleep_cron_script.py`` exists but is the *template*
  for a generated cron script and refuses to run from the package -- see
  :class:`CashewAdapter`.
* Chronicle's consolidation is its curation queue + maintenance scheduler, run
  in process (``ChronicleCore.tick``'s two halves). The ``scripts/*.py`` named
  by earlier specs exist but are research/repair tools, not a consolidation
  pass -- see :class:`ChronicleAdapter`.
* Memex8 runs ``qdrant`` + ``memex8`` via ``docker compose`` and exposes a REST
  API on :8080; its consolidation ("Slumber") pipeline has **13** phases.
* Nachos is text-only (manifest/prefetch/recall); it has no offline "sleep"
  pass -- consolidation is inline compaction + snapshotting.

Cashew and Chronicle are the two provisioned here, and both have been exercised
against their real engines: Chronicle is stdlib-only and runs off a checkout
(:mod:`memory_provider_evals.chronicle_backend`), Cashew installs as an extra
(``uv sync --extra cashew``, :mod:`memory_provider_evals.cashew_backend`).
Memex8 is a separate service and has **not** been exercised end-to-end here.
Every adapter supports ``dry_run=True``, which records the exact command / URL
/ config it *would* execute (available as ``.actions``) without touching an
external process -- this is what the unit tests assert, and what you can use to
smoke-test wiring before a real backend is provisioned.
"""

from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path
from typing import Any

from traced_harness.memory import MemoryProviderAdapter, MemoryToolContract

from memory_provider_evals.cashew_backend import (
    CashewStore,
    cashew_config,
    cashew_db_path,
    cashew_home,
    cashew_store_paths,
    write_cashew_config,
)
from memory_provider_evals.chronicle_backend import (
    ChronicleStore,
    chronicle_db_path,
    chronicle_home,
)

__all__ = [
    "ADAPTERS",
    "CashewAdapter",
    "ChronicleAdapter",
    "Memex8Adapter",
    "NachosAdapter",
    "build_adapter",
    "default_provider",
]

# ---------------------------------------------------------------------------
# Cashew — magnus919/hermes-cashew
# ---------------------------------------------------------------------------
class CashewAdapter(MemoryProviderAdapter):
    """Cashew thought-graph memory, run in process off an installed backend.

    ``setup`` sandboxes Cashew in an ephemeral ``HERMES_HOME`` under the
    scenario workspace; :meth:`observe_turn` is its real per-turn capture;
    :meth:`trigger_consolidation` is its real sleep pass; ``teardown`` shuts
    the provider down and deletes the home.

    Spec corrections, all verified against the upstream checkout rather than
    inferred from the plugin's prose. :mod:`memory_provider_evals.
    cashew_backend` carries the evidence for each; in brief:

    ``CASHEW_CONFIG`` does not exist.
        ``cashew.json`` is read from ``$HERMES_HOME``, and ``cashew_db_path``
        is resolved relative to it — absolute paths are rejected. So the
        sandbox knob is ``HERMES_HOME``, as it is for Chronicle and Nachos,
        and the config this adapter used to write (an absolute
        ``database_path``, plus an ``offline`` flag that is not a key at all)
        was discarded by ``load_config`` on the way in.

    ``plugins.memory.cashew.sleep_cron_script`` is a template, not an entry point.
        The module is real, but its ``_INSTALLATION_MARKER`` is filled in when
        ``initialize()`` stages a copy into ``$HERMES_HOME/scripts``. Run as
        ``python -m`` it refuses before doing any work, installed or not. The
        consolidation pass is ``sleep_adapter.run_sleep_cycle``, run here in
        process against this run's own store.

    Capture is implicit.
        Cashew does not ask the agent to remember: ``sync_turn`` hands every
        turn to the extractor. The agent gets ``cashew_query`` and nothing
        else — see :class:`~memory_provider_evals.cashew_backend.CashewStore`
        for why ``cashew_extract`` is withheld.
    """

    name = "cashew"

    def __init__(
        self,
        home_env_var: str = "HERMES_HOME",
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.home_env_var = home_env_var
        self.hermes_home: Path | None = None
        self.db_path: Path | None = None
        self._store: CashewStore | None = None

    def _cashew(self) -> CashewStore:
        """The store sharing this run's provider (a singleton keyed by home)."""
        if self.hermes_home is None:
            raise RuntimeError("CashewAdapter.setup() has not run yet.")
        if self._store is None:
            self._store = CashewStore.get(self.hermes_home)
        return self._store

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        self.hermes_home = cashew_home(workspace_dir)
        self.db_path = cashew_db_path(self.hermes_home)
        self._record(
            "init_store",
            {
                "hermes_home": str(self.hermes_home),
                "db": str(self.db_path),
                **cashew_config(),
            },
        )
        if not self.dry_run:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            write_cashew_config(self.hermes_home)
        # The two SQLite stores and their sidecars, named individually: the
        # directory also holds the embedding model cache, and sampling that
        # would book a 90MB model download as consolidation growth.
        self.store_paths = cashew_store_paths(self.hermes_home)
        return {
            "env": {self.home_env_var: str(self.hermes_home)},
            "contract": self.contract(),
            "store_paths": self.store_paths,
        }

    def observe_turn(
        self, user_content: str, assistant_content: str, session_id: str = ""
    ) -> None:
        """Capture one turn — Cashew's write path, not the agent's.

        Driven by :mod:`memory_provider_evals.live` after every turn, which is
        where Hermes itself calls ``sync_turn``.
        """
        self._record("observe_turn", {"session_id": session_id})
        if self.dry_run:
            return
        self._cashew().observe(user_content, assistant_content, session_id)

    def trigger_consolidation(self) -> None:
        self._record("consolidate", self.name)
        if self.dry_run:
            return
        self._record("sleep_cycle", self._cashew().consolidate())

    def teardown(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        if self.hermes_home and not self.dry_run:
            self._purge(self.hermes_home)
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["cashew_query"],
            # Upstream names these as the Hermes host hooks it implements, not
            # as tools the agent can invoke: ``prefetch`` warms context before
            # a call and ``on_pre_compress`` turns a window about to be
            # discarded into insight nodes. Neither is wired here — the
            # harness has no pre-LLM injection point and no window to compress.
            context_hooks=["prefetch", "on_pre_compress"],
            system_prompt=(
                "Durable memory is provided by Cashew, which records every "
                "turn of every session automatically into a thought graph and "
                "extracts the facts from it — you never have to decide to "
                "store anything. Each session starts with an empty context "
                "window, so call `cashew_query` before answering any question "
                "about something said earlier. It returns the graph nodes "
                "related to your query, each labelled with its domain and "
                "type; when two of them disagree, the most recent one is what "
                "is true now."
            ),
        )


# ---------------------------------------------------------------------------
# Chronicle — indigokarasu/chronicle-agent-context-and-memory
# ---------------------------------------------------------------------------
class ChronicleAdapter(MemoryProviderAdapter):
    """Chronicle event-sourced memory, run in process off a checkout.

    ``setup`` sandboxes Chronicle in an ephemeral ``HERMES_HOME`` under the
    scenario workspace; :meth:`observe_turn` is its real per-turn capture;
    :meth:`trigger_consolidation` is its real offline pass; ``teardown``
    closes the store and deletes the home.

    Spec corrections, all verified against the upstream checkout rather than
    inferred from the plugin's prose:

    ``CHRONICLE_DB`` / ``CHRONICLE_VECTORS`` do not exist.
        Nothing upstream reads either name. ``ChronicleCore`` derives its
        database from ``hermes_home`` (``commons/db/chronicle/chronicle.db``)
        and keeps vectors *in that database*, not in a directory beside it. So
        the sandbox knob is ``HERMES_HOME``, and there is one store path to
        purge, not two.

    ``scripts/sweep_abstain.py``, ``prune_vectors.py``, ``writeback_vectors.py``
        exist, but none of them is a consolidation pass, so an earlier spec's
        correction (from the ``sweeps.py`` / ``reducer.py`` that never existed)
        landed on the wrong three files:

        * ``sweep_abstain.py`` is a LongMemEval parameter sweep. It takes an
          ``oracle.json`` dataset, builds a fresh temp home per instance, and
          prints a recommended ``retrieval.abstain_gate``. Between sessions it
          would tune a threshold against a foreign dataset and never touch
          this run's store.
        * ``prune_vectors.py --db PATH`` deletes vectors for sessions matching
          an exclusion prefix. Nothing here excludes a session, so it is a
          no-op by construction.
        * ``writeback_vectors.py`` is step 4 of an off-box re-embedding repair,
          and requires a migrated copy plus a pre-built manifest.

        What Chronicle actually calls consolidation is the curation queue plus
        the maintenance scheduler — ``ChronicleCore.tick``'s two halves. That
        is what :meth:`trigger_consolidation` runs, in process, against this
        run's own store.

    Capture is implicit.
        Chronicle does not ask the agent to remember; ``CaptureEngine.observe``
        appends each turn and extracts beliefs from it. The agent gets
        ``chronicle_search`` and nothing else.
    """

    name = "chronicle"

    def __init__(
        self,
        repo_root: str | Path | None = None,
        home_env_var: str = "HERMES_HOME",
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.repo_root = Path(repo_root) if repo_root else None
        self.home_env_var = home_env_var
        self.hermes_home: Path | None = None
        self.db_path: Path | None = None
        self._store: ChronicleStore | None = None

    def _chronicle(self) -> ChronicleStore:
        """The store sharing this run's core (a singleton keyed by the home)."""
        if self.hermes_home is None:
            raise RuntimeError("ChronicleAdapter.setup() has not run yet.")
        if self._store is None:
            self._store = ChronicleStore(
                self.hermes_home, repo_root=self.repo_root
            )
        return self._store

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        self.hermes_home = chronicle_home(workspace_dir)
        db_path = chronicle_db_path(self.hermes_home)
        self.db_path = db_path
        self._record(
            "init_store",
            {"hermes_home": str(self.hermes_home), "db": str(db_path)},
        )
        if not self.dry_run:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # The directory, not the file: Chronicle writes WAL/shm sidecars and an
        # optional sqlite-vec index beside the database, and consolidation
        # byte-growth should see all of it.
        self.store_paths = [str(db_path.parent)]
        return {
            "env": {self.home_env_var: str(self.hermes_home)},
            "contract": self.contract(),
            "store_paths": self.store_paths,
        }

    def observe_turn(
        self, user_content: str, assistant_content: str, session_id: str = ""
    ) -> None:
        """Capture one turn — Chronicle's write path, not the agent's.

        Driven by :mod:`memory_provider_evals.live` after every turn, which is
        where Hermes itself calls ``sync_turn``.
        """
        self._record("observe_turn", {"session_id": session_id})
        if self.dry_run:
            return
        self._chronicle().observe(user_content, assistant_content, session_id)

    def trigger_consolidation(self) -> None:
        self._record("consolidate", self.name)
        if self.dry_run:
            return
        jobs = self._chronicle().consolidate()
        self._record("curation_jobs", jobs)

    def teardown(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        if self.hermes_home and not self.dry_run:
            self._purge(self.hermes_home)
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["chronicle_search"],
            # Named for what they are upstream: `ChronicleMemoryProvider`
            # methods a Hermes host calls, not tools the agent can invoke.
            # Neither is wired here — the harness has no pre-LLM injection
            # point and no window to compress — so the prompt says what
            # Chronicle does for the agent without it, and nothing more.
            context_hooks=["pre_llm_call", "on_pre_compress"],
            system_prompt=(
                "Durable memory is provided by Chronicle, which records every "
                "turn of every session automatically and extracts the facts "
                "from them — you never have to decide to store anything. Each "
                "session starts with an empty context window, so call "
                "`chronicle_search` before answering any question about "
                "something said earlier. It returns both consolidated beliefs "
                "and the transcript lines behind them; when they disagree, "
                "the most recent one is what is true now."
            ),
        )


# ---------------------------------------------------------------------------
# Memex8 — Ex8-ca/memex8
# ---------------------------------------------------------------------------
class Memex8Adapter(MemoryProviderAdapter):
    """Memex8 (Rust daemon + Qdrant) started via ``docker compose``.

    ``trigger_consolidation`` POSTs to the Slumber endpoint, running the 13-phase
    consolidation pipeline (dedupe -> compress -> re-cluster -> ... -> verify).
    The exact slumber route is kept configurable (``slumber_path``); the REST
    API is served on :8080 per the upstream ``docker-compose.yml``.
    """

    name = "memex8"

    def __init__(
        self,
        compose_file: str | Path | None = None,
        base_url: str = "http://localhost:8080",
        slumber_path: str = "/api/v1/slumber",
        api_key: str | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.compose_file = Path(compose_file) if compose_file else None
        self.base_url = base_url.rstrip("/")
        self.slumber_path = slumber_path
        self.api_key = api_key or os.environ.get("MEMEX8_API_KEY", "")

    def _compose(self, *args: str) -> list[str]:
        cmd = ["docker", "compose"]
        if self.compose_file:
            cmd += ["-f", str(self.compose_file)]
        return cmd + list(args)

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        # Qdrant persists in a named docker volume; nothing to seed on host.
        self._run(self._compose("up", "-d", "qdrant", "memex8"))
        self.store_paths = []  # footprint lives inside the Qdrant container
        return {
            "env": {
                "MEMEX8_URL": self.base_url,
                "MEMEX8_API_KEY": self.api_key,
            },
            "contract": self.contract(),
            "store_paths": self.store_paths,
        }

    def trigger_consolidation(self) -> None:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}{self.slumber_path}"
        try:
            self._post_json(url, {"trigger": "manual"}, headers)
        except urllib.error.URLError as exc:  # pragma: no cover - live only
            raise RuntimeError(f"memex8 slumber request failed: {exc}") from exc

    def teardown(self) -> None:
        # ``-v`` drops the Qdrant volume so each suite starts from empty.
        self._run(self._compose("down", "-v"))
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["memex8_search"],
            context_hooks=["memex8_autorecall"],
            system_prompt=(
                "Memex8 provides human-like decaying memory with auto-recall. "
                "Use `memex8_search` to retrieve relevant memories."
            ),
        )


# ---------------------------------------------------------------------------
# Nachos — Nacho-Labs-LLC/hermes-plugin-nachos
# ---------------------------------------------------------------------------
class NachosAdapter(MemoryProviderAdapter):
    """Nachos durable-memory / context engine (text-only, local-first).

    Three-tier assembly: always-on manifest, bounded prefetch, explicit recall.
    Stores a SQLite/flat-file corpus plus transcript snapshots under a profile
    directory. There is **no** offline "sleep" pass — consolidation is inline
    compaction + pre-compress snapshotting (``nachos_core.compactor`` /
    ``nachos_core.snapshots``), so ``trigger_consolidation`` invokes compaction
    rather than a nightly reconciliation job.
    """

    name = "nachos"

    def __init__(
        self,
        compaction_cmd: list[str] | None = None,
        store: str = "sqlite",
        scorer: str = "lexical",
        home_env_var: str = "HERMES_HOME",
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        # Optional explicit compaction entry point; None => no-op (inline only).
        self.compaction_cmd = compaction_cmd
        self.store = store
        self.scorer = scorer
        self.home_env_var = home_env_var
        self.home_dir: Path | None = None
        self.config_path: Path | None = None

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        self.home_dir = Path(workspace_dir) / "nachos_home"
        nachos_dir = self.home_dir / "nachos"
        nachos_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = nachos_dir / "config.json"
        config = {
            "store": self.store,
            "scorer": self.scorer,
            "prefetch_top_n": 5,
            "prefetch_char_budget": 1500,
            "manifest_char_budget": 1200,
        }
        self._record("write_config", {"path": str(self.config_path), **config})
        if not self.dry_run:
            self.config_path.write_text(json.dumps(config, indent=2))
        self.store_paths = [str(nachos_dir)]
        return {
            "env": {self.home_env_var: str(self.home_dir)},
            "contract": self.contract(),
            "store_paths": self.store_paths,
        }

    def trigger_consolidation(self) -> None:
        if self.compaction_cmd is None:
            # Faithful: Nachos consolidates inline; no offline sleep routine.
            self._record("noop_inline_compaction", self.name)
            return
        env = os.environ.copy()
        if self.home_dir:
            env[self.home_env_var] = str(self.home_dir)
        self._run(self.compaction_cmd, env=env)

    def teardown(self) -> None:
        if self.home_dir:
            self._purge(self.home_dir)
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=[
                "nachos_memory_recall",
                "nachos_memory_put",
                "nachos_memory_remove",
            ],
            # Only recall reads memory; a put or a remove is not a retrieval,
            # and counting one would inflate the column that proves the
            # provider was consulted.
            retrieval_tools=["nachos_memory_recall"],
            context_hooks=["nachos_manifest", "nachos_prefetch"],
            system_prompt=(
                "Nachos provides durable memory via a manifest + bounded "
                "prefetch. Every session starts with an empty context window, "
                "so memory is the only thing that survives: call "
                "`nachos_memory_put` to store each durable fact the user "
                "states, phrased to stand alone without the conversation, and "
                "call `nachos_memory_recall` before answering any question "
                "about something the user told you earlier. When a stored "
                "fact is superseded, `nachos_memory_remove` the stale entry "
                "and put the new one."
            ),
        )

#: Provider id -> adapter class, for ``MEMORY_PROVIDER`` selection.
ADAPTERS: dict[str, type[MemoryProviderAdapter]] = {
    CashewAdapter.name: CashewAdapter,
    ChronicleAdapter.name: ChronicleAdapter,
    Memex8Adapter.name: Memex8Adapter,
    NachosAdapter.name: NachosAdapter,
}


def build_adapter(provider: str, **kwargs: Any) -> MemoryProviderAdapter:
    """Instantiate the adapter for ``provider`` (see :data:`ADAPTERS`)."""
    try:
        cls = ADAPTERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown memory provider {provider!r}. "
            f"Known providers: {sorted(ADAPTERS)}"
        ) from None
    return cls(**kwargs)


def default_provider() -> str:
    """Provider under test, from ``MEMORY_PROVIDER`` (default ``cashew``)."""
    return os.environ.get("MEMORY_PROVIDER", CashewAdapter.name)
