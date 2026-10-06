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

* Cashew consolidation really is ``plugins/memory/cashew/sleep_cron_script.py``.
* Chronicle's consolidation scripts are ``scripts/sweep_abstain.py``,
  ``scripts/prune_vectors.py`` and ``scripts/writeback_vectors.py`` (the task's
  ``sweeps.py``/``reducer.py`` do not exist upstream).
* Memex8 runs ``qdrant`` + ``memex8`` via ``docker compose`` and exposes a REST
  API on :8080; its consolidation ("Slumber") pipeline has **13** phases.
* Nachos is text-only (manifest/prefetch/recall); it has no offline "sleep"
  pass -- consolidation is inline compaction + snapshotting.

None of these backends are installed in this environment, so the adapters have
**not** been exercised end-to-end against a live backend here. Every adapter
supports ``dry_run=True``, which records the exact command / URL / config it
*would* execute (available as ``.actions``) without touching an external
process -- this is what the unit tests assert, and what you can use to
smoke-test wiring before a real backend is provisioned.
"""

from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path
from typing import Any

from traced_harness.memory import MemoryProviderAdapter, MemoryToolContract

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
    """Cashew thought-graph memory (SQLite + sentence-transformers).

    ``setup`` writes a *sandboxed* ``cashew.json`` (under the workspace, not the
    user's real ``~/.hermes/cashew.json`` — we never clobber the operator's
    profile) pointing at an ephemeral SQLite brain database.
    ``trigger_consolidation`` invokes Cashew's real sleep reconciliation entry
    point, ``plugins/memory/cashew/sleep_cron_script.py``.
    """

    name = "cashew"

    def __init__(
        self,
        consolidation_cmd: list[str] | None = None,
        config_env_var: str = "CASHEW_CONFIG",
        embedding_model: str = "all-MiniLM-L6-v2",
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        # Real upstream module path; overridable if installed elsewhere.
        self.consolidation_cmd = consolidation_cmd or [
            "python",
            "-m",
            "plugins.memory.cashew.sleep_cron_script",
        ]
        self.config_env_var = config_env_var
        self.embedding_model = embedding_model
        self.config_path: Path | None = None
        self.db_path: Path | None = None

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        root = Path(workspace_dir) / "cashew"
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "brain.db"
        self.config_path = root / "cashew.json"
        config = {
            "database_path": str(self.db_path),
            "embedding_model": self.embedding_model,
            # Offline in CI: avoid the ~500MB model download on first recall.
            "offline": True,
        }
        self._record("write_config", {"path": str(self.config_path), **config})
        if not self.dry_run:
            self.config_path.write_text(json.dumps(config, indent=2))
        self.store_paths = [str(self.db_path)]
        return {
            "env": {self.config_env_var: str(self.config_path)},
            "contract": self.contract(),
            "store_paths": self.store_paths,
        }

    def trigger_consolidation(self) -> None:
        env = os.environ.copy()
        if self.config_path:
            env[self.config_env_var] = str(self.config_path)
        self._run(self.consolidation_cmd, env=env)

    def teardown(self) -> None:
        if self.config_path:
            self._purge(self.config_path.parent)
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["cashew_query"],
            system_prompt=(
                "Durable memory is provided by Cashew. Call `cashew_query` to "
                "recall stored facts before answering from assumption."
            ),
        )


# ---------------------------------------------------------------------------
# Chronicle — indigokarasu/chronicle-agent-context-and-memory
# ---------------------------------------------------------------------------
class ChronicleAdapter(MemoryProviderAdapter):
    """Chronicle local-first memory (SQLite + local vector store).

    ``setup`` points Chronicle's engine at temporary SQLite and vector-store
    paths. ``trigger_consolidation`` runs Chronicle's real maintenance passes.

    Spec correction: the task referenced ``sweeps.py`` / ``reducer.py`` — those
    do not exist upstream. Chronicle's actual consolidation/maintenance scripts
    live under ``scripts/`` (``sweep_abstain.py``, ``prune_vectors.py``,
    ``writeback_vectors.py``); the engine itself is under ``engine/``.
    """

    name = "chronicle"

    def __init__(
        self,
        repo_root: str | Path | None = None,
        consolidation_scripts: list[str] | None = None,
        db_env_var: str = "CHRONICLE_DB",
        vectors_env_var: str = "CHRONICLE_VECTORS",
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.repo_root = Path(repo_root) if repo_root else None
        self.consolidation_scripts = consolidation_scripts or [
            "scripts/sweep_abstain.py",
            "scripts/prune_vectors.py",
            "scripts/writeback_vectors.py",
        ]
        self.db_env_var = db_env_var
        self.vectors_env_var = vectors_env_var
        self.db_path: Path | None = None
        self.vectors_path: Path | None = None

    def setup(self, workspace_dir: Path) -> dict[str, Any]:
        root = Path(workspace_dir) / "chronicle"
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "chronicle.db"
        self.vectors_path = root / "vectors"
        self.vectors_path.mkdir(exist_ok=True)
        self._record(
            "init_store",
            {"db": str(self.db_path), "vectors": str(self.vectors_path)},
        )
        self.store_paths = [str(self.db_path), str(self.vectors_path)]
        return {
            "env": {
                self.db_env_var: str(self.db_path),
                self.vectors_env_var: str(self.vectors_path),
            },
            "contract": self.contract(),
            "store_paths": self.store_paths,
        }

    def trigger_consolidation(self) -> None:
        env = os.environ.copy()
        if self.db_path:
            env[self.db_env_var] = str(self.db_path)
        if self.vectors_path:
            env[self.vectors_env_var] = str(self.vectors_path)
        for script in self.consolidation_scripts:
            script_path = (
                str(self.repo_root / script) if self.repo_root else script
            )
            self._run(["python", script_path], env=env)

    def teardown(self) -> None:
        self._purge(*(p for p in (self.db_path, self.vectors_path) if p))
        self._record("teardown", self.name)

    def contract(self) -> MemoryToolContract:
        return MemoryToolContract(
            provider=self.name,
            tools=["recall"],
            context_hooks=["chronicle_prefetch", "chronicle_compress"],
            system_prompt=(
                "Chronicle supplies recalled context and manages compression. "
                "Use `recall` for explicit lookups of earlier facts."
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
