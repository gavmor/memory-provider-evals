"""Stores behind the memory tools the agent actually calls.

A provider's tool names are only a contract. Something has to answer
``nachos_memory_recall`` when the agent calls it, and until it does, the
benchmark measures a model with no memory rather than a memory provider.
This module supplies those answers.

Two implementations, for two honestly different situations:

* :class:`LexicalMemoryStore` — a real, local, text-only store: SQLite rows
  scored by lexical overlap. This is what Nachos *is* (``store="sqlite"``,
  ``scorer="lexical"``, bounded prefetch; see :class:`~memory_provider_evals
  .adapters.NachosAdapter`), so for that provider it is the subject under
  test, not a stand-in. It also makes the wiring runnable with no backend to
  provision, which is what lets the harness-side telemetry be verified
  end-to-end.
* :class:`UnprovisionedStore` — for Cashew, Chronicle and Memex8, whose real
  backends are separate services. It raises :class:`BackendNotProvisioned`
  with the provisioning step named, so a run against an absent backend fails
  loudly and lands in the benchmark's ``error`` column instead of quietly
  scoring zero and looking like bad recall.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BackendNotProvisioned",
    "LexicalMemoryStore",
    "Memex8MemoryStore",
    "MemoryStore",
    "UnprovisionedStore",
]

_WORD = re.compile(r"[a-z0-9]+")

#: Words carrying no retrieval signal. Deliberately tiny: an aggressive list
#: would silently drop content words and flatter the scorer.
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did", "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "that", "the", "their", "there", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your"]
)


def _tokens(text: str) -> list[str]:
    return [
        w for w in _WORD.findall(str(text).lower()) if w and w not in _STOPWORDS
    ]


@runtime_checkable
class MemoryStore(Protocol):
    """What a memory tool needs from whatever is behind it."""

    def put(self, text: str, tags: list[str] | None = None) -> str:
        """Persist ``text`` and return its entry id."""
        ...

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Return up to ``limit`` stored entries relevant to ``query``."""
        ...

    def remove(self, entry_id: str) -> bool:
        """Delete one entry; return whether it existed."""
        ...


class BackendNotProvisioned(RuntimeError):
    """Raised when a memory tool is called but its backend is not running."""


class UnprovisionedStore:
    """Stands in for a provider whose backend this run has not provisioned.

    Every operation raises. Silence would be worse: a store that returned
    ``[]`` would be scored as a provider that remembered nothing, which is a
    claim about the provider rather than about this machine.
    """

    def __init__(self, provider: str, provisioning_hint: str = "") -> None:
        self.provider = provider
        self.provisioning_hint = provisioning_hint

    def _fail(self, operation: str) -> Any:
        hint = f" {self.provisioning_hint}" if self.provisioning_hint else ""
        raise BackendNotProvisioned(
            f"{self.provider}: cannot {operation} — its backend is not "
            f"provisioned in this environment.{hint}"
        )

    def put(self, text: str, tags: list[str] | None = None) -> str:
        return self._fail("store")

    def recall(self, query: str, limit: int = 5) -> list[str]:
        return self._fail("recall")

    def remove(self, entry_id: str) -> bool:
        return self._fail("remove")


class LexicalMemoryStore:
    """SQLite-backed durable memory with lexical relevance scoring.

    The database file is created on first write, not at construction: the
    provider adapter creates its store directory during ``setup()``, which
    runs after the agent (and therefore this store) has been built.

    Scoring is the fraction of the query's content words present in an entry,
    ties broken by recency. No embeddings: that is what "text-only,
    local-first, lexical scorer" means, and an entry sharing no content word
    with the query is not returned at all rather than being returned with a
    low score.
    """

    def __init__(
        self,
        db_path: str | Path,
        min_score: float = 0.34,
        default_limit: int = 5,
    ) -> None:
        self.db_path = Path(db_path)
        self.min_score = min_score
        self.default_limit = default_limit

    # -- storage ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT    NOT NULL,
                tags      TEXT    NOT NULL DEFAULT '',
                stored_at REAL    NOT NULL
            )
            """
        )
        return conn

    def put(self, text: str, tags: list[str] | None = None) -> str:
        entry = str(text).strip()
        if not entry:
            raise ValueError("refusing to store an empty memory")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO memories (text, tags, stored_at) VALUES (?, ?, ?)",
                (entry, " ".join(tags or []), time.time()),
            )
        return str(cur.lastrowid)

    def remove(self, entry_id: str) -> bool:
        if not self.db_path.exists():
            return False
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        return cur.rowcount > 0

    # -- retrieval --------------------------------------------------------
    def recall(self, query: str, limit: int | None = None) -> list[str]:
        if not self.db_path.exists():
            return []
        wanted = limit or self.default_limit
        query_words = set(_tokens(query))
        if not query_words:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text, tags, stored_at FROM memories"
            ).fetchall()

        scored: list[tuple[float, float, str]] = []
        for text, tags, stored_at in rows:
            entry_words = set(_tokens(f"{text} {tags}"))
            if not entry_words:
                continue
            score = len(query_words & entry_words) / len(query_words)
            if score >= self.min_score:
                scored.append((score, stored_at, text))
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [text for _score, _at, text in scored[:wanted]]

    def all_entries(self) -> list[str]:
        """Every stored entry, newest first — for assertions and debugging."""
        if not self.db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text FROM memories ORDER BY stored_at DESC, id DESC"
            ).fetchall()
        return [row[0] for row in rows]


class Memex8MemoryStore:
    """REST-backed store calling a live memex8 daemon.

    Endpoints (upstream Ex8-ca/memex8 REST API):
    - POST /api/v1/memories          — store a new memory
    - POST /api/v1/memories/search   — semantic search
    - DELETE /api/v1/memories/{id}   — delete by id
    - GET /api/v1/health             — health check (no auth)

    All endpoints except /health require ``Authorization: Bearer <key>``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise BackendNotProvisioned(
                f"memex8 {method} {path} returned {exc.code}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendNotProvisioned(
                f"memex8 unreachable at {self.base_url}: {exc.reason}"
            ) from exc

    def put(self, text: str, tags: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"content": text}
        if tags:
            payload["tags"] = tags
        result = self._request("POST", "/api/v1/memories", payload)
        # Upstream returns the created memory object with an "id" field.
        return str(result.get("id", ""))

    def recall(self, query: str, limit: int = 5) -> list[str]:
        result = self._request(
            "POST",
            "/api/v1/memories/search",
            {"query": query, "limit": limit},
        )
        # Upstream returns a list of memory objects; extract the content.
        if isinstance(result, list):
            return [str(m.get("content", m.get("text", ""))) for m in result[:limit]]
        # Some versions nest under a "results" or "memories" key.
        if isinstance(result, dict):
            items = result.get("results", result.get("memories", []))
            return [str(m.get("content", m.get("text", ""))) for m in items[:limit]]
        return []

    def remove(self, entry_id: str) -> bool:
        try:
            self._request("DELETE", f"/api/v1/memories/{entry_id}")
            return True
        except BackendNotProvisioned:
            return False
