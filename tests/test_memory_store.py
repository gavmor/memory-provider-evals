"""The stores behind the memory tools.

``LexicalMemoryStore`` is the Nachos subject under test, not a mock, so its
retrieval behaviour is asserted directly: what it returns, what it refuses to
return, and that a superseded fact really is gone once removed.
"""

from __future__ import annotations

import pytest

from memory_provider_evals.memory_store import (
    BackendNotProvisioned,
    LexicalMemoryStore,
    MemoryStore,
    UnprovisionedStore,
)


@pytest.fixture
def store(tmp_path) -> LexicalMemoryStore:
    return LexicalMemoryStore(tmp_path / "nachos" / "memories.db")


def test_both_stores_satisfy_the_protocol(store):
    assert isinstance(store, MemoryStore)
    assert isinstance(UnprovisionedStore("cashew"), MemoryStore)


# -- LexicalMemoryStore -----------------------------------------------------
def test_recall_before_anything_is_stored_is_empty_not_an_error(store):
    """A cold store has no database file yet; that is not a failure."""
    assert not store.db_path.exists()
    assert store.recall("what do I drive") == []


def test_put_then_recall_round_trips(store):
    store.put("Gavin drives a Honda Civic.")
    assert store.recall("what does Gavin drive") == ["Gavin drives a Honda Civic."]


def test_put_creates_the_database_under_the_providers_own_path(store):
    store.put("Gavin lives in Berlin.")
    assert store.db_path.exists()
    assert store.db_path.parent.name == "nachos"


def test_recall_ignores_entries_sharing_no_content_word(store):
    store.put("Gavin drives a Honda Civic.")
    store.put("The capital of France is Paris.")
    assert store.recall("what does Gavin drive") == ["Gavin drives a Honda Civic."]


def test_recall_ranks_better_overlap_first(store):
    store.put("Gavin drives a car.")
    store.put("Gavin drives a Honda Civic car.")
    assert store.recall("Gavin drives a Honda Civic car")[0] == (
        "Gavin drives a Honda Civic car."
    )


def test_recall_breaks_score_ties_by_recency(store):
    """Two equally relevant entries: the newer belief wins."""
    store.put("Gavin drives a Honda Civic.")
    store.put("Gavin drives a Tesla Model 3.")
    top = store.recall("what does Gavin drive", limit=1)
    assert top == ["Gavin drives a Tesla Model 3."]


def test_recall_respects_the_limit(store):
    for i in range(5):
        store.put(f"Gavin drives vehicle number {i}.")
    assert len(store.recall("what does Gavin drive", limit=2)) == 2


def test_a_query_of_only_stopwords_retrieves_nothing(store):
    """Otherwise every stored entry would match every vacuous question."""
    store.put("Gavin drives a Honda Civic.")
    assert store.recall("what is it") == []


def test_remove_deletes_the_superseded_entry(store):
    stale = store.put("Gavin drives a Honda Civic.")
    store.put("Gavin drives a Tesla Model 3.")
    assert store.remove(stale) is True
    assert store.recall("what does Gavin drive") == [
        "Gavin drives a Tesla Model 3."
    ]


def test_remove_reports_a_miss(store):
    store.put("Gavin drives a Honda Civic.")
    assert store.remove("9999") is False


def test_remove_on_a_cold_store_is_a_miss_not_an_error(store):
    assert store.remove("1") is False


def test_empty_memories_are_refused(store):
    with pytest.raises(ValueError, match="empty memory"):
        store.put("   ")


def test_tags_are_searchable(store):
    store.put("Model 3.", tags=["vehicle", "gavin"])
    assert store.recall("gavin vehicle") == ["Model 3."]


def test_all_entries_is_newest_first(store):
    store.put("first")
    store.put("second")
    assert store.all_entries() == ["second", "first"]


# -- UnprovisionedStore -----------------------------------------------------
@pytest.mark.parametrize(
    ("operation", "args"),
    [("put", ("a fact",)), ("recall", ("a query",)), ("remove", ("1",))],
)
def test_an_unprovisioned_backend_fails_loudly(operation, args):
    """Returning [] would be scored as a provider that remembered nothing.

    That is a claim about the provider; the truth is a claim about this
    machine. The benchmark records the raised error in its `error` column.
    """
    store = UnprovisionedStore("memex8", "Run `docker compose up -d`.")
    with pytest.raises(BackendNotProvisioned) as exc:
        getattr(store, operation)(*args)
    assert "memex8" in str(exc.value)
    assert "docker compose" in str(exc.value)
