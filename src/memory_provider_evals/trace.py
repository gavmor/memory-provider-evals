"""Memory-trace model and deterministic judges for memory-provider evaluation.

Reads traces produced by the ``traced_harness`` instrument -- specifically the
enriched JSONL written by ``traced_harness.session_runner.SessionRunner`` -- and
exposes the memory metadata the DeepEval metrics consume.

A turn's memory metadata lives under ``additional_metadata["memory"]`` with the
shape produced by the harness telemetry ``*.as_record()`` /
``record_memory_injection`` helpers::

    {
      "provider": "cashew",
      "session_id": "s2",
      "injection": {"base_tokens", "injected_tokens", "overhead_tokens"},
      "retrievals": [{"query", "passage_count", "passages", "latency_ms"}],
      "consolidation": {"wall_seconds", "cpu_seconds", "db_growth_bytes", ...},
      "generated_tokens": int,
    }

The scoring heuristics here are the *internals* of the DeepEval metrics in
``memory_provider_evals.metrics``; keeping them in one place means trace-level
analysis and eval-suite scoring cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traced_harness.eval import TraceTurn, load_trace

__all__ = [
    "DEFAULT_MEMORY_TOOL_NAMES",
    "MEMORY_META_KEY",
    "MemoryEvalSuite",
    "TraceRecord",
    "connects_entities_across_sessions",
    "text_contains",
    "token_f1",
]

MEMORY_META_KEY = "memory"

# Tool names across the four providers that perform memory retrieval. Used to
# treat a tool call's output as retrieved context even when the harness did not
# emit an explicit retrieval span (e.g. the agent called the tool directly).
DEFAULT_MEMORY_TOOL_NAMES = frozenset(
    {
        "cashew_query",
        "cashew_recall",
        "recall",
        "chronicle_recall",
        "memex8_search",
        "memex8_recall",
        "nachos_memory_recall",
        "memory",
    }
)


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _word_set(text: str) -> set[str]:
    return {w.strip(".,;:!?\"'()[]") for w in _normalize(text).split()} - {""}


def text_contains(haystack: str, needle: str, threshold: float = 0.6) -> bool:
    """True if ``needle`` appears in ``haystack`` by substring or word overlap.

    Substring match (after whitespace/case normalization) is authoritative.
    Otherwise, returns True when the fraction of ``needle`` content words also
    present in ``haystack`` meets ``threshold`` — tolerant of paraphrase/order.
    """
    h = _normalize(haystack)
    n = _normalize(needle)
    if not n:
        return False
    if n in h:
        return True
    needle_words = _word_set(needle)
    if not needle_words:
        return False
    hay_words = _word_set(haystack)
    overlap = len(needle_words & hay_words) / len(needle_words)
    return overlap >= threshold


def token_f1(reference: str, candidate: str) -> float:
    """Standard QA token-level F1 between a reference and candidate string."""
    ref = _word_set(reference)
    cand = _word_set(candidate)
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    common = ref & cand
    if not common:
        return 0.0
    precision = len(common) / len(cand)
    recall = len(common) / len(ref)
    return 2 * precision * recall / (precision + recall)


def connects_entities_across_sessions(
    retrieval_records: list[dict[str, Any]],
    entities: list[str],
    match_threshold: float = 0.6,
) -> bool:
    """True iff retrievals surface every entity across >= 2 distinct sessions.

    Shared by :meth:`MemoryEvalSuite.eval_multi_hop` (trace-level analysis) and
    the DeepEval ``MultiHopRetrievalMetric`` (eval-suite scoring), so both read
    the same definition of a cross-session hop.

    Each record is ``{"query": str, "passages": list[str], "session_id": str}``.
    """
    if len(entities) < 2 or not retrieval_records:
        return False
    all_sessions: set[str] = set()
    for ent in entities:
        hits: set[str] = set()
        for rec in retrieval_records:
            passages = " ".join(str(p) for p in rec.get("passages", []) or [])
            blob = f"{rec.get('query', '')} {passages}"
            if text_contains(blob, ent, match_threshold):
                hits.add(str(rec.get("session_id", "")))
        if not hits:
            return False
        all_sessions |= hits
    return len(all_sessions) >= 2


@dataclass
class TraceRecord:
    """A full (possibly multi-session) memory trace loaded from JSONL.

    Thin view over :class:`TraceTurn` list that surfaces the memory metadata
    the :class:`MemoryEvalSuite` judges consume.
    """

    turns: list[TraceTurn]
    file_path: Path | None = None
    memory_tool_names: frozenset[str] = DEFAULT_MEMORY_TOOL_NAMES

    @classmethod
    def from_file(cls, trace_file: str | Path) -> TraceRecord:
        turns = load_trace(trace_file)
        return cls(turns=turns, file_path=Path(trace_file))

    # -- memory metadata accessors ----------------------------------------
    @staticmethod
    def _mem(turn: TraceTurn) -> dict[str, Any]:
        meta = turn.additional_metadata or {}
        mem = meta.get(MEMORY_META_KEY)
        return mem if isinstance(mem, dict) else {}

    def session_id(self, turn: TraceTurn) -> str:
        mem = self._mem(turn)
        if mem.get("session_id"):
            return str(mem["session_id"])
        return str((turn.additional_metadata or {}).get("session_id", ""))

    @property
    def final_turn(self) -> TraceTurn | None:
        return self.turns[-1] if self.turns else None

    def retrieval_records(self) -> list[dict[str, Any]]:
        """All retrieval span records across every turn, tagged with session."""
        records: list[dict[str, Any]] = []
        for turn in self.turns:
            sid = self.session_id(turn)
            for rec in self._mem(turn).get("retrievals", []) or []:
                item = dict(rec)
                item.setdefault("session_id", sid)
                records.append(item)
        return records

    def retrieved_texts(self) -> list[str]:
        """Every passage injected via retrieval spans or memory-tool outputs."""
        texts: list[str] = []
        for rec in self.retrieval_records():
            texts.extend(str(p) for p in rec.get("passages", []) or [])
        for turn in self.turns:
            for tool in turn.tools_called:
                if tool.name in self.memory_tool_names and tool.output:
                    texts.append(tool.output)
        return texts

    def consolidation_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for turn in self.turns:
            cons = self._mem(turn).get("consolidation")
            if isinstance(cons, dict):
                out.append(cons)
        return out

    def injected_tokens_total(self) -> int:
        total = 0
        for turn in self.turns:
            inj = self._mem(turn).get("injection") or {}
            total += int(inj.get("overhead_tokens", 0) or 0)
        return total

    def generated_tokens_total(self) -> int:
        total = 0
        for turn in self.turns:
            gen = self._mem(turn).get("generated_tokens")
            if gen is None:
                # Fall back to a heuristic on the produced answer text.
                gen = max(1, round(len(turn.actual_output) / 4)) if (
                    turn.actual_output
                ) else 0
            total += int(gen)
        return total


class MemoryEvalSuite:
    """Assertion judges for head-to-head memory-provider traces.

    All judges are text-only: they read retrieved passages, memory-tool
    outputs, and generated answers. No visual/multimodal signals are used.
    """

    def __init__(self, match_threshold: float = 0.6) -> None:
        self.match_threshold = match_threshold

    def _present_in_context(self, trace: TraceRecord, fact: str) -> bool:
        if any(
            text_contains(t, fact, self.match_threshold)
            for t in trace.retrieved_texts()
        ):
            return True
        return any(
            text_contains(turn.actual_output, fact, self.match_threshold)
            for turn in trace.turns
        )

    def eval_precision_recall(
        self, trace: TraceRecord, ground_truth: str
    ) -> float:
        """Token-F1 that the agent surfaced the target fact without noise.

        Measured against the final generated answer: recall rewards recalling
        the ground-truth fact, precision penalizes padding the answer with
        unrelated (potentially hallucinated) content.
        """
        final = trace.final_turn
        if final is None:
            return 0.0
        return token_f1(ground_truth, final.actual_output)

    def eval_temporal_invalidation(
        self, trace: TraceRecord, stale_fact: str
    ) -> bool:
        """True iff an outdated/superseded fact was NOT recalled anywhere.

        Checks both retrieved context spans and generated answers.
        """
        return not self._present_in_context(trace, stale_fact)

    def eval_token_overhead_ratio(self, trace: TraceRecord) -> float:
        """Ratio of memory-injection overhead tokens to generated tokens."""
        generated = trace.generated_tokens_total()
        if generated == 0:
            return 0.0
        return trace.injected_tokens_total() / generated

    def eval_contradiction_rejection(
        self, trace: TraceRecord, superseded_facts: list[str]
    ) -> bool:
        """True iff NO superseded fact appears in the final answer or context.

        Scoped to the final belief-revision turn: its generated answer plus the
        retrieval spans / memory-tool outputs that supported it. (Whole-trace
        recall of a now-stale fact is judged by ``eval_temporal_invalidation``.)
        """
        final = trace.final_turn
        if final is None:
            return False
        contexts = [final.actual_output]
        for rec in trace._mem(final).get("retrievals", []) or []:
            contexts.extend(str(p) for p in rec.get("passages", []) or [])
        for tool in final.tools_called:
            if tool.name in trace.memory_tool_names and tool.output:
                contexts.append(tool.output)
        for fact in superseded_facts:
            if any(
                text_contains(c, fact, self.match_threshold) for c in contexts
            ):
                return False
        return True

    def eval_multi_hop(
        self, trace: TraceRecord, entities: list[str]
    ) -> bool:
        """True iff retrievals connect entities introduced in different sessions.

        Requires (a) every entity to appear in some retrieved passage and
        (b) the connecting retrievals to span at least two distinct sessions —
        evidence of a cross-session graph/passage hop rather than single-turn
        recall.

        Delegates to :func:`connects_entities_across_sessions` so the DeepEval
        ``MultiHopRetrievalMetric`` scores on exactly this definition.
        """
        return connects_entities_across_sessions(
            trace.retrieval_records(), entities, self.match_threshold
        )

    def cost_accuracy_point(
        self, trace: TraceRecord, accuracy: float
    ) -> tuple[float, float]:
        """One (context_token_overhead, accuracy) point for the frontier plot.

        Pair across providers to compare retrieval cost vs accuracy.
        """
        return (float(trace.injected_tokens_total()), float(accuracy))
