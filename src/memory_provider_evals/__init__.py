"""DeepEval suite benchmarking Hermes durable-memory providers.

Layered on the ``traced-harness`` instrument: the harness drives the agent,
emits OpenTelemetry spans and JSONL traces, and provides first-class memory
support (``MemoryProviderAdapter``, memory telemetry, prompt wiring). This
package supplies the providers under test, defines what "good memory" means,
and scores it with DeepEval.
"""

from memory_provider_evals.adapters import (
    ADAPTERS,
    CashewAdapter,
    ChronicleAdapter,
    Memex8Adapter,
    NachosAdapter,
    build_adapter,
    default_provider,
)
from memory_provider_evals.bridge import (
    conversational_test_case_from_trace,
    run_memory_scenario,
    scenario_from_golden,
)
from memory_provider_evals.ingest import to_deepeval_test_cases
from memory_provider_evals.metrics import (
    MemoryRecallF1Metric,
    MemoryTokenOverheadMetric,
    MultiHopRetrievalMetric,
)
from memory_provider_evals.trace import MemoryEvalSuite, TraceRecord

__all__ = [
    "ADAPTERS",
    "CashewAdapter",
    "ChronicleAdapter",
    "Memex8Adapter",
    "MemoryEvalSuite",
    "MemoryRecallF1Metric",
    "MemoryTokenOverheadMetric",
    "MultiHopRetrievalMetric",
    "NachosAdapter",
    "TraceRecord",
    "build_adapter",
    "conversational_test_case_from_trace",
    "default_provider",
    "run_memory_scenario",
    "scenario_from_golden",
    "to_deepeval_test_cases",
]
