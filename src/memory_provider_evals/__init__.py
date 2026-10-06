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
from memory_provider_evals.chronicle_backend import (
    ChronicleStore,
    chronicle_repo_root,
)
from memory_provider_evals.ingest import to_deepeval_test_cases
from memory_provider_evals.mcp_server import (
    build_memory_server,
    memory_server_for,
    store_for,
)
from memory_provider_evals.memory_store import (
    BackendNotProvisioned,
    LexicalMemoryStore,
    MemoryStore,
    UnprovisionedStore,
)
from memory_provider_evals.metrics import (
    MemoryRecallF1Metric,
    MemoryTokenOverheadMetric,
    MultiHopRetrievalMetric,
)
from memory_provider_evals.trace import MemoryEvalSuite, TraceRecord

__all__ = [
    "ADAPTERS",
    "BackendNotProvisioned",
    "CashewAdapter",
    "ChronicleAdapter",
    "ChronicleStore",
    "LexicalMemoryStore",
    "Memex8Adapter",
    "MemoryEvalSuite",
    "MemoryRecallF1Metric",
    "MemoryStore",
    "MemoryTokenOverheadMetric",
    "MultiHopRetrievalMetric",
    "NachosAdapter",
    "TraceRecord",
    "UnprovisionedStore",
    "build_adapter",
    "build_memory_server",
    "chronicle_repo_root",
    "conversational_test_case_from_trace",
    "default_provider",
    "memory_server_for",
    "run_memory_scenario",
    "scenario_from_golden",
    "store_for",
    "to_deepeval_test_cases",
]
