"""The live path: a real agent, with real memory tools, over a real MCP client.

Everything else in this package is either a pure scorer or a fake-driven
test. This module is the one that connects the pieces for an actual run:

    adapter -> MCP server exposing its tools
            -> mcp Client (in process)
            -> create_agent(client=..., memory_adapter=...)
            -> make_memory_turn_executor(agent, adapter)
            -> SessionRunner / benchmark

The ``client=`` argument is the whole point. ``create_agent`` registers tool
names and injects the provider's system-prompt contract, but the concrete
implementations arrive over MCP; without a client the agent is told it has
``cashew_query`` and then cannot call it.

Model
-----
``AGENT_MODEL_NAME`` selects the agent model (the harness default is
``gemini-3.1-flash-lite-preview``, which returned 503s throughout development
— prefer ``gemini-flash-lite-latest``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval.dataset import ConversationalGolden, EvaluationDataset
from deepeval.test_case import ConversationalTestCase
from traced_harness.agent import create_agent
from traced_harness.client import connect_mcp
from traced_harness.memory import (
    MemoryProviderAdapter,
    make_memory_turn_executor,
)
from traced_harness.session_runner import TurnExecutor

from memory_provider_evals.adapters import build_adapter, default_provider
from memory_provider_evals.benchmark import (
    BenchmarkReport,
    render_table,
    run_provider_benchmark,
)
from memory_provider_evals.bridge import conversational_test_case_from_trace
from memory_provider_evals.mcp_server import memory_server_for
from memory_provider_evals.memory_store import MemoryStore
from memory_provider_evals.trace import TraceRecord

__all__ = [
    "LiveMemoryAgent",
    "live_executor_factory",
    "live_memory_agent",
    "run_live_benchmark",
    "run_live_scenario",
]

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "tests/evals/.dataset.json"


def agent_model_name() -> str:
    return os.environ.get("AGENT_MODEL_NAME", "gemini-flash-lite-latest")


@dataclass
class LiveMemoryAgent:
    """A wired agent plus the things a caller needs to inspect it."""

    executor: TurnExecutor
    store: MemoryStore
    mcp_label: str
    agent: Any
    tool_names: list[str]


@asynccontextmanager
async def live_memory_agent(
    adapter: MemoryProviderAdapter,
    workspace_dir: str | Path,
    model_name: str | None = None,
) -> AsyncIterator[LiveMemoryAgent]:
    """Stand up this provider's MCP server and an agent connected to it.

    The connection lives for the duration of the block: the MCP client, the
    agent and the store all belong to one scenario's workspace, so nothing
    one scenario remembered can reach the next.
    """
    server, store = memory_server_for(adapter, workspace_dir)
    async with connect_mcp(server=server) as (client, label):
        tools = await client.list_tools()
        agent = await create_agent(
            client=client,
            memory_adapter=adapter,
            model_name=model_name or agent_model_name(),
        )
        yield LiveMemoryAgent(
            executor=_capturing(make_memory_turn_executor(agent, adapter), adapter),
            store=store,
            mcp_label=label,
            agent=agent,
            tool_names=[t.name for t in tools.tools],
        )


def _capturing(executor: TurnExecutor, adapter: MemoryProviderAdapter) -> TurnExecutor:
    """Hand each finished turn to a provider that captures turns implicitly.

    Some providers do not ask the agent to remember: Chronicle appends every
    turn to an event log and extracts beliefs from it, which is the whole
    reason its contract exposes a read tool and no write tool. Hermes drives
    that with ``sync_turn`` after each turn; this is the harness's equivalent
    hook, and the adapter opts in by defining ``observe_turn``.

    A provider that writes only when the agent calls a tool (Nachos) defines
    nothing and is passed through untouched.

    Capture failures are not swallowed. Upstream lets a broken capture
    degrade a live session rather than break it, but here a provider that
    silently stopped recording would be scored as one that remembers nothing
    — a claim about the provider when the truth is that it crashed.
    """
    observe = getattr(adapter, "observe_turn", None)
    if observe is None:
        return executor

    async def _executor(prompt: str, session_id: str, peripheral: str) -> Any:
        turn = await executor(prompt, session_id, peripheral)
        observe(
            turn.prompt,
            turn.output,
            session_id=getattr(turn, "session_id", "") or session_id,
        )
        return turn

    return _executor


def live_executor_factory(
    adapter: MemoryProviderAdapter,
    model_name: str | None = None,
):
    """An ``ExecutorFactory`` for :func:`run_provider_benchmark`."""

    @asynccontextmanager
    async def _factory(scenario_workspace: Path) -> AsyncIterator[TurnExecutor]:
        async with live_memory_agent(
            adapter, scenario_workspace, model_name
        ) as live:
            yield live.executor

    return _factory


async def run_live_scenario(
    golden: ConversationalGolden,
    adapter: MemoryProviderAdapter,
    workspace_dir: str | Path,
    model_name: str | None = None,
    trace_dir: str | Path | None = None,
    reset_after: bool = True,
) -> ConversationalTestCase:
    """Replay one golden against a live, tool-equipped memory provider."""
    from traced_harness.memory import make_memory_session_runner

    from memory_provider_evals.bridge import scenario_from_golden

    async with live_memory_agent(adapter, workspace_dir, model_name) as live:
        runner = make_memory_session_runner(
            adapter, live.executor, workspace_dir, trace_dir=trace_dir
        )
        try:
            result = await runner.run_scenario(scenario_from_golden(golden))
        finally:
            if reset_after:
                runner.reset_suite()

    trace = TraceRecord.from_file(result.trace_file)
    return conversational_test_case_from_trace(
        trace, golden, provider=adapter.name
    )


async def run_live_benchmark(
    goldens: list[ConversationalGolden],
    adapter: MemoryProviderAdapter,
    workspace_dir: str | Path,
    model_name: str | None = None,
    output_dir: str | Path | None = None,
) -> BenchmarkReport:
    """Benchmark one provider end to end, one MCP connection per scenario."""
    model = model_name or agent_model_name()
    return await run_provider_benchmark(
        goldens=goldens,
        adapter=adapter,
        executor_factory=live_executor_factory(adapter, model),
        workspace_dir=workspace_dir,
        model=model,
        output_dir=output_dir,
    )


def _load_goldens(path: Path) -> list[ConversationalGolden]:
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(file_path=str(path))
    if not dataset.goldens:
        raise SystemExit(f"No goldens found at {path}.")
    return list(dataset.goldens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="memorybench",
        description="Run the memory benchmark against one live provider.",
    )
    parser.add_argument("--provider", default=default_provider())
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--workspace", type=Path, default=Path(".runs/live"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit(
            "GOOGLE_API_KEY / GEMINI_API_KEY unset — the benchmark drives a "
            "live agent model."
        )

    report = asyncio.run(
        run_live_benchmark(
            goldens=_load_goldens(args.dataset),
            adapter=build_adapter(args.provider),
            workspace_dir=args.workspace,
            model_name=args.model,
            output_dir=args.output_dir,
        )
    )
    print(render_table([report]))
    if report.archive_path:
        print(f"\narchive: {report.archive_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
