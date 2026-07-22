import argparse
import asyncio
import json
import os
import selectors
import sys
from typing import Any

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}


def assert_strict_msgpack_enabled() -> None:
    value = os.environ.get(
        "LANGGRAPH_STRICT_MSGPACK",
        "",
    )

    if value.strip().lower() not in _TRUE_VALUES:
        raise RuntimeError("LANGGRAPH_STRICT_MSGPACK must be set to true before starting Python.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Attempt to claim the same paused AgentRun concurrently.")
    )

    parser.add_argument(
        "run_id",
        help="Paused AgentRun ID.",
    )

    return parser.parse_args()


async def async_main(
    run_id: str,
) -> None:
    assert_strict_msgpack_enabled()

    from app.agent.checkpoint import (
        open_postgres_checkpointer,
    )
    from app.agent.model import (
        ScriptedEmployeeLookupModel,
    )
    from app.agent.runtime import AgentRuntime
    from app.tools.gateway import ToolGateway
    from app.tools.registry import (
        build_default_registry,
    )

    def build_runtime() -> AgentRuntime:
        registry = build_default_registry()

        return AgentRuntime(
            model=ScriptedEmployeeLookupModel(),
            model_provider="scripted",
            model_name=("scripted-checkpoint-pause-v1"),
            registry=registry,
            gateway=ToolGateway(registry),
            checkpointer_factory=(open_postgres_checkpointer),
        )

    runtime_a = build_runtime()
    runtime_b = build_runtime()

    start_gate = asyncio.Event()

    async def attempt_claim(
        *,
        worker: str,
        runtime: AgentRuntime,
    ) -> dict[str, Any]:
        await start_gate.wait()

        try:
            prepared = await runtime._claim_resume(  # noqa: SLF001
                run_id=run_id,
            )
        except RuntimeError as exc:
            return {
                "worker": worker,
                "outcome": "rejected",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        return {
            "worker": worker,
            "outcome": "claimed",
            "checkpoint_ref": (prepared.checkpoint_ref),
            "breakpoint_nodes": list(prepared.breakpoint_nodes),
        }

    task_a = asyncio.create_task(
        attempt_claim(
            worker="worker_a",
            runtime=runtime_a,
        )
    )

    task_b = asyncio.create_task(
        attempt_claim(
            worker="worker_b",
            runtime=runtime_b,
        )
    )

    # 让两个协程都先运行到等待门的位置。
    await asyncio.sleep(0)

    start_gate.set()

    results = await asyncio.gather(
        task_a,
        task_b,
    )

    claimed_count = sum(result["outcome"] == "claimed" for result in results)

    rejected_count = sum(result["outcome"] == "rejected" for result in results)

    payload = {
        "run_id": run_id,
        "claimed_count": claimed_count,
        "rejected_count": rejected_count,
        "results": sorted(
            results,
            key=lambda item: item["worker"],
        ),
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    if claimed_count != 1:
        raise RuntimeError("Exactly one worker must acquire the resume claim.")

    if rejected_count != 1:
        raise RuntimeError("Exactly one worker must be rejected.")


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    arguments = parse_arguments()

    if sys.platform == "win32":
        asyncio.run(
            async_main(arguments.run_id),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(async_main(arguments.run_id))
