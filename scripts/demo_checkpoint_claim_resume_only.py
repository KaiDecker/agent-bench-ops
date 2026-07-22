import argparse
import asyncio
import json
import os
import selectors
import sys

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
        description=("Claim a paused AgentRun for resume, then exit before invoking LangGraph.")
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

    registry = build_default_registry()
    gateway = ToolGateway(registry)

    runtime = AgentRuntime(
        model=ScriptedEmployeeLookupModel(),
        model_provider="scripted",
        model_name=("scripted-checkpoint-pause-v1"),
        registry=registry,
        gateway=gateway,
        checkpointer_factory=(open_postgres_checkpointer),
    )

    prepared = await runtime._claim_resume(  # noqa: SLF001
        run_id=run_id,
    )

    print(
        json.dumps(
            {
                "run_id": prepared.run_id,
                "checkpoint_ref": (prepared.checkpoint_ref),
                "breakpoint_nodes": list(prepared.breakpoint_nodes),
                "resume_claim_acquired": True,
                "simulated_process_exit": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
