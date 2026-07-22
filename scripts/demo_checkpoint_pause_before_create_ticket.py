import asyncio
import json
import os
import selectors
import sys

from app.config import settings

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


async def async_main() -> None:
    assert_strict_msgpack_enabled()

    from app.agent.checkpoint import (
        open_postgres_checkpointer,
    )
    from app.agent.model import (
        ScriptedCreateTicketModel,
    )
    from app.agent.runtime import AgentRuntime
    from app.tools.gateway import ToolGateway
    from app.tools.registry import (
        build_default_registry,
    )

    registry = build_default_registry()
    gateway = ToolGateway(registry)

    runtime = AgentRuntime(
        model=ScriptedCreateTicketModel(),
        model_provider="scripted",
        model_name=("scripted-create-ticket-checkpoint-v1"),
        registry=registry,
        gateway=gateway,
        checkpointer_factory=(open_postgres_checkpointer),
    )

    result = await runtime.run_benchmark_task(
        task_key="create_ticket_001",
        task_version=1,
        actor_id="benchmark-agent",
        permissions=[
            "ticket.write",
        ],
        prompt_version=("stage5d-side-effect-pause-v1"),
        configuration={
            "stage": "5D",
            "test_case": ("pause_before_create_ticket"),
            "checkpoint_backend": ("postgresql"),
            "database": (settings.database_url.split("@")[-1]),
        },
        pause_before_tools=True,
    )

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(
            async_main(),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(async_main())
