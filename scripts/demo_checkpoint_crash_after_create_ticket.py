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


class InjectedProcessCrash(BaseException):
    """模拟工具副作用成功后的进程级崩溃。"""


class CrashAfterSuccessfulGateway:
    """
    委托真实 ToolGateway 执行。

    当工具第一次成功返回后，在调用方收到结果之前抛出
    BaseException。create_tool_node 的 except Exception
    不会捕获它，因此节点不会返回，也不会保存节点完成状态。
    """

    def __init__(
        self,
        delegate: Any,
    ) -> None:
        self._delegate = delegate
        self._triggered = False

    async def execute(
        self,
        **kwargs: Any,
    ) -> Any:
        response = await self._delegate.execute(**kwargs)

        if not self._triggered and response.status == "succeeded":
            self._triggered = True

            raise InjectedProcessCrash("Injected process crash after successful tool execution.")

        return response


def contains_injected_crash(
    error: BaseException,
) -> bool:
    """处理异步框架可能产生的 BaseExceptionGroup。"""

    if isinstance(
        error,
        InjectedProcessCrash,
    ):
        return True

    if isinstance(
        error,
        BaseExceptionGroup,
    ):
        return any(contains_injected_crash(child) for child in error.exceptions)

    return False


def assert_strict_msgpack_enabled() -> None:
    value = os.environ.get(
        "LANGGRAPH_STRICT_MSGPACK",
        "",
    )

    if value.strip().lower() not in _TRUE_VALUES:
        raise RuntimeError("LANGGRAPH_STRICT_MSGPACK must be set to true before starting Python.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a process crash after "
            "create_ticket succeeds but before "
            "the tools node is checkpointed."
        )
    )

    parser.add_argument(
        "run_id",
        help=("A create_ticket AgentRun currently paused before the tools node."),
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
        ScriptedCreateTicketModel,
    )
    from app.agent.runtime import AgentRuntime
    from app.tools.gateway import ToolGateway
    from app.tools.registry import (
        build_default_registry,
    )

    registry = build_default_registry()

    real_gateway = ToolGateway(registry)

    crashing_gateway = CrashAfterSuccessfulGateway(real_gateway)

    runtime = AgentRuntime(
        model=ScriptedCreateTicketModel(),
        model_provider="scripted",
        model_name=("scripted-create-ticket-checkpoint-v1"),
        registry=registry,
        gateway=crashing_gateway,
        checkpointer_factory=(open_postgres_checkpointer),
    )

    graph_config: dict[str, Any] = {
        "recursion_limit": 12,
        "configurable": {
            "thread_id": run_id,
        },
    }

    try:
        # 这是故障注入探针，不调用 _claim_resume()。
        # AgentRun 保留 paused=true，便于随后使用正式接口恢复。
        await runtime._invoke_graph(  # noqa: SLF001
            initial_state=None,
            config=graph_config,
            interrupt_before=("tools",),
        )
    except BaseException as error:
        if not contains_injected_crash(error):
            raise

        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "simulated_crash": True,
                    "exception_type": (type(error).__name__),
                    "message": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    raise RuntimeError("The graph completed without triggering the injected process crash.")


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
