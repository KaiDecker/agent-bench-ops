import asyncio
import json
import os
import selectors
import sys

from app.config import settings

TASK_KEY = "employee_lookup_001"
TASK_VERSION = 1

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}


def assert_strict_msgpack_enabled() -> None:
    """
    确认 LangGraph 严格 MessagePack 模式已启用。

    这个检查必须发生在导入 Checkpointer 相关模块之前。
    """

    value = os.environ.get(
        "LANGGRAPH_STRICT_MSGPACK",
        "",
    )

    if value.strip().lower() not in _TRUE_VALUES:
        raise RuntimeError("LANGGRAPH_STRICT_MSGPACK must be set to true before starting Python.")


async def async_main() -> None:
    assert_strict_msgpack_enabled()

    # 严格模式检查完成后再导入 Agent 和 Checkpointer 模块。
    from app.agent.checkpoint import (
        open_postgres_checkpointer,
    )
    from app.agent.deepseek import (
        DeepSeekToolCallingModel,
    )
    from app.agent.runtime import AgentRuntime
    from app.tools.gateway import ToolGateway
    from app.tools.registry import (
        build_default_registry,
    )

    api_key = settings.deepseek_api_key

    if api_key is None:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured in the project .env file.")

    base_url = settings.deepseek_base_url.strip()
    model_name = settings.deepseek_model.strip()

    if not base_url:
        raise RuntimeError("DEEPSEEK_BASE_URL cannot be empty.")

    if not model_name:
        raise RuntimeError("DEEPSEEK_MODEL cannot be empty.")

    registry = build_default_registry()
    gateway = ToolGateway(registry)

    model = DeepSeekToolCallingModel(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=60.0,
        max_retries=2,
    )

    runtime = AgentRuntime(
        model=model,
        model_provider="deepseek",
        model_name=model_name,
        registry=registry,
        gateway=gateway,
        checkpointer_factory=(open_postgres_checkpointer),
    )

    result = await runtime.run_benchmark_task(
        task_key=TASK_KEY,
        task_version=TASK_VERSION,
        actor_id="benchmark-agent",
        permissions=[
            "employee.read",
        ],
        prompt_version="stage5d-checkpoint-v1",
        agent_strategy=("langgraph-model-tool-loop"),
        memory_strategy="messages-state",
        configuration={
            "stage": "5D",
            "provider": "deepseek",
            "checkpoint_enabled": True,
            "thinking_mode": "disabled",
            "parallel_tool_calls": False,
        },
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
    """创建 Psycopg 异步连接兼容的事件循环。"""

    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(
            async_main(),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(async_main())
