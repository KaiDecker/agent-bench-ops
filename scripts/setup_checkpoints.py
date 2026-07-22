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
    """
    在导入 LangGraph Checkpointer 前检查安全配置。

    LANGGRAPH_STRICT_MSGPACK 在当前实现中需要在
    Python 进程启动和相关模块导入前设置。
    """

    value = os.environ.get(
        "LANGGRAPH_STRICT_MSGPACK",
        "",
    )

    if value.strip().lower() not in _TRUE_VALUES:
        raise RuntimeError("LANGGRAPH_STRICT_MSGPACK must be set to true before starting Python.")


async def async_main() -> None:
    assert_strict_msgpack_enabled()

    # 必须在环境变量检查之后导入。
    from app.agent.checkpoint import (
        masked_checkpoint_connection_string,
        setup_postgres_checkpointer,
    )

    await setup_postgres_checkpointer()

    print(
        json.dumps(
            {
                "status": "ready",
                "backend": "postgresql",
                "connection": (masked_checkpoint_connection_string()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    """创建 Psycopg 在 Windows 上兼容的事件循环。"""

    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(
            async_main(),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(async_main())
