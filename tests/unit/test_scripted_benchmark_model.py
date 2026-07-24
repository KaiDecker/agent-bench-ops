from types import SimpleNamespace
from typing import Any, cast

from app.agent.model import (
    ScriptedBenchmarkModel,
)


def fake_tool(
    name: str,
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            metadata=SimpleNamespace(
                name=name,
            )
        ),
    )


async def test_routes_employee_lookup_task() -> None:
    model = ScriptedBenchmarkModel()

    response = await model.ainvoke(
        messages=[],
        tools=[
            fake_tool("get_employee"),
        ],
    )

    assert len(response.tool_calls) == 1

    assert response.tool_calls[0]["name"] == "get_employee"


async def test_routes_create_ticket_task() -> None:
    model = ScriptedBenchmarkModel()

    response = await model.ainvoke(
        messages=[],
        tools=[
            fake_tool("get_employee"),
            fake_tool("create_ticket"),
        ],
    )

    assert len(response.tool_calls) == 1

    assert response.tool_calls[0]["name"] == "create_ticket"


async def test_returns_message_for_unsupported_task() -> None:
    model = ScriptedBenchmarkModel()

    response = await model.ainvoke(
        messages=[],
        tools=[
            fake_tool("unsupported_tool"),
        ],
    )

    assert response.tool_calls == []

    assert "无法继续执行" in str(response.content)
