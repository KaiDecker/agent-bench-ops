from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import build_agent_graph
from app.agent.model import (
    ScriptedEmployeeLookupModel,
)
from app.agent.state import build_initial_state
from app.tools.registry import build_default_registry
from app.tools.schemas import (
    ToolExecutionContext,
    ToolExecutionResponse,
)


class FakeToolGateway:
    """不访问数据库的 Agent Graph 测试替身。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        session: AsyncSession,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        idempotency_key: str | None = None,
    ) -> ToolExecutionResponse:
        del session

        self.calls.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "context": context,
                "idempotency_key": idempotency_key,
            }
        )

        return ToolExecutionResponse(
            tool_name=tool_name,
            status="succeeded",
            output={
                "employee": {
                    "id": "emp_001",
                    "employee_no": "E10001",
                    "name": "张三",
                    "department": "数据平台部",
                    "status": "active",
                }
            },
            latency_ms=1.0,
        )


@asynccontextmanager
async def fake_session_factory() -> AsyncIterator[AsyncSession]:
    yield cast(AsyncSession, object())


async def test_agent_graph_executes_tool_loop() -> None:
    gateway = FakeToolGateway()

    graph = build_agent_graph(
        model=ScriptedEmployeeLookupModel(),
        registry=build_default_registry(),
        gateway=gateway,
        session_factory=fake_session_factory,
    )

    result = await graph.ainvoke(
        build_initial_state(
            user_request=("请查询张三的员工号、部门和状态。"),
            run_id=None,
            actor_id="unit-test",
            available_tools=["get_employee"],
            permissions=["employee.read"],
            max_steps=3,
            max_tool_calls=2,
        ),
        config={
            "recursion_limit": 10,
        },
    )

    assert result["error"] is None
    assert result["step_count"] == 2
    assert result["tool_call_count"] == 1

    assert result["final_response"] == (
        "张三的员工号是 E10001，部门是 数据平台部，当前状态是 active。"
    )

    assert len(gateway.calls) == 1
    assert gateway.calls[0]["tool_name"] == ("get_employee")
    assert gateway.calls[0]["arguments"] == {
        "name": "张三",
    }


async def test_agent_graph_enforces_tool_budget() -> None:
    gateway = FakeToolGateway()

    graph = build_agent_graph(
        model=ScriptedEmployeeLookupModel(),
        registry=build_default_registry(),
        gateway=gateway,
        session_factory=fake_session_factory,
    )

    result = await graph.ainvoke(
        build_initial_state(
            user_request=("请查询张三的员工号、部门和状态。"),
            run_id=None,
            actor_id="unit-test",
            available_tools=["get_employee"],
            permissions=["employee.read"],
            max_steps=3,
            max_tool_calls=0,
        ),
        config={
            "recursion_limit": 10,
        },
    )

    assert result["error"] is not None
    assert result["error"]["code"] == ("agent_tool_budget_exceeded")

    assert result["tool_call_count"] == 0
    assert gateway.calls == []
