import json

import pytest
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)

from app.agent.model import (
    ScriptedCreateTicketModel,
)
from app.tools.registry import (
    build_default_registry,
)


@pytest.mark.asyncio
async def test_create_ticket_model_requests_tool() -> None:
    registry = build_default_registry()
    model = ScriptedCreateTicketModel()

    result = await model.ainvoke(
        messages=[HumanMessage(content="创建普通服务工单。")],
        tools=[registry.get("create_ticket")],
    )

    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]

    assert tool_call["name"] == "create_ticket"
    assert tool_call["id"] == ("call_create_ticket_001")

    assert tool_call["args"] == {
        "requester_employee_id": "emp_001",
        "target_employee_id": "emp_002",
        "ticket_type": "general",
        "risk_level": "medium",
        "title": "数据平台访问问题",
        "description": ("李四无法访问数据平台，请协助检查账号权限"),
    }


@pytest.mark.asyncio
async def test_create_ticket_model_returns_final_answer() -> None:
    registry = build_default_registry()
    model = ScriptedCreateTicketModel()

    result = await model.ainvoke(
        messages=[
            HumanMessage(content="创建普通服务工单。"),
            ToolMessage(
                tool_call_id=("call_create_ticket_001"),
                content=json.dumps(
                    {
                        "status": "succeeded",
                        "output": {
                            "ticket": {
                                "id": "ticket_test_001",
                                "status": "open",
                                "title": ("数据平台访问问题"),
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        tools=[registry.get("create_ticket")],
    )

    assert result.tool_calls == []
    assert "ticket_test_001" in str(result.content)
    assert "open" in str(result.content)
