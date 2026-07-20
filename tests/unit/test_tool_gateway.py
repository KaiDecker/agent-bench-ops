from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolExecutionContext


async def test_gateway_rejects_tool_outside_task_scope() -> None:
    gateway = ToolGateway(build_default_registry())
    session = AsyncMock(spec=AsyncSession)

    response = await gateway.execute(
        session=session,
        tool_name="get_employee",
        arguments={"employee_id": "emp_001"},
        context=ToolExecutionContext(
            available_tools=set(),
            permissions={"employee.read"},
        ),
    )

    assert response.status == "rejected"
    assert response.error is not None
    assert response.error.code == "tool_not_allowed"


async def test_gateway_rejects_missing_permission() -> None:
    gateway = ToolGateway(build_default_registry())
    session = AsyncMock(spec=AsyncSession)

    response = await gateway.execute(
        session=session,
        tool_name="get_employee",
        arguments={"employee_id": "emp_001"},
        context=ToolExecutionContext(
            available_tools={"get_employee"},
            permissions=set(),
        ),
    )

    assert response.status == "rejected"
    assert response.error is not None
    assert response.error.code == "permission_denied"


async def test_gateway_rejects_invalid_arguments() -> None:
    gateway = ToolGateway(build_default_registry())
    session = AsyncMock(spec=AsyncSession)

    response = await gateway.execute(
        session=session,
        tool_name="get_employee",
        arguments={
            "employee_id": "emp_001",
            "name": "张三",
        },
        context=ToolExecutionContext(
            available_tools={"get_employee"},
            permissions={"employee.read"},
        ),
    )

    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "invalid_arguments"


async def test_gateway_rejects_unknown_tool() -> None:
    gateway = ToolGateway(build_default_registry())
    session = AsyncMock(spec=AsyncSession)

    response = await gateway.execute(
        session=session,
        tool_name="unknown_tool",
        arguments={},
        context=ToolExecutionContext(),
    )

    assert response.status == "rejected"
    assert response.error is not None
    assert response.error.code == "tool_not_registered"
