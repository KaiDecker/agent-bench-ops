from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.employees import Employee
from app.domain.tickets import Ticket
from app.tools.schemas import (
    ToolBusinessError,
    ToolDefinition,
    ToolExecutionContext,
    ToolMetadata,
)


class CreateTicketArguments(BaseModel):
    """创建工单的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    requester_employee_id: str = Field(
        min_length=1,
        max_length=64,
    )

    target_employee_id: str = Field(
        min_length=1,
        max_length=64,
    )

    ticket_type: Literal[
        "permission_grant",
        "permission_revoke",
        "account_recovery",
        "general",
    ]

    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] = "medium"

    title: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        min_length=1,
        max_length=5000,
    )


class TicketResult(BaseModel):
    """工单数据。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    requester_employee_id: str
    target_employee_id: str
    ticket_type: str
    status: str
    risk_level: str
    title: str
    description: str
    version: int


class CreateTicketResult(BaseModel):
    """创建工单工具的输出。"""

    model_config = ConfigDict(extra="forbid")

    ticket: TicketResult


class GetTicketArguments(BaseModel):
    """查询工单工具的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    ticket_id: str = Field(
        min_length=1,
        max_length=64,
    )


class GetTicketResult(BaseModel):
    """查询工单工具的输出。"""

    model_config = ConfigDict(extra="forbid")

    ticket: TicketResult


def ticket_to_result(ticket: Ticket) -> CreateTicketResult:
    """将 Ticket ORM 对象转换为工具输出。"""

    return CreateTicketResult(
        ticket=TicketResult(
            id=ticket.id,
            requester_employee_id=ticket.requester_employee_id,
            target_employee_id=ticket.target_employee_id,
            ticket_type=ticket.ticket_type,
            status=ticket.status,
            risk_level=ticket.risk_level,
            title=ticket.title,
            description=ticket.description,
            version=ticket.version,
        )
    )


async def get_ticket(
    session: AsyncSession,
    arguments: GetTicketArguments,
    context: ToolExecutionContext,
) -> GetTicketResult:
    """根据工单 ID 查询一张工单。"""

    del context

    result = await session.execute(select(Ticket).where(Ticket.id == arguments.ticket_id))

    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise ToolBusinessError(
            code="ticket_not_found",
            message="The requested ticket does not exist.",
            details={
                "ticket_id": arguments.ticket_id,
            },
        )

    return GetTicketResult(ticket=ticket_to_result(ticket).ticket)


async def create_ticket(
    session: AsyncSession,
    arguments: CreateTicketArguments,
    context: ToolExecutionContext,
) -> CreateTicketResult:
    """
    创建一张业务工单。

    operation_id 会被保存到 source_operation_id，
    用于幂等执行和 unknown 状态恢复。
    """

    # 相同 operation_id 已经创建过工单时，
    # 直接返回原有结果，避免重复副作用。
    if context.operation_id is not None:
        existing_result = await session.execute(
            select(Ticket).where(Ticket.source_operation_id == context.operation_id)
        )

        existing_ticket = existing_result.scalar_one_or_none()

        if existing_ticket is not None:
            return ticket_to_result(existing_ticket)

    required_employee_ids = {
        arguments.requester_employee_id,
        arguments.target_employee_id,
    }

    employee_result = await session.execute(
        select(Employee.id).where(Employee.id.in_(required_employee_ids))
    )

    existing_employee_ids = set(employee_result.scalars().all())

    missing_employee_ids = sorted(required_employee_ids - existing_employee_ids)

    if missing_employee_ids:
        raise ToolBusinessError(
            code="employee_not_found",
            message=("One or more employees referenced by the ticket do not exist."),
            details={
                "missing_employee_ids": missing_employee_ids,
            },
        )

    ticket = Ticket(
        id=f"ticket_{uuid4().hex[:24]}",
        source_operation_id=context.operation_id,
        requester_employee_id=(arguments.requester_employee_id),
        target_employee_id=arguments.target_employee_id,
        ticket_type=arguments.ticket_type,
        status="open",
        risk_level=arguments.risk_level,
        title=arguments.title,
        description=arguments.description,
        version=1,
    )

    session.add(ticket)
    await session.flush()

    return ticket_to_result(ticket)


CREATE_TICKET_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="create_ticket",
        description=("Create an account, permission, recovery, or general service ticket."),
        risk_level="medium",
        required_permissions={"ticket.write"},
        requires_approval=False,
        is_idempotent=True,
        read_only=False,
        timeout_seconds=5.0,
        external_reference_path="ticket.id",
    ),
    arguments_model=CreateTicketArguments,
    result_model=CreateTicketResult,
    handler=create_ticket,
)

GET_TICKET_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="get_ticket",
        description="Query one service ticket by ticket ID.",
        risk_level="low",
        required_permissions={"ticket.read"},
        requires_approval=False,
        is_idempotent=True,
        read_only=True,
        timeout_seconds=3.0,
    ),
    arguments_model=GetTicketArguments,
    result_model=GetTicketResult,
    handler=get_ticket,
)
