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

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=5000)


class TicketResult(BaseModel):
    """创建后的工单数据。"""

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

    ticket: TicketResult


async def create_ticket(
    session: AsyncSession,
    arguments: CreateTicketArguments,
    context: ToolExecutionContext,
) -> CreateTicketResult:
    """创建一张业务工单。"""

    del context

    required_employee_ids = {
        arguments.requester_employee_id,
        arguments.target_employee_id,
    }

    result = await session.execute(
        select(Employee.id).where(Employee.id.in_(required_employee_ids))
    )

    existing_employee_ids = set(result.scalars().all())
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
        requester_employee_id=arguments.requester_employee_id,
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


CREATE_TICKET_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="create_ticket",
        description=("Create an account, permission, recovery, or general service ticket."),
        risk_level="medium",
        required_permissions={"ticket.write"},
        requires_approval=False,
        is_idempotent=False,
        read_only=False,
        timeout_seconds=5.0,
    ),
    arguments_model=CreateTicketArguments,
    result_model=CreateTicketResult,
    handler=create_ticket,
)
