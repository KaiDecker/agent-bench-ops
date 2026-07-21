from typing import Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.employees import Employee
from app.domain.tickets import Ticket, TicketMutation
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
    resolution: str | None = None
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
            requester_employee_id=(ticket.requester_employee_id),
            target_employee_id=(ticket.target_employee_id),
            ticket_type=ticket.ticket_type,
            status=ticket.status,
            risk_level=ticket.risk_level,
            title=ticket.title,
            description=ticket.description,
            resolution=ticket.resolution,
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

type MutableTicketStatus = Literal[
    "open",
    "in_progress",
    "resolved",
    "closed",
    "cancelled",
]


ALLOWED_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "open": {
        "in_progress",
        "resolved",
        "cancelled",
    },
    "in_progress": {
        "open",
        "resolved",
        "cancelled",
    },
    "resolved": {
        "in_progress",
        "closed",
    },
    "closed": set(),
    "cancelled": set(),
}


class UpdateTicketArguments(BaseModel):
    """更新工单的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    ticket_id: str = Field(
        min_length=1,
        max_length=64,
    )

    expected_version: int = Field(
        ge=1,
    )

    status: MutableTicketStatus | None = None

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    description: str | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
    )

    resolution: str | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
    )

    @model_validator(mode="after")
    def validate_changes_are_present(self) -> Self:
        change_fields = {
            "status",
            "title",
            "description",
            "resolution",
        }

        if not self.model_fields_set.intersection(change_fields):
            raise ValueError("At least one ticket field must be updated")

        return self


class UpdateTicketResult(BaseModel):
    """更新工单工具的输出。"""

    model_config = ConfigDict(extra="forbid")

    ticket: TicketResult
    previous_version: int
    updated_fields: list[str]


async def update_ticket(
    session: AsyncSession,
    arguments: UpdateTicketArguments,
    context: ToolExecutionContext,
) -> UpdateTicketResult:
    """
    使用乐观锁更新工单。

    expected_version 必须与当前数据库版本一致。
    同一 operation_id 已经成功更新时，直接返回审计快照。
    """

    if context.operation_id is not None:
        existing_mutation_result = await session.execute(
            select(TicketMutation).where(TicketMutation.operation_id == context.operation_id)
        )

        existing_mutation = existing_mutation_result.scalar_one_or_none()

        if existing_mutation is not None:
            return UpdateTicketResult.model_validate(existing_mutation.result_snapshot)

    current_result = await session.execute(select(Ticket).where(Ticket.id == arguments.ticket_id))

    current_ticket = current_result.scalar_one_or_none()

    if current_ticket is None:
        raise ToolBusinessError(
            code="ticket_not_found",
            message="The requested ticket does not exist.",
            details={
                "ticket_id": arguments.ticket_id,
            },
        )

    if current_ticket.version != arguments.expected_version:
        raise ToolBusinessError(
            code="ticket_version_conflict",
            message=("The ticket was modified by another operation."),
            details={
                "ticket_id": arguments.ticket_id,
                "expected_version": (arguments.expected_version),
                "current_version": current_ticket.version,
            },
        )

    requested_changes = arguments.model_dump(
        mode="python",
        exclude={
            "ticket_id",
            "expected_version",
        },
        exclude_unset=True,
    )

    effective_changes = {
        field_name: field_value
        for field_name, field_value in requested_changes.items()
        if getattr(current_ticket, field_name) != field_value
    }

    if not effective_changes:
        raise ToolBusinessError(
            code="no_ticket_changes",
            message=("The requested values are already present on the ticket."),
            details={
                "ticket_id": arguments.ticket_id,
                "version": current_ticket.version,
            },
        )

    new_status = effective_changes.get("status")

    if isinstance(new_status, str):
        allowed_statuses = ALLOWED_STATUS_TRANSITIONS.get(
            current_ticket.status,
            set(),
        )

        if new_status not in allowed_statuses:
            raise ToolBusinessError(
                code="invalid_ticket_status_transition",
                message=("The requested ticket status transition is not allowed."),
                details={
                    "ticket_id": arguments.ticket_id,
                    "current_status": current_ticket.status,
                    "requested_status": new_status,
                    "allowed_statuses": sorted(allowed_statuses),
                },
            )

    statement = (
        update(Ticket)
        .where(
            Ticket.id == arguments.ticket_id,
            Ticket.version == arguments.expected_version,
        )
        .values(
            **effective_changes,
            version=Ticket.version + 1,
        )
        .returning(
            Ticket.id,
            Ticket.requester_employee_id,
            Ticket.target_employee_id,
            Ticket.ticket_type,
            Ticket.status,
            Ticket.risk_level,
            Ticket.title,
            Ticket.description,
            Ticket.resolution,
            Ticket.version,
        )
    )

    update_result = await session.execute(statement)
    updated_row = update_result.mappings().one_or_none()

    if updated_row is None:
        latest_result = await session.execute(
            select(Ticket.version).where(Ticket.id == arguments.ticket_id)
        )

        latest_version = latest_result.scalar_one_or_none()

        if latest_version is None:
            raise ToolBusinessError(
                code="ticket_not_found",
                message=("The ticket disappeared before the update completed."),
                details={
                    "ticket_id": arguments.ticket_id,
                },
            )

        raise ToolBusinessError(
            code="ticket_version_conflict",
            message=("The ticket was modified concurrently."),
            details={
                "ticket_id": arguments.ticket_id,
                "expected_version": (arguments.expected_version),
                "current_version": latest_version,
            },
        )

    ticket_result = TicketResult.model_validate(dict(updated_row))

    tool_result = UpdateTicketResult(
        ticket=ticket_result,
        previous_version=arguments.expected_version,
        updated_fields=sorted(effective_changes),
    )

    if context.operation_id is not None:
        mutation = TicketMutation(
            id=f"mutation_{uuid4().hex[:24]}",
            ticket_id=ticket_result.id,
            operation_id=context.operation_id,
            previous_version=arguments.expected_version,
            new_version=ticket_result.version,
            change_payload=effective_changes,
            result_snapshot=tool_result.model_dump(mode="json"),
        )

        session.add(mutation)
        await session.flush()

    return tool_result


UPDATE_TICKET_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="update_ticket",
        description=("Update an existing ticket using optimistic version control."),
        risk_level="medium",
        required_permissions={"ticket.write"},
        requires_approval=False,
        is_idempotent=True,
        read_only=False,
        timeout_seconds=5.0,
        external_reference_path="ticket.id",
    ),
    arguments_model=UpdateTicketArguments,
    result_model=UpdateTicketResult,
    handler=update_ticket,
)
