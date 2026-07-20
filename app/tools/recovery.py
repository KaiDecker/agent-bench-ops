from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.domain.tickets import Ticket
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import ToolOperation
from app.tools.implementations.tickets import ticket_to_result

type RecoveryStatus = Literal[
    "succeeded",
    "failed",
    "unknown",
]


class RecoveryResolution(BaseModel):
    """某个具体工具的状态恢复结论。"""

    model_config = ConfigDict(extra="forbid")

    status: RecoveryStatus
    result: dict[str, Any] | None = None
    external_reference: str | None = None

    error_type: str | None = None
    error_message: str | None = None

    details: dict[str, Any] = Field(default_factory=dict)


class RecoveryResponse(BaseModel):
    """恢复服务的统一返回结果。"""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    tool_name: str

    previous_status: str
    current_status: str

    recovered: bool
    recovery_count: int

    result: dict[str, Any] | None = None
    external_reference: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


type RecoveryHandler = Callable[
    [AsyncSession, ToolOperation],
    Awaitable[RecoveryResolution],
]


class RecoveryRegistry:
    """按工具名称注册状态恢复逻辑。"""

    def __init__(self) -> None:
        self._handlers: dict[str, RecoveryHandler] = {}

    def register(
        self,
        tool_name: str,
        handler: RecoveryHandler,
    ) -> None:
        if tool_name in self._handlers:
            raise ValueError(f"Recovery handler already registered: {tool_name}")

        self._handlers[tool_name] = handler

    def get(
        self,
        tool_name: str,
    ) -> RecoveryHandler | None:
        return self._handlers.get(tool_name)

    def names(self) -> list[str]:
        return sorted(self._handlers)


async def recover_create_ticket(
    session: AsyncSession,
    operation: ToolOperation,
) -> RecoveryResolution:
    """根据 source_operation_id 检查工单是否已创建。"""

    result = await session.execute(
        select(Ticket).where(Ticket.source_operation_id == operation.operation_id)
    )

    ticket = result.scalar_one_or_none()

    if ticket is None:
        return RecoveryResolution(
            status="failed",
            error_type="recovery_effect_not_found",
            error_message=("No ticket was created for the unknown operation."),
            details={
                "source_operation_id": operation.operation_id,
                "ticket_found": False,
            },
        )

    tool_result = ticket_to_result(ticket)

    return RecoveryResolution(
        status="succeeded",
        result=tool_result.model_dump(mode="json"),
        external_reference=ticket.id,
        details={
            "source_operation_id": operation.operation_id,
            "ticket_found": True,
            "ticket_id": ticket.id,
        },
    )


def build_default_recovery_registry() -> RecoveryRegistry:
    registry = RecoveryRegistry()
    registry.register(
        "create_ticket",
        recover_create_ticket,
    )
    return registry


class OperationRecoveryError(RuntimeError):
    """工具操作无法恢复。"""


class UnknownOperationRecoveryService:
    """恢复处于 unknown 状态的工具操作。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
        registry: RecoveryRegistry | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._registry = registry or build_default_recovery_registry()

    async def recover(
        self,
        operation_id: str,
    ) -> RecoveryResponse:
        async with self._session_factory.begin() as session:
            result = await session.execute(
                select(ToolOperation)
                .where(ToolOperation.operation_id == operation_id)
                .with_for_update()
            )

            operation = result.scalar_one_or_none()

            if operation is None:
                raise OperationRecoveryError(f"Tool operation not found: {operation_id}")

            previous_status = operation.status

            if previous_status != "unknown":
                return RecoveryResponse(
                    operation_id=operation.operation_id,
                    tool_name=operation.tool_name,
                    previous_status=previous_status,
                    current_status=operation.status,
                    recovered=False,
                    recovery_count=operation.recovery_count,
                    result=operation.result,
                    external_reference=(operation.external_reference),
                    details={
                        "reason": "operation_is_not_unknown",
                    },
                )

            handler = self._registry.get(operation.tool_name)
            previous_error_type = operation.error_type

            operation.recovery_count += 1
            operation.recovered_at = datetime.now(UTC)

            if handler is None:
                details = {
                    "reason": "recovery_handler_not_registered",
                    "previous_error_type": previous_error_type,
                }

                operation.recovery_details = details

                return RecoveryResponse(
                    operation_id=operation.operation_id,
                    tool_name=operation.tool_name,
                    previous_status=previous_status,
                    current_status="unknown",
                    recovered=False,
                    recovery_count=operation.recovery_count,
                    result=operation.result,
                    external_reference=(operation.external_reference),
                    details=details,
                )

            resolution = await handler(
                session,
                operation,
            )

            details = {
                "resolver": operation.tool_name,
                "previous_error_type": previous_error_type,
                **resolution.details,
            }

            operation.recovery_details = details

            if resolution.status == "succeeded":
                operation.status = "succeeded"
                operation.result = resolution.result
                operation.external_reference = resolution.external_reference

                operation.error_type = None
                operation.error_message = None
                operation.error_details = None

            elif resolution.status == "failed":
                operation.status = "failed"
                operation.error_type = resolution.error_type
                operation.error_message = resolution.error_message
                operation.error_details = resolution.details

            return RecoveryResponse(
                operation_id=operation.operation_id,
                tool_name=operation.tool_name,
                previous_status=previous_status,
                current_status=operation.status,
                recovered=operation.status != "unknown",
                recovery_count=operation.recovery_count,
                result=operation.result,
                external_reference=operation.external_reference,
                details=details,
            )
