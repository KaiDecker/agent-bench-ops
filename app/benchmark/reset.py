from dataclasses import dataclass

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.benchmark.schemas import BusinessInitialState
from app.persistence.models import (
    Account,
    Employee,
    EmployeePermission,
    Permission,
    Ticket,
)
from app.persistence.platform_models import ToolOperation

UNRESOLVED_TOOL_OPERATION_STATUSES = (
    "prepared",
    "running",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class UnresolvedToolOperation:
    operation_id: str
    run_id: str
    tool_name: str
    status: str


class BusinessStateResetBlockedError(RuntimeError):
    """存在未决工具操作时拒绝重置业务状态。"""

    def __init__(
        self,
        operations: list[UnresolvedToolOperation],
    ) -> None:
        self.operations = tuple(operations)

        operation_ids = ", ".join(operation.operation_id for operation in operations)

        super().__init__(
            "Business state reset is blocked because "
            "unresolved tool operations exist: "
            f"{operation_ids}"
        )


async def assert_business_state_reset_safe(
    session: AsyncSession,
) -> None:
    """
    确认当前不存在依赖业务表进行恢复的未决操作。

    PostgreSQL 下先获取 SHARE 表锁：
    - 阻止其他事务并发插入或更新 tool_operations；
    - 等待正在修改账本的事务结束；
    - 在同一事务内检查状态并执行 reset。
    """

    bind = session.get_bind()

    if bind.dialect.name == "postgresql":
        await session.execute(text("LOCK TABLE tool_operations IN SHARE MODE"))

    result = await session.execute(
        select(
            ToolOperation.operation_id,
            ToolOperation.run_id,
            ToolOperation.tool_name,
            ToolOperation.status,
        )
        .where(ToolOperation.status.in_(UNRESOLVED_TOOL_OPERATION_STATUSES))
        .order_by(ToolOperation.created_at)
    )

    unresolved = [
        UnresolvedToolOperation(
            operation_id=row.operation_id,
            run_id=row.run_id,
            tool_name=row.tool_name,
            status=row.status,
        )
        for row in result
    ]

    if unresolved:
        raise BusinessStateResetBlockedError(unresolved)


async def reset_business_state(
    session: AsyncSession,
    state: BusinessInitialState,
) -> None:
    """
    清空并恢复模拟业务状态。

    调用方必须负责开启数据库事务。
    """

    await assert_business_state_reset_safe(session)

    # 按外键依赖的反方向删除。
    for model in (
        Ticket,
        EmployeePermission,
        Account,
        Permission,
        Employee,
    ):
        await session.execute(delete(model))

    # 按外键依赖的正方向插入。
    session.add_all([Employee(**item.model_dump()) for item in state.employees])

    session.add_all([Permission(**item.model_dump()) for item in state.permissions])

    await session.flush()

    session.add_all([Account(**item.model_dump()) for item in state.accounts])

    employee_permissions: list[EmployeePermission] = []

    for item in state.employee_permissions:
        values = item.model_dump(exclude_none=True)

        employee_permissions.append(EmployeePermission(**values))

    session.add_all(employee_permissions)

    session.add_all([Ticket(**item.model_dump()) for item in state.tickets])

    await session.flush()


__all__ = [
    "BusinessStateResetBlockedError",
    "UNRESOLVED_TOOL_OPERATION_STATUSES",
    "UnresolvedToolOperation",
    "assert_business_state_reset_safe",
    "reset_business_state",
]
