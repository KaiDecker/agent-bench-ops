from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.benchmark.schemas import BusinessInitialState
from app.domain.accounts import Account
from app.domain.employees import Employee
from app.domain.permissions import EmployeePermission, Permission
from app.domain.tickets import Ticket


async def reset_business_state(
    session: AsyncSession,
    state: BusinessInitialState,
) -> None:
    """
    清空并恢复模拟业务状态。

    调用方必须负责开启数据库事务。
    """

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


async def capture_business_state(
    session: AsyncSession,
) -> dict[str, list[dict[str, Any]]]:
    """读取当前业务数据库的确定性状态快照。"""

    employees = (await session.execute(select(Employee).order_by(Employee.id))).scalars().all()

    accounts = (await session.execute(select(Account).order_by(Account.id))).scalars().all()

    permissions = (
        (await session.execute(select(Permission).order_by(Permission.id))).scalars().all()
    )

    employee_permissions = (
        (
            await session.execute(
                select(EmployeePermission).order_by(
                    EmployeePermission.employee_id,
                    EmployeePermission.permission_id,
                )
            )
        )
        .scalars()
        .all()
    )

    tickets = (await session.execute(select(Ticket).order_by(Ticket.id))).scalars().all()

    return {
        "employees": [
            {
                "id": item.id,
                "employee_no": item.employee_no,
                "name": item.name,
                "department": item.department,
                "status": item.status,
            }
            for item in employees
        ],
        "accounts": [
            {
                "id": item.id,
                "employee_id": item.employee_id,
                "username": item.username,
                "status": item.status,
                "version": item.version,
            }
            for item in accounts
        ],
        "permissions": [
            {
                "id": item.id,
                "code": item.code,
                "name": item.name,
                "description": item.description,
                "risk_level": item.risk_level,
                "requires_approval": item.requires_approval,
            }
            for item in permissions
        ],
        "employee_permissions": [
            {
                "employee_id": item.employee_id,
                "permission_id": item.permission_id,
                "status": item.status,
                "granted_by": item.granted_by,
                "revoked_at": (
                    item.revoked_at.isoformat() if item.revoked_at is not None else None
                ),
            }
            for item in employee_permissions
        ],
        "tickets": [
            {
                "id": item.id,
                "requester_employee_id": item.requester_employee_id,
                "target_employee_id": item.target_employee_id,
                "ticket_type": item.ticket_type,
                "status": item.status,
                "risk_level": item.risk_level,
                "title": item.title,
                "description": item.description,
                "resolution": item.resolution,
                "version": item.version,
            }
            for item in tickets
        ],
    }


def normalize_initial_state(
    state: BusinessInitialState,
) -> dict[str, list[dict[str, Any]]]:
    """
    将 YAML 初始状态转换为可与数据库快照比较的结构。

    排除数据库自动生成的 created_at、updated_at 和 granted_at。
    """

    return {
        "employees": sorted(
            [item.model_dump(mode="json") for item in state.employees],
            key=lambda item: item["id"],
        ),
        "accounts": sorted(
            [item.model_dump(mode="json") for item in state.accounts],
            key=lambda item: item["id"],
        ),
        "permissions": sorted(
            [item.model_dump(mode="json") for item in state.permissions],
            key=lambda item: item["id"],
        ),
        "employee_permissions": sorted(
            [
                {
                    "employee_id": item.employee_id,
                    "permission_id": item.permission_id,
                    "status": item.status,
                    "granted_by": item.granted_by,
                    "revoked_at": (
                        item.revoked_at.isoformat() if item.revoked_at is not None else None
                    ),
                }
                for item in state.employee_permissions
            ],
            key=lambda item: (
                item["employee_id"],
                item["permission_id"],
            ),
        ),
        "tickets": sorted(
            [item.model_dump(mode="json") for item in state.tickets],
            key=lambda item: item["id"],
        ),
    }
