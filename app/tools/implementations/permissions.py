from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.employees import Employee
from app.domain.permissions import (
    EmployeePermission,
    Permission,
)
from app.tools.schemas import (
    ToolBusinessError,
    ToolDefinition,
    ToolExecutionContext,
    ToolMetadata,
)


class ListEmployeePermissionsArguments(BaseModel):
    """查询员工权限工具的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    employee_id: str = Field(
        min_length=1,
        max_length=64,
    )

    include_revoked: bool = False


class EmployeePermissionResult(BaseModel):
    """一条员工权限分配记录。"""

    model_config = ConfigDict(extra="forbid")

    permission_id: str
    code: str
    name: str
    description: str | None
    risk_level: str
    requires_approval: bool

    status: str
    granted_by: str | None
    granted_at: datetime | None
    revoked_at: datetime | None


class ListEmployeePermissionsResult(BaseModel):
    """员工权限列表。"""

    model_config = ConfigDict(extra="forbid")

    employee_id: str
    permissions: list[EmployeePermissionResult]
    count: int


async def list_employee_permissions(
    session: AsyncSession,
    arguments: ListEmployeePermissionsArguments,
    context: ToolExecutionContext,
) -> ListEmployeePermissionsResult:
    """查询指定员工当前或历史权限。"""

    del context

    employee_result = await session.execute(
        select(Employee.id).where(Employee.id == arguments.employee_id)
    )

    employee_id = employee_result.scalar_one_or_none()

    if employee_id is None:
        raise ToolBusinessError(
            code="employee_not_found",
            message="The requested employee does not exist.",
            details={
                "employee_id": arguments.employee_id,
            },
        )

    statement = (
        select(
            EmployeePermission,
            Permission,
        )
        .join(
            Permission,
            Permission.id == EmployeePermission.permission_id,
        )
        .where(EmployeePermission.employee_id == arguments.employee_id)
        .order_by(Permission.code)
    )

    if not arguments.include_revoked:
        statement = statement.where(EmployeePermission.status == "active")

    result = await session.execute(statement)
    rows = result.all()

    permissions = [
        EmployeePermissionResult(
            permission_id=permission.id,
            code=permission.code,
            name=permission.name,
            description=permission.description,
            risk_level=permission.risk_level,
            requires_approval=permission.requires_approval,
            status=assignment.status,
            granted_by=assignment.granted_by,
            granted_at=assignment.granted_at,
            revoked_at=assignment.revoked_at,
        )
        for assignment, permission in rows
    ]

    return ListEmployeePermissionsResult(
        employee_id=arguments.employee_id,
        permissions=permissions,
        count=len(permissions),
    )


LIST_EMPLOYEE_PERMISSIONS_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="list_employee_permissions",
        description=("List active or historical permissions assigned to an employee."),
        risk_level="low",
        required_permissions={"permission.read"},
        requires_approval=False,
        is_idempotent=True,
        read_only=True,
        timeout_seconds=3.0,
    ),
    arguments_model=ListEmployeePermissionsArguments,
    result_model=ListEmployeePermissionsResult,
    handler=list_employee_permissions,
)
