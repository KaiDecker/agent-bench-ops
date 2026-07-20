from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.employees import Employee
from app.tools.schemas import (
    ToolBusinessError,
    ToolDefinition,
    ToolExecutionContext,
    ToolMetadata,
)


class GetEmployeeArguments(BaseModel):
    """查询员工工具的输入参数。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    employee_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
    )

    employee_no: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
    )

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_exactly_one_selector(self) -> Self:
        selectors = [
            self.employee_id,
            self.employee_no,
            self.name,
        ]

        provided_count = sum(selector is not None for selector in selectors)

        if provided_count != 1:
            raise ValueError("Exactly one of employee_id, employee_no or name must be provided")

        return self


class EmployeeResult(BaseModel):
    """查询到的员工信息。"""

    id: str
    employee_no: str
    name: str
    department: str | None
    status: str


class GetEmployeeResult(BaseModel):
    """查询员工工具的输出结构。"""

    employee: EmployeeResult


async def get_employee(
    session: AsyncSession,
    arguments: GetEmployeeArguments,
    context: ToolExecutionContext,
) -> GetEmployeeResult:
    """根据唯一查询条件获取员工。"""

    del context

    statement = select(Employee)

    if arguments.employee_id is not None:
        statement = statement.where(Employee.id == arguments.employee_id)
    elif arguments.employee_no is not None:
        statement = statement.where(Employee.employee_no == arguments.employee_no)
    else:
        statement = statement.where(Employee.name == arguments.name)

    result = await session.execute(statement.order_by(Employee.id).limit(2))

    employees = result.scalars().all()

    if not employees:
        raise ToolBusinessError(
            code="employee_not_found",
            message="No employee matched the provided selector.",
            details=arguments.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )

    if len(employees) > 1:
        raise ToolBusinessError(
            code="employee_ambiguous",
            message=(
                "Multiple employees matched the provided selector. "
                "Use employee_id or employee_no instead."
            ),
            details={"matched_employee_ids": [employee.id for employee in employees]},
        )

    employee = employees[0]

    return GetEmployeeResult(
        employee=EmployeeResult(
            id=employee.id,
            employee_no=employee.employee_no,
            name=employee.name,
            department=employee.department,
            status=employee.status,
        )
    )


GET_EMPLOYEE_TOOL = ToolDefinition(
    metadata=ToolMetadata(
        name="get_employee",
        description=("Query one employee by employee ID, employee number, or employee name."),
        risk_level="low",
        required_permissions={"employee.read"},
        requires_approval=False,
        is_idempotent=True,
        read_only=True,
        timeout_seconds=3.0,
    ),
    arguments_model=GetEmployeeArguments,
    result_model=GetEmployeeResult,
    handler=get_employee,
)
