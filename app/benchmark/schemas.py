from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class EmployeeSeed(BaseModel):
    """员工初始数据。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    employee_no: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=100)
    department: str | None = Field(default=None, max_length=100)
    status: Literal["active", "inactive", "terminated"] = "active"


class AccountSeed(BaseModel):
    """账号初始数据。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    employee_id: str = Field(min_length=1, max_length=64)
    username: str = Field(min_length=1, max_length=100)
    status: Literal["active", "disabled", "locked"] = "active"
    version: int = Field(default=1, gt=0)


class PermissionSeed(BaseModel):
    """权限定义初始数据。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    code: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    risk_level: Literal["low", "medium", "high", "critical"] = "low"
    requires_approval: bool = False


class EmployeePermissionSeed(BaseModel):
    """员工权限关系初始数据。"""

    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(min_length=1, max_length=64)
    permission_id: str = Field(min_length=1, max_length=64)
    status: Literal["active", "revoked"] = "active"
    granted_by: str | None = Field(default=None, max_length=64)
    granted_at: datetime | None = None
    revoked_at: datetime | None = None


class TicketSeed(BaseModel):
    """工单初始数据。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    requester_employee_id: str = Field(min_length=1, max_length=64)
    target_employee_id: str = Field(min_length=1, max_length=64)

    ticket_type: Literal[
        "permission_grant",
        "permission_revoke",
        "account_recovery",
        "general",
    ]

    status: Literal[
        "open",
        "pending_approval",
        "approved",
        "rejected",
        "resolved",
        "cancelled",
    ] = "open"

    risk_level: Literal["low", "medium", "high", "critical"] = "medium"

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    resolution: str | None = None
    version: int = Field(default=1, gt=0)


class BusinessInitialState(BaseModel):
    """一个评测任务需要恢复的完整业务状态。"""

    model_config = ConfigDict(extra="forbid")

    employees: list[EmployeeSeed] = Field(default_factory=list)
    accounts: list[AccountSeed] = Field(default_factory=list)
    permissions: list[PermissionSeed] = Field(default_factory=list)
    employee_permissions: list[EmployeePermissionSeed] = Field(default_factory=list)
    tickets: list[TicketSeed] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> Self:
        """检查唯一性以及业务对象之间的引用关系。"""

        def ensure_unique(values: list[object], label: str) -> None:
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate values found for {label}")

        ensure_unique(
            [employee.id for employee in self.employees],
            "employees.id",
        )
        ensure_unique(
            [employee.employee_no for employee in self.employees],
            "employees.employee_no",
        )
        ensure_unique(
            [account.id for account in self.accounts],
            "accounts.id",
        )
        ensure_unique(
            [account.username for account in self.accounts],
            "accounts.username",
        )
        ensure_unique(
            [account.employee_id for account in self.accounts],
            "accounts.employee_id",
        )
        ensure_unique(
            [permission.id for permission in self.permissions],
            "permissions.id",
        )
        ensure_unique(
            [permission.code for permission in self.permissions],
            "permissions.code",
        )
        ensure_unique(
            [(item.employee_id, item.permission_id) for item in self.employee_permissions],
            "employee_permissions identity",
        )
        ensure_unique(
            [ticket.id for ticket in self.tickets],
            "tickets.id",
        )

        employee_ids = {employee.id for employee in self.employees}
        permission_ids = {permission.id for permission in self.permissions}

        for account in self.accounts:
            if account.employee_id not in employee_ids:
                raise ValueError(f"Account references an unknown employee: {account.employee_id}")

        for item in self.employee_permissions:
            if item.employee_id not in employee_ids:
                raise ValueError(
                    f"Employee permission references an unknown employee: {item.employee_id}"
                )

            if item.permission_id not in permission_ids:
                raise ValueError(
                    f"Employee permission references an unknown permission: {item.permission_id}"
                )

        for ticket in self.tickets:
            if ticket.requester_employee_id not in employee_ids:
                raise ValueError(
                    f"Ticket references an unknown requester: {ticket.requester_employee_id}"
                )

            if ticket.target_employee_id not in employee_ids:
                raise ValueError(
                    f"Ticket references an unknown target: {ticket.target_employee_id}"
                )

        return self


class TaskBudget(BaseModel):
    """单个评测任务允许使用的执行预算。"""

    model_config = ConfigDict(extra="forbid")

    max_agent_steps: int = Field(default=10, gt=0)
    max_tool_calls: int = Field(default=6, gt=0)
    max_tokens: int = Field(default=10_000, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)


class BenchmarkTaskSpec(BaseModel):
    """YAML 中一条固定评测任务的结构。"""

    model_config = ConfigDict(extra="forbid")

    task_key: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    version: int = Field(default=1, gt=0)
    dataset_version: str = Field(default="v1", min_length=1, max_length=50)

    name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=50)
    description: str | None = None
    user_request: str = Field(min_length=1)

    initial_state: BusinessInitialState = Field(default_factory=BusinessInitialState)
    available_tools: list[str] = Field(default_factory=list)

    expected_state: list[dict[str, Any]] = Field(default_factory=list)
    required_events: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_events: list[dict[str, Any]] = Field(default_factory=list)
    temporal_rules: list[dict[str, Any]] = Field(default_factory=list)

    budget: TaskBudget = Field(default_factory=TaskBudget)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

    @field_validator("available_tools")
    @classmethod
    def validate_available_tools(cls, tools: list[str]) -> list[str]:
        normalized_tools = [tool.strip() for tool in tools]

        if any(not tool for tool in normalized_tools):
            raise ValueError("available_tools cannot contain empty names")

        if len(normalized_tools) != len(set(normalized_tools)):
            raise ValueError("available_tools cannot contain duplicates")

        return normalized_tools
