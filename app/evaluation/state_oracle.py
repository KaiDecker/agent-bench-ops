from collections.abc import (
    Mapping,
    Sequence,
)
from datetime import datetime
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.domain.tickets import (
    TicketMutation,
)
from app.evaluation.rules import (
    EvaluationViolation,
    StateAssertion,
    StateEntity,
    StateExpectation,
)
from app.persistence.database import (
    AsyncSessionFactory,
)
from app.persistence.models import (
    Account,
    Employee,
    EmployeePermission,
    Permission,
    Ticket,
)


def normalize_json_value(
    value: Any,
) -> Any:
    """
    递归规范化 JSON 对象。

    字典按键排序；列表保持业务顺序。
    """

    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]

    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]

    return value


def models_from_rows[SnapshotModel: BaseModel](
    model_type: type[SnapshotModel],
    rows: Sequence[Mapping[str, Any]],
) -> list[SnapshotModel]:
    """将数据库 Mapping 行转换成快照模型。"""

    return [model_type.model_validate(dict(row)) for row in rows]


class EmployeeSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    id: str
    employee_no: str
    name: str
    department: str | None
    status: str


class AccountSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    id: str
    employee_id: str
    username: str
    status: str
    version: int


class PermissionSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    id: str
    code: str
    name: str
    description: str | None
    risk_level: str
    requires_approval: bool


class EmployeePermissionSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    employee_id: str
    permission_id: str
    status: str
    granted_by: str | None
    granted_at: datetime
    revoked_at: datetime | None


class TicketSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    id: str
    source_operation_id: str | None
    requester_employee_id: str
    target_employee_id: str
    ticket_type: str
    status: str
    risk_level: str
    title: str
    description: str
    resolution: str | None
    version: int


class TicketMutationSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    id: str
    ticket_id: str
    operation_id: str
    previous_version: int
    new_version: int
    change_payload: dict[str, Any]
    result_snapshot: dict[str, Any]

    @field_validator(
        "change_payload",
        "result_snapshot",
        mode="before",
    )
    @classmethod
    def normalize_payload(
        cls,
        value: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("Ticket mutation payload must be a JSON object")

        normalized = normalize_json_value(value)

        if not isinstance(
            normalized,
            dict,
        ):
            raise ValueError("Normalized mutation payload must remain an object")

        return normalized


class BusinessStateSnapshot(BaseModel):
    """用于状态 Oracle 的完整业务快照。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    employees: list[EmployeeSnapshot] = Field(
        default_factory=list,
    )

    accounts: list[AccountSnapshot] = Field(
        default_factory=list,
    )

    permissions: list[PermissionSnapshot] = Field(
        default_factory=list,
    )

    employee_permissions: list[EmployeePermissionSnapshot] = Field(
        default_factory=list,
    )

    tickets: list[TicketSnapshot] = Field(
        default_factory=list,
    )

    ticket_mutations: list[TicketMutationSnapshot] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def sort_entities(
        self,
    ) -> Self:
        self.employees.sort(key=lambda item: item.id)

        self.accounts.sort(key=lambda item: item.id)

        self.permissions.sort(key=lambda item: item.id)

        self.employee_permissions.sort(
            key=lambda item: (
                item.employee_id,
                item.permission_id,
            )
        )

        self.tickets.sort(key=lambda item: item.id)

        self.ticket_mutations.sort(
            key=lambda item: (
                item.ticket_id,
                item.previous_version,
                item.new_version,
                item.id,
            )
        )

        return self

    def to_json_dict(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
        )


class BusinessStateSnapshotService:
    """从业务数据库读取确定性最终状态。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
    ) -> None:
        self._session_factory = session_factory

    async def capture(
        self,
    ) -> BusinessStateSnapshot:
        """
        在一个数据库事务内采集完整业务快照。
        """

        async with self._session_factory() as session:
            async with session.begin():
                return await self.capture_from_session(session)

    async def capture_from_session(
        self,
        session: AsyncSession,
    ) -> BusinessStateSnapshot:
        """使用调用方提供的 Session 采集快照。"""

        employee_result = await session.execute(
            select(
                Employee.id,
                Employee.employee_no,
                Employee.name,
                Employee.department,
                Employee.status,
            ).order_by(Employee.id)
        )

        account_result = await session.execute(
            select(
                Account.id,
                Account.employee_id,
                Account.username,
                Account.status,
                Account.version,
            ).order_by(Account.id)
        )

        permission_result = await session.execute(
            select(
                Permission.id,
                Permission.code,
                Permission.name,
                Permission.description,
                Permission.risk_level,
                Permission.requires_approval,
            ).order_by(Permission.id)
        )

        employee_permission_result = await session.execute(
            select(
                EmployeePermission.employee_id,
                EmployeePermission.permission_id,
                EmployeePermission.status,
                EmployeePermission.granted_by,
                EmployeePermission.granted_at,
                EmployeePermission.revoked_at,
            ).order_by(
                EmployeePermission.employee_id,
                EmployeePermission.permission_id,
            )
        )

        ticket_result = await session.execute(
            select(
                Ticket.id,
                Ticket.source_operation_id,
                Ticket.requester_employee_id,
                Ticket.target_employee_id,
                Ticket.ticket_type,
                Ticket.status,
                Ticket.risk_level,
                Ticket.title,
                Ticket.description,
                Ticket.resolution,
                Ticket.version,
            ).order_by(Ticket.id)
        )

        mutation_result = await session.execute(
            select(
                TicketMutation.id,
                TicketMutation.ticket_id,
                TicketMutation.operation_id,
                TicketMutation.previous_version,
                TicketMutation.new_version,
                TicketMutation.change_payload,
                TicketMutation.result_snapshot,
            ).order_by(
                TicketMutation.ticket_id,
                TicketMutation.previous_version,
                TicketMutation.new_version,
                TicketMutation.id,
            )
        )

        return BusinessStateSnapshot(
            employees=models_from_rows(
                EmployeeSnapshot,
                employee_result.mappings().all(),
            ),
            accounts=models_from_rows(
                AccountSnapshot,
                account_result.mappings().all(),
            ),
            permissions=models_from_rows(
                PermissionSnapshot,
                permission_result.mappings().all(),
            ),
            employee_permissions=models_from_rows(
                EmployeePermissionSnapshot,
                employee_permission_result.mappings().all(),
            ),
            tickets=models_from_rows(
                TicketSnapshot,
                ticket_result.mappings().all(),
            ),
            ticket_mutations=models_from_rows(
                TicketMutationSnapshot,
                mutation_result.mappings().all(),
            ),
        )


_MISSING = object()


class StateExpectationResult(BaseModel):
    """一条最终状态期望的判定结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    rule_index: int
    entity: StateEntity
    matched_count: int
    passed_assertions: int
    total_assertions: int
    passed: bool


class FinalStateOracleResult(BaseModel):
    """Final State Oracle 的完整结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    passed: bool
    score: float
    passed_assertions: int
    total_assertions: int
    rule_results: list[StateExpectationResult] = Field(
        default_factory=list,
    )
    violations: list[EvaluationViolation] = Field(
        default_factory=list,
    )


def resolve_field(
    record: Mapping[str, Any],
    field_path: str,
) -> tuple[bool, Any]:
    """
    解析点分字段路径。

    返回：
        (字段是否存在, 字段值)
    """

    current: Any = record

    for path_part in field_path.split("."):
        if not isinstance(
            current,
            Mapping,
        ):
            return False, _MISSING

        if path_part not in current:
            return False, _MISSING

        current = current[path_part]

    return True, current


def record_matches_where(
    record: Mapping[str, Any],
    where: Mapping[str, Any],
) -> bool:
    """判断一个业务对象是否符合 where 条件。"""

    for field_path, expected in where.items():
        exists, actual = resolve_field(
            record,
            field_path,
        )

        if not exists:
            return False

        if normalize_json_value(actual) != normalize_json_value(expected):
            return False

    return True


def evaluate_state_assertion(
    *,
    record: Mapping[str, Any],
    assertion: StateAssertion,
) -> tuple[bool, bool, Any]:
    """
    执行单个字段断言。

    返回：
        (断言是否通过, 字段是否存在, 实际值)
    """

    exists, actual = resolve_field(
        record,
        assertion.field,
    )

    if assertion.operator == "exists":
        return exists, exists, (actual if exists else None)

    if assertion.operator == "not_exists":
        return (
            not exists,
            exists,
            actual if exists else None,
        )

    # 除 exists/not_exists 外，
    # 字段不存在都视为断言失败。
    if not exists:
        return False, False, None

    expected = normalize_json_value(assertion.value)
    normalized_actual = normalize_json_value(actual)

    if assertion.operator == "eq":
        passed = normalized_actual == expected

    elif assertion.operator == "ne":
        passed = normalized_actual != expected

    elif assertion.operator == "in":
        passed = normalized_actual in expected

    elif assertion.operator == "not_in":
        passed = normalized_actual not in expected

    else:
        raise ValueError(f"Unsupported state operator: {assertion.operator}")

    return passed, True, normalized_actual


class FinalStateOracle:
    """
    根据结构化业务状态判断任务最终结果。

    每条 StateExpectation 默认要求 where 精确匹配一条记录。
    因此重复创建相同业务对象也会导致评估失败。
    """

    def evaluate(
        self,
        *,
        snapshot: BusinessStateSnapshot,
        expectations: Sequence[StateExpectation],
    ) -> FinalStateOracleResult:
        state_payload = snapshot.to_json_dict()

        violations: list[EvaluationViolation] = []

        rule_results: list[StateExpectationResult] = []

        total_assertions = sum(len(expectation.assertions) for expectation in expectations)

        passed_assertions = 0

        for rule_index, expectation in enumerate(expectations):
            entity_records = state_payload[expectation.entity]

            matches = [
                record
                for record in entity_records
                if record_matches_where(
                    record,
                    expectation.where,
                )
            ]

            rule_assertion_count = len(expectation.assertions)

            if len(matches) == 0:
                violations.append(
                    EvaluationViolation(
                        oracle="state",
                        code=("state_entity_not_found"),
                        message=("No business entity matched the state expectation."),
                        rule_index=rule_index,
                        details={
                            "entity": (expectation.entity),
                            "where": (expectation.where),
                            "matched_count": 0,
                        },
                    )
                )

                rule_results.append(
                    StateExpectationResult(
                        rule_index=rule_index,
                        entity=(expectation.entity),
                        matched_count=0,
                        passed_assertions=0,
                        total_assertions=(rule_assertion_count),
                        passed=False,
                    )
                )

                continue

            if len(matches) > 1:
                violations.append(
                    EvaluationViolation(
                        oracle="state",
                        code=("state_entity_ambiguous"),
                        message=("More than one business entity matched the state expectation."),
                        rule_index=rule_index,
                        details={
                            "entity": (expectation.entity),
                            "where": (expectation.where),
                            "matched_count": (len(matches)),
                        },
                    )
                )

                rule_results.append(
                    StateExpectationResult(
                        rule_index=rule_index,
                        entity=(expectation.entity),
                        matched_count=len(matches),
                        passed_assertions=0,
                        total_assertions=(rule_assertion_count),
                        passed=False,
                    )
                )

                continue

            record = matches[0]
            rule_passed_assertions = 0

            for assertion in expectation.assertions:
                (
                    assertion_passed,
                    field_exists,
                    actual_value,
                ) = evaluate_state_assertion(
                    record=record,
                    assertion=assertion,
                )

                if assertion_passed:
                    rule_passed_assertions += 1
                    passed_assertions += 1
                    continue

                details: dict[str, Any] = {
                    "entity": (expectation.entity),
                    "where": (expectation.where),
                    "field": assertion.field,
                    "operator": (assertion.operator),
                    "field_exists": (field_exists),
                    "actual_value": (actual_value),
                }

                if "value" in assertion.model_fields_set:
                    details["expected_value"] = assertion.value

                violations.append(
                    EvaluationViolation(
                        oracle="state",
                        code=("state_assertion_failed"),
                        message=(
                            "A final-state assertion did not match the actual business state."
                        ),
                        rule_index=rule_index,
                        details=details,
                    )
                )

            rule_results.append(
                StateExpectationResult(
                    rule_index=rule_index,
                    entity=expectation.entity,
                    matched_count=1,
                    passed_assertions=(rule_passed_assertions),
                    total_assertions=(rule_assertion_count),
                    passed=(rule_passed_assertions == rule_assertion_count),
                )
            )

        score = 1.0

        if total_assertions > 0:
            score = round(
                passed_assertions / total_assertions,
                6,
            )

        return FinalStateOracleResult(
            passed=not violations,
            score=score,
            passed_assertions=(passed_assertions),
            total_assertions=(total_assertions),
            rule_results=rule_results,
            violations=violations,
        )


__all__ = [
    "AccountSnapshot",
    "BusinessStateSnapshot",
    "BusinessStateSnapshotService",
    "EmployeePermissionSnapshot",
    "EmployeeSnapshot",
    "FinalStateOracle",
    "FinalStateOracleResult",
    "PermissionSnapshot",
    "StateExpectationResult",
    "TicketMutationSnapshot",
    "TicketSnapshot",
    "evaluate_state_assertion",
    "models_from_rows",
    "normalize_json_value",
    "record_matches_where",
    "resolve_field",
]
