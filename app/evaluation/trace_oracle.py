from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, Self

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

from app.evaluation.rules import (
    EvaluationViolation,
    TraceEventRule,
    TraceEventType,
)
from app.evaluation.state_oracle import (
    models_from_rows,
    normalize_json_value,
)
from app.persistence.database import (
    AsyncSessionFactory,
)
from app.persistence.platform_models import (
    RunStep,
    ToolOperation,
)

type TraceRuleKind = Literal[
    "required",
    "forbidden",
]


def normalize_json_object(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """规范化必填 JSON 对象。"""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")

    normalized = normalize_json_value(value)

    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must remain a JSON object")

    return normalized


def normalize_optional_json_object(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    """规范化可空 JSON 对象。"""

    if value is None:
        return None

    return normalize_json_object(
        value,
        field_name=field_name,
    )


class RunStepTrace(BaseModel):
    """一个持久化 Agent 执行步骤。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    id: str
    parent_step_id: str | None
    step_no: int
    step_type: str
    status: str
    model_name: str | None
    tool_name: str | None
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    latency_ms: float | None
    error_type: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator(
        "input_payload",
        mode="before",
    )
    @classmethod
    def normalize_input_payload(
        cls,
        value: Any,
    ) -> dict[str, Any]:
        return normalize_json_object(
            value,
            field_name="input_payload",
        )

    @field_validator(
        "output_payload",
        mode="before",
    )
    @classmethod
    def normalize_output_payload(
        cls,
        value: Any,
    ) -> dict[str, Any] | None:
        return normalize_optional_json_object(
            value,
            field_name="output_payload",
        )


class ToolOperationTrace(BaseModel):
    """一个结构化工具操作事实。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    operation_id: str
    step_id: str | None
    tool_name: str
    arguments: dict[str, Any]
    arguments_hash: str
    idempotency_key: str
    risk_level: str
    requires_approval: bool
    is_idempotent: bool
    status: str
    retry_count: int
    recovery_count: int
    recovery_details: dict[str, Any] | None
    result: dict[str, Any] | None
    latency_ms: float | None
    external_reference: str | None
    error_type: str | None
    error_message: str | None
    error_details: dict[str, Any] | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime

    @field_validator(
        "arguments",
        mode="before",
    )
    @classmethod
    def normalize_arguments(
        cls,
        value: Any,
    ) -> dict[str, Any]:
        return normalize_json_object(
            value,
            field_name="arguments",
        )

    @field_validator(
        "recovery_details",
        "result",
        "error_details",
        mode="before",
    )
    @classmethod
    def normalize_optional_payload(
        cls,
        value: Any,
    ) -> dict[str, Any] | None:
        return normalize_optional_json_object(
            value,
            field_name="tool operation payload",
        )


class TraceSnapshot(BaseModel):
    """一次 AgentRun 的结构化执行轨迹。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    run_id: str

    steps: list[RunStepTrace] = Field(
        default_factory=list,
    )

    operations: list[ToolOperationTrace] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def sort_trace(
        self,
    ) -> Self:
        self.steps.sort(
            key=lambda step: (
                step.step_no,
                step.id,
            )
        )

        self.operations.sort(
            key=lambda operation: (
                operation.created_at,
                operation.operation_id,
            )
        )

        return self

    def to_json_dict(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
        )


class TraceRuleResult(BaseModel):
    """一条轨迹规则的判定结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    rule_kind: TraceRuleKind
    rule_index: int
    event: TraceEventType
    tool_name: str
    observed_count: int
    matching_operation_ids: list[str] = Field(
        default_factory=list,
    )
    passed: bool


class TraceOracleResult(BaseModel):
    """Trace Oracle 的完整结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    passed: bool
    score: float
    passed_rules: int
    total_rules: int

    rule_results: list[TraceRuleResult] = Field(
        default_factory=list,
    )

    violations: list[EvaluationViolation] = Field(
        default_factory=list,
    )


def operation_event_count(
    operation: ToolOperationTrace,
    event: TraceEventType,
) -> int:
    """计算单条 ToolOperation 对某类事件贡献的次数。"""

    if event == "tool_called":
        return 1 + operation.retry_count

    if event == "tool_replayed":
        return operation.retry_count

    if event == "tool_succeeded":
        return int(operation.status == "succeeded")

    if event == "tool_failed":
        return int(operation.status == "failed")

    if event == "tool_rejected":
        return int(operation.status == "rejected")

    raise ValueError(f"Unsupported trace event: {event}")


def count_trace_event(
    *,
    operations: Sequence[ToolOperationTrace],
    rule: TraceEventRule,
) -> tuple[int, list[str]]:
    """统计符合规则的轨迹事件。"""

    observed_count = 0
    matching_operation_ids: list[str] = []

    for operation in operations:
        if operation.tool_name != rule.tool_name:
            continue

        contribution = operation_event_count(
            operation,
            rule.event,
        )

        if contribution <= 0:
            continue

        observed_count += contribution
        matching_operation_ids.append(operation.operation_id)

    return (
        observed_count,
        matching_operation_ids,
    )


class TraceOracle:
    """根据 ToolOperation 事实判定轨迹规则。"""

    def evaluate(
        self,
        *,
        snapshot: TraceSnapshot,
        required_events: Sequence[TraceEventRule],
        forbidden_events: Sequence[TraceEventRule],
    ) -> TraceOracleResult:
        violations: list[EvaluationViolation] = []

        rule_results: list[TraceRuleResult] = []

        passed_rules = 0

        for rule_index, rule in enumerate(required_events):
            (
                observed_count,
                operation_ids,
            ) = count_trace_event(
                operations=snapshot.operations,
                rule=rule,
            )

            passed = observed_count > 0

            if passed:
                passed_rules += 1
            else:
                violations.append(
                    EvaluationViolation(
                        oracle="trace",
                        code=("required_trace_event_missing"),
                        message=("A required trace event was not observed."),
                        rule_index=rule_index,
                        details={
                            "event": rule.event,
                            "tool_name": (rule.tool_name),
                            "observed_count": 0,
                        },
                    )
                )

            rule_results.append(
                TraceRuleResult(
                    rule_kind="required",
                    rule_index=rule_index,
                    event=rule.event,
                    tool_name=rule.tool_name,
                    observed_count=(observed_count),
                    matching_operation_ids=(operation_ids),
                    passed=passed,
                )
            )

        for rule_index, rule in enumerate(forbidden_events):
            (
                observed_count,
                operation_ids,
            ) = count_trace_event(
                operations=snapshot.operations,
                rule=rule,
            )

            passed = observed_count == 0

            if passed:
                passed_rules += 1
            else:
                violations.append(
                    EvaluationViolation(
                        oracle="trace",
                        code=("forbidden_trace_event_observed"),
                        message=("A forbidden trace event was observed."),
                        rule_index=rule_index,
                        details={
                            "event": rule.event,
                            "tool_name": (rule.tool_name),
                            "observed_count": (observed_count),
                            "operation_ids": (operation_ids),
                        },
                    )
                )

            rule_results.append(
                TraceRuleResult(
                    rule_kind="forbidden",
                    rule_index=rule_index,
                    event=rule.event,
                    tool_name=rule.tool_name,
                    observed_count=(observed_count),
                    matching_operation_ids=(operation_ids),
                    passed=passed,
                )
            )

        total_rules = len(required_events) + len(forbidden_events)

        score = 1.0

        if total_rules > 0:
            score = round(
                passed_rules / total_rules,
                6,
            )

        return TraceOracleResult(
            passed=not violations,
            score=score,
            passed_rules=passed_rules,
            total_rules=total_rules,
            rule_results=rule_results,
            violations=violations,
        )


class TraceSnapshotService:
    """从持久化平台表读取 AgentRun 轨迹。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
    ) -> None:
        self._session_factory = session_factory

    async def capture(
        self,
        *,
        run_id: str,
    ) -> TraceSnapshot:
        async with self._session_factory() as session:
            async with session.begin():
                return await self.capture_from_session(
                    session=session,
                    run_id=run_id,
                )

    async def capture_from_session(
        self,
        *,
        session: AsyncSession,
        run_id: str,
    ) -> TraceSnapshot:
        step_result = await session.execute(
            select(
                RunStep.id,
                RunStep.parent_step_id,
                RunStep.step_no,
                RunStep.step_type,
                RunStep.status,
                RunStep.model_name,
                RunStep.tool_name,
                RunStep.input_payload,
                RunStep.output_payload,
                RunStep.input_tokens,
                RunStep.output_tokens,
                RunStep.latency_ms,
                RunStep.error_type,
                RunStep.error_message,
                RunStep.started_at,
                RunStep.finished_at,
            )
            .where(RunStep.run_id == run_id)
            .order_by(
                RunStep.step_no,
                RunStep.id,
            )
        )

        operation_result = await session.execute(
            select(
                ToolOperation.operation_id,
                ToolOperation.step_id,
                ToolOperation.tool_name,
                ToolOperation.arguments,
                ToolOperation.arguments_hash,
                ToolOperation.idempotency_key,
                ToolOperation.risk_level,
                ToolOperation.requires_approval,
                ToolOperation.is_idempotent,
                ToolOperation.status,
                ToolOperation.retry_count,
                ToolOperation.recovery_count,
                ToolOperation.recovery_details,
                ToolOperation.result,
                ToolOperation.latency_ms,
                ToolOperation.external_reference,
                ToolOperation.error_type,
                ToolOperation.error_message,
                ToolOperation.error_details,
                ToolOperation.started_at,
                ToolOperation.finished_at,
                ToolOperation.created_at,
            )
            .where(ToolOperation.run_id == run_id)
            .order_by(
                ToolOperation.created_at,
                ToolOperation.operation_id,
            )
        )

        return TraceSnapshot(
            run_id=run_id,
            steps=models_from_rows(
                RunStepTrace,
                step_result.mappings().all(),
            ),
            operations=models_from_rows(
                ToolOperationTrace,
                operation_result.mappings().all(),
            ),
        )


__all__ = [
    "RunStepTrace",
    "ToolOperationTrace",
    "TraceOracle",
    "TraceOracleResult",
    "TraceRuleKind",
    "TraceRuleResult",
    "TraceSnapshot",
    "TraceSnapshotService",
    "count_trace_event",
    "normalize_json_object",
    "normalize_optional_json_object",
    "operation_event_count",
]
