from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.evaluation.rules import (
    EvaluationViolation,
    TemporalEventReference,
    TemporalEventType,
    TemporalRelation,
    TemporalRule,
)
from app.evaluation.trace_oracle import (
    ToolOperationTrace,
    TraceSnapshot,
)

type TemporalPosition = tuple[
    int,
    datetime,
    str,
    int,
]


type TemporalFailureReason = Literal[
    "first_event_missing",
    "second_event_missing",
    "both_events_missing",
    "order_violation",
]


class TemporalEventOccurrence(BaseModel):
    """一个可排序的工具事件实例。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    event: TemporalEventType
    tool_name: str
    operation_id: str
    step_no: int | None
    occurred_at: datetime
    phase_rank: int = Field(
        ge=0,
    )

    def position(
        self,
    ) -> TemporalPosition:
        # 没有关联 RunStep 的历史操作排在有步骤信息之后；
        # created_at 和 operation_id 仍保证稳定顺序。
        normalized_step_no = self.step_no if self.step_no is not None else 2_147_483_647

        return (
            normalized_step_no,
            self.occurred_at,
            self.operation_id,
            self.phase_rank,
        )


class TemporalRuleResult(BaseModel):
    """一条时序规则的判定结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    rule_index: int = Field(
        ge=0,
    )

    relation: TemporalRelation

    first_event: TemporalEventReference
    second_event: TemporalEventReference

    first_operation_id: str | None = None
    second_operation_id: str | None = None

    passed: bool

    failure_reason: TemporalFailureReason | None = None


class TemporalOracleResult(BaseModel):
    """Temporal Oracle 的完整结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    passed: bool

    score: float = Field(
        ge=0,
        le=1,
    )

    passed_rules: int = Field(
        ge=0,
    )

    total_rules: int = Field(
        ge=0,
    )

    rule_results: list[TemporalRuleResult] = Field(
        default_factory=list,
    )

    violations: list[EvaluationViolation] = Field(
        default_factory=list,
    )


def operation_step_numbers(
    snapshot: TraceSnapshot,
) -> dict[str, int]:
    """构建 RunStep ID 到 step_no 的映射。"""

    return {step.id: step.step_no for step in snapshot.steps}


def operation_final_event(
    operation: ToolOperationTrace,
) -> TemporalEventType | None:
    """将最终账本状态转换成时序事件。"""

    status_event_map: dict[
        str,
        TemporalEventType,
    ] = {
        "succeeded": "tool_succeeded",
        "failed": "tool_failed",
        "rejected": "tool_rejected",
    }

    return status_event_map.get(operation.status)


def build_temporal_occurrences(
    snapshot: TraceSnapshot,
) -> list[TemporalEventOccurrence]:
    """从 TraceSnapshot 构建可排序事件实例。"""

    step_numbers = operation_step_numbers(snapshot)

    occurrences: list[TemporalEventOccurrence] = []

    for operation in snapshot.operations:
        step_no = step_numbers.get(operation.step_id) if operation.step_id is not None else None

        called_at = operation.started_at or operation.created_at

        occurrences.append(
            TemporalEventOccurrence(
                event="tool_called",
                tool_name=operation.tool_name,
                operation_id=(operation.operation_id),
                step_no=step_no,
                occurred_at=called_at,
                phase_rank=0,
            )
        )

        final_event = operation_final_event(operation)

        if final_event is None:
            continue

        finished_at = operation.finished_at or operation.started_at or operation.created_at

        occurrences.append(
            TemporalEventOccurrence(
                event=final_event,
                tool_name=operation.tool_name,
                operation_id=(operation.operation_id),
                step_no=step_no,
                occurred_at=finished_at,
                phase_rank=1,
            )
        )

    occurrences.sort(key=lambda item: item.position())

    return occurrences


def select_temporal_occurrence(
    *,
    occurrences: Sequence[TemporalEventOccurrence],
    reference: TemporalEventReference,
) -> TemporalEventOccurrence | None:
    """根据 first/last 选择一个事件实例。"""

    matches = [
        occurrence
        for occurrence in occurrences
        if (occurrence.event == reference.event and occurrence.tool_name == reference.tool_name)
    ]

    if not matches:
        return None

    if reference.occurrence == "first":
        return matches[0]

    return matches[-1]


def temporal_relation_passes(
    *,
    first_position: TemporalPosition,
    relation: TemporalRelation,
    second_position: TemporalPosition,
) -> bool:
    """判断两个事件位置是否符合顺序关系。"""

    if relation == "before":
        return first_position < second_position

    if relation == "after":
        return first_position > second_position

    raise ValueError(f"Unsupported temporal relation: {relation}")


def missing_event_reason(
    *,
    first_missing: bool,
    second_missing: bool,
) -> TemporalFailureReason:
    if first_missing and second_missing:
        return "both_events_missing"

    if first_missing:
        return "first_event_missing"

    return "second_event_missing"


class TemporalOracle:
    """根据持久化工具轨迹判断事件先后关系。"""

    def evaluate(
        self,
        *,
        snapshot: TraceSnapshot,
        rules: Sequence[TemporalRule],
    ) -> TemporalOracleResult:
        occurrences = build_temporal_occurrences(snapshot)

        violations: list[EvaluationViolation] = []

        rule_results: list[TemporalRuleResult] = []

        passed_rules = 0

        for rule_index, rule in enumerate(rules):
            first_occurrence = select_temporal_occurrence(
                occurrences=occurrences,
                reference=rule.first,
            )

            second_occurrence = select_temporal_occurrence(
                occurrences=occurrences,
                reference=rule.second,
            )

            if first_occurrence is None or second_occurrence is None:
                failure_reason = missing_event_reason(
                    first_missing=(first_occurrence is None),
                    second_missing=(second_occurrence is None),
                )

                violations.append(
                    EvaluationViolation(
                        oracle="temporal",
                        code=("temporal_event_missing"),
                        message=("A temporal rule referenced an event that was not observed."),
                        rule_index=rule_index,
                        details={
                            "failure_reason": (failure_reason),
                            "first": (rule.first.model_dump(mode="json")),
                            "second": (rule.second.model_dump(mode="json")),
                        },
                    )
                )

                rule_results.append(
                    TemporalRuleResult(
                        rule_index=rule_index,
                        relation=rule.relation,
                        first_event=rule.first,
                        second_event=rule.second,
                        first_operation_id=(
                            first_occurrence.operation_id if first_occurrence is not None else None
                        ),
                        second_operation_id=(
                            second_occurrence.operation_id
                            if second_occurrence is not None
                            else None
                        ),
                        passed=False,
                        failure_reason=(failure_reason),
                    )
                )

                continue

            passed = temporal_relation_passes(
                first_position=(first_occurrence.position()),
                relation=rule.relation,
                second_position=(second_occurrence.position()),
            )

            if passed:
                passed_rules += 1
                failure_reason = None
            else:
                failure_reason = "order_violation"

                violations.append(
                    EvaluationViolation(
                        oracle="temporal",
                        code=("temporal_order_violation"),
                        message=("Observed tool events did not satisfy the required order."),
                        rule_index=rule_index,
                        details={
                            "relation": (rule.relation),
                            "first": (rule.first.model_dump(mode="json")),
                            "second": (rule.second.model_dump(mode="json")),
                            "first_operation_id": (first_occurrence.operation_id),
                            "second_operation_id": (second_occurrence.operation_id),
                        },
                    )
                )

            rule_results.append(
                TemporalRuleResult(
                    rule_index=rule_index,
                    relation=rule.relation,
                    first_event=rule.first,
                    second_event=rule.second,
                    first_operation_id=(first_occurrence.operation_id),
                    second_operation_id=(second_occurrence.operation_id),
                    passed=passed,
                    failure_reason=(failure_reason),
                )
            )

        total_rules = len(rules)

        score = 1.0

        if total_rules > 0:
            score = round(
                passed_rules / total_rules,
                6,
            )

        return TemporalOracleResult(
            passed=not violations,
            score=score,
            passed_rules=passed_rules,
            total_rules=total_rules,
            rule_results=rule_results,
            violations=violations,
        )


__all__ = [
    "TemporalEventOccurrence",
    "TemporalFailureReason",
    "TemporalOracle",
    "TemporalOracleResult",
    "TemporalPosition",
    "TemporalRuleResult",
    "build_temporal_occurrences",
    "missing_event_reason",
    "operation_final_event",
    "operation_step_numbers",
    "select_temporal_occurrence",
    "temporal_relation_passes",
]
