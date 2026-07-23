from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

type StateEntity = Literal[
    "employees",
    "accounts",
    "permissions",
    "employee_permissions",
    "tickets",
    "ticket_mutations",
]


type StateOperator = Literal[
    "eq",
    "ne",
    "exists",
    "not_exists",
    "in",
    "not_in",
]


type TraceEventType = Literal[
    "tool_called",
    "tool_succeeded",
    "tool_failed",
    "tool_rejected",
    "tool_replayed",
]


type EvaluationOracle = Literal[
    "state",
    "trace",
    "temporal",
    "budget",
    "runtime",
]


type TemporalEventType = Literal[
    "tool_called",
    "tool_succeeded",
    "tool_failed",
    "tool_rejected",
]


type TemporalRelation = Literal[
    "before",
    "after",
]


type TemporalOccurrence = Literal[
    "first",
    "last",
]


class StateAssertion(BaseModel):
    """对一个业务实体字段执行的断言。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    field: str = Field(
        min_length=1,
        max_length=100,
        pattern=(
            r"^[A-Za-z_]"
            r"[A-Za-z0-9_.]*$"
        ),
    )

    operator: StateOperator

    # 必须允许显式断言 JSON null，
    # 因此不能简单通过 value is None
    # 判断该字段是否被提供。
    value: Any = None

    @model_validator(mode="after")
    def validate_operator_value(
        self,
    ) -> Self:
        value_was_provided = "value" in self.model_fields_set

        if self.operator in {
            "exists",
            "not_exists",
        }:
            if value_was_provided:
                raise ValueError(f"{self.operator} assertions must not define value")

            return self

        if not value_was_provided:
            raise ValueError(f"{self.operator} assertions must define value")

        if self.operator in {
            "in",
            "not_in",
        }:
            if not isinstance(
                self.value,
                list,
            ):
                raise ValueError(f"{self.operator} assertions require value to be a list")

            if not self.value:
                raise ValueError(f"{self.operator} assertions require a non-empty list")

        return self


class StateExpectation(BaseModel):
    """在最终业务状态中查找实体并执行字段断言。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    entity: StateEntity

    where: dict[str, Any] = Field(
        default_factory=dict,
    )

    assertions: list[StateAssertion] = Field(
        min_length=1,
    )

    @field_validator("where")
    @classmethod
    def validate_where(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}

        for raw_key, expected_value in value.items():
            key = raw_key.strip()

            if not key:
                raise ValueError("where cannot contain an empty field name")

            if key in normalized:
                raise ValueError(f"where contains duplicate normalized field: {key}")

            normalized[key] = expected_value

        return normalized


class TraceEventRule(BaseModel):
    """要求或禁止出现的结构化工具事件。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    event: TraceEventType

    tool_name: str = Field(
        min_length=1,
        max_length=100,
        pattern=(
            r"^[A-Za-z_]"
            r"[A-Za-z0-9_]*$"
        ),
    )


class TemporalEventReference(BaseModel):
    """时序规则中的一个工具事件引用。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    event: TemporalEventType

    tool_name: str = Field(
        min_length=1,
        max_length=100,
        pattern=(
            r"^[A-Za-z_]"
            r"[A-Za-z0-9_]*$"
        ),
    )

    occurrence: TemporalOccurrence = "first"


class TemporalRule(BaseModel):
    """两个结构化工具事件之间的顺序约束。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    first: TemporalEventReference

    relation: TemporalRelation

    second: TemporalEventReference

    @model_validator(mode="after")
    def reject_identical_event_reference(
        self,
    ) -> Self:
        if self.first == self.second:
            raise ValueError("Temporal rule endpoints must not be identical")

        return self


class EvaluationViolation(BaseModel):
    """一个结构化评估违规项。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    oracle: EvaluationOracle
    code: str = Field(
        min_length=1,
        max_length=100,
    )
    message: str = Field(
        min_length=1,
    )
    rule_index: int | None = Field(
        default=None,
        ge=0,
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
    )


__all__ = [
    "EvaluationOracle",
    "EvaluationViolation",
    "StateAssertion",
    "StateEntity",
    "StateExpectation",
    "StateOperator",
    "TemporalEventReference",
    "TemporalEventType",
    "TemporalOccurrence",
    "TemporalRelation",
    "TemporalRule",
    "TraceEventRule",
    "TraceEventType",
]
