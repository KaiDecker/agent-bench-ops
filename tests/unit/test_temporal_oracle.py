from datetime import (
    UTC,
    datetime,
    timedelta,
)

from app.evaluation.rules import (
    TemporalRule,
)
from app.evaluation.temporal_oracle import (
    TemporalOracle,
    build_temporal_occurrences,
)
from app.evaluation.trace_oracle import (
    RunStepTrace,
    ToolOperationTrace,
    TraceSnapshot,
)

BASE_TIME = datetime(
    2026,
    7,
    23,
    10,
    0,
    tzinfo=UTC,
)


def step(
    *,
    step_id: str,
    step_no: int,
    tool_name: str,
) -> RunStepTrace:
    return RunStepTrace(
        id=step_id,
        parent_step_id=None,
        step_no=step_no,
        step_type="tool",
        status="succeeded",
        model_name=None,
        tool_name=tool_name,
        input_payload={},
        output_payload={},
        input_tokens=0,
        output_tokens=0,
        latency_ms=10,
        error_type=None,
        error_message=None,
        started_at=(BASE_TIME + timedelta(seconds=step_no)),
        finished_at=(
            BASE_TIME
            + timedelta(
                seconds=step_no,
                milliseconds=100,
            )
        ),
    )


def operation(
    *,
    operation_id: str,
    step_id: str,
    tool_name: str,
    status: str = "succeeded",
    offset_seconds: int,
) -> ToolOperationTrace:
    started_at = BASE_TIME + timedelta(seconds=offset_seconds)

    return ToolOperationTrace(
        operation_id=operation_id,
        step_id=step_id,
        tool_name=tool_name,
        arguments={},
        arguments_hash="a" * 64,
        idempotency_key=(f"key:{operation_id}"),
        risk_level="low",
        requires_approval=False,
        is_idempotent=True,
        status=status,
        retry_count=0,
        recovery_count=0,
        recovery_details=None,
        result={} if status == "succeeded" else None,
        latency_ms=10,
        external_reference=None,
        error_type=None,
        error_message=None,
        error_details=None,
        started_at=started_at,
        finished_at=(started_at + timedelta(milliseconds=100)),
        created_at=started_at,
    )


def temporal_rule(
    *,
    first_event: str,
    first_tool: str,
    relation: str,
    second_event: str,
    second_tool: str,
    first_occurrence: str = "first",
    second_occurrence: str = "first",
) -> TemporalRule:
    return TemporalRule.model_validate(
        {
            "first": {
                "event": first_event,
                "tool_name": first_tool,
                "occurrence": (first_occurrence),
            },
            "relation": relation,
            "second": {
                "event": second_event,
                "tool_name": second_tool,
                "occurrence": (second_occurrence),
            },
        }
    )


def trace_snapshot() -> TraceSnapshot:
    return TraceSnapshot(
        run_id="run_001",
        steps=[
            step(
                step_id="step_002",
                step_no=2,
                tool_name="get_employee",
            ),
            step(
                step_id="step_004",
                step_no=4,
                tool_name="create_ticket",
            ),
        ],
        operations=[
            operation(
                operation_id="op_lookup",
                step_id="step_002",
                tool_name="get_employee",
                offset_seconds=2,
            ),
            operation(
                operation_id="op_ticket",
                step_id="step_004",
                tool_name="create_ticket",
                offset_seconds=4,
            ),
        ],
    )


def test_temporal_oracle_passes_before_rule() -> None:
    result = TemporalOracle().evaluate(
        snapshot=trace_snapshot(),
        rules=[
            temporal_rule(
                first_event=("tool_succeeded"),
                first_tool="get_employee",
                relation="before",
                second_event="tool_called",
                second_tool="create_ticket",
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.passed_rules == 1
    assert result.violations == []


def test_temporal_oracle_passes_after_rule() -> None:
    result = TemporalOracle().evaluate(
        snapshot=trace_snapshot(),
        rules=[
            temporal_rule(
                first_event="tool_called",
                first_tool="create_ticket",
                relation="after",
                second_event=("tool_succeeded"),
                second_tool="get_employee",
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0


def test_temporal_oracle_reports_order_violation() -> None:
    result = TemporalOracle().evaluate(
        snapshot=trace_snapshot(),
        rules=[
            temporal_rule(
                first_event="tool_called",
                first_tool="create_ticket",
                relation="before",
                second_event=("tool_succeeded"),
                second_tool="get_employee",
            )
        ],
    )

    assert result.passed is False
    assert result.score == 0.0

    assert result.violations[0].code == "temporal_order_violation"


def test_temporal_oracle_reports_missing_event() -> None:
    result = TemporalOracle().evaluate(
        snapshot=trace_snapshot(),
        rules=[
            temporal_rule(
                first_event="tool_called",
                first_tool="update_ticket",
                relation="before",
                second_event="tool_called",
                second_tool="create_ticket",
            )
        ],
    )

    assert result.passed is False

    assert result.violations[0].code == "temporal_event_missing"

    assert result.rule_results[0].failure_reason == "first_event_missing"


def test_called_precedes_final_event_for_same_operation() -> None:
    snapshot = TraceSnapshot(
        run_id="run_001",
        steps=[
            step(
                step_id="step_002",
                step_no=2,
                tool_name="create_ticket",
            )
        ],
        operations=[
            operation(
                operation_id="op_ticket",
                step_id="step_002",
                tool_name="create_ticket",
                offset_seconds=2,
            )
        ],
    )

    result = TemporalOracle().evaluate(
        snapshot=snapshot,
        rules=[
            temporal_rule(
                first_event="tool_called",
                first_tool="create_ticket",
                relation="before",
                second_event=("tool_succeeded"),
                second_tool="create_ticket",
            )
        ],
    )

    assert result.passed is True


def test_temporal_oracle_supports_first_and_last() -> None:
    snapshot = TraceSnapshot(
        run_id="run_001",
        steps=[
            step(
                step_id="step_002",
                step_no=2,
                tool_name="get_employee",
            ),
            step(
                step_id="step_004",
                step_no=4,
                tool_name="get_employee",
            ),
            step(
                step_id="step_006",
                step_no=6,
                tool_name="create_ticket",
            ),
        ],
        operations=[
            operation(
                operation_id="op_lookup_1",
                step_id="step_002",
                tool_name="get_employee",
                offset_seconds=2,
            ),
            operation(
                operation_id="op_lookup_2",
                step_id="step_004",
                tool_name="get_employee",
                offset_seconds=4,
            ),
            operation(
                operation_id="op_ticket",
                step_id="step_006",
                tool_name="create_ticket",
                offset_seconds=6,
            ),
        ],
    )

    result = TemporalOracle().evaluate(
        snapshot=snapshot,
        rules=[
            temporal_rule(
                first_event=("tool_succeeded"),
                first_tool="get_employee",
                first_occurrence="last",
                relation="before",
                second_event="tool_called",
                second_tool="create_ticket",
            )
        ],
    )

    assert result.passed is True

    assert result.rule_results[0].first_operation_id == "op_lookup_2"


def test_temporal_occurrences_are_deterministic() -> None:
    occurrences = build_temporal_occurrences(trace_snapshot())

    assert [
        (
            occurrence.operation_id,
            occurrence.event,
        )
        for occurrence in occurrences
    ] == [
        (
            "op_lookup",
            "tool_called",
        ),
        (
            "op_lookup",
            "tool_succeeded",
        ),
        (
            "op_ticket",
            "tool_called",
        ),
        (
            "op_ticket",
            "tool_succeeded",
        ),
    ]


def test_temporal_oracle_passes_empty_contract() -> None:
    result = TemporalOracle().evaluate(
        snapshot=TraceSnapshot(run_id="run_001"),
        rules=[],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.total_rules == 0
