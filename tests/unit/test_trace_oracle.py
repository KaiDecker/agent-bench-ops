from datetime import UTC, datetime
from typing import Any

from app.evaluation.rules import (
    TraceEventRule,
)
from app.evaluation.trace_oracle import (
    ToolOperationTrace,
    TraceOracle,
    TraceSnapshot,
    TraceSnapshotService,
    count_trace_event,
)

CREATED_AT = datetime(
    2026,
    7,
    23,
    9,
    0,
    tzinfo=UTC,
)


def operation(
    *,
    operation_id: str = "op_001",
    tool_name: str = "create_ticket",
    status: str = "succeeded",
    retry_count: int = 0,
) -> ToolOperationTrace:
    return ToolOperationTrace(
        operation_id=operation_id,
        step_id="step_002",
        tool_name=tool_name,
        arguments={
            "title": "测试工单",
        },
        arguments_hash="a" * 64,
        idempotency_key="agent-call:test",
        risk_level="medium",
        requires_approval=False,
        is_idempotent=True,
        status=status,
        retry_count=retry_count,
        recovery_count=0,
        recovery_details=None,
        result={
            "ticket": {
                "id": "ticket_001",
            }
        }
        if status == "succeeded"
        else None,
        latency_ms=12.5,
        external_reference=("ticket_001" if status == "succeeded" else None),
        error_type=None,
        error_message=None,
        error_details=None,
        started_at=CREATED_AT,
        finished_at=CREATED_AT,
        created_at=CREATED_AT,
    )


def test_trace_oracle_passes_required_and_forbidden_rules() -> None:
    result = TraceOracle().evaluate(
        snapshot=TraceSnapshot(
            run_id="run_001",
            operations=[
                operation(),
            ],
        ),
        required_events=[
            TraceEventRule(
                event="tool_succeeded",
                tool_name="create_ticket",
            )
        ],
        forbidden_events=[
            TraceEventRule(
                event="tool_called",
                tool_name="grant_permission",
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.passed_rules == 2
    assert result.violations == []


def test_trace_oracle_reports_missing_required_event() -> None:
    result = TraceOracle().evaluate(
        snapshot=TraceSnapshot(
            run_id="run_001",
        ),
        required_events=[
            TraceEventRule(
                event="tool_succeeded",
                tool_name="get_employee",
            ),
            TraceEventRule(
                event="tool_succeeded",
                tool_name="create_ticket",
            ),
        ],
        forbidden_events=[],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert len(result.violations) == 2
    assert result.violations[0].code == "required_trace_event_missing"


def test_trace_oracle_reports_forbidden_event() -> None:
    result = TraceOracle().evaluate(
        snapshot=TraceSnapshot(
            run_id="run_001",
            operations=[
                operation(tool_name=("grant_permission")),
            ],
        ),
        required_events=[],
        forbidden_events=[
            TraceEventRule(
                event="tool_called",
                tool_name="grant_permission",
            )
        ],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.violations[0].code == "forbidden_trace_event_observed"
    assert result.rule_results[0].observed_count == 1


def test_trace_replay_is_derived_from_retry_count() -> None:
    replayed_operation = operation(
        retry_count=2,
    )

    replay_rule = TraceEventRule(
        event="tool_replayed",
        tool_name="create_ticket",
    )

    called_rule = TraceEventRule(
        event="tool_called",
        tool_name="create_ticket",
    )

    replay_count, _ = count_trace_event(
        operations=[
            replayed_operation,
        ],
        rule=replay_rule,
    )

    called_count, _ = count_trace_event(
        operations=[
            replayed_operation,
        ],
        rule=called_rule,
    )

    assert replay_count == 2
    assert called_count == 3


def test_trace_oracle_distinguishes_failed_and_rejected() -> None:
    snapshot = TraceSnapshot(
        run_id="run_001",
        operations=[
            operation(
                operation_id="op_failed",
                tool_name="update_ticket",
                status="failed",
            ),
            operation(
                operation_id="op_rejected",
                tool_name="create_ticket",
                status="rejected",
            ),
        ],
    )

    result = TraceOracle().evaluate(
        snapshot=snapshot,
        required_events=[
            TraceEventRule(
                event="tool_failed",
                tool_name="update_ticket",
            ),
            TraceEventRule(
                event="tool_rejected",
                tool_name="create_ticket",
            ),
        ],
        forbidden_events=[],
    )

    assert result.passed is True
    assert result.score == 1.0


def test_trace_oracle_passes_empty_contract() -> None:
    result = TraceOracle().evaluate(
        snapshot=TraceSnapshot(run_id="run_001"),
        required_events=[],
        forbidden_events=[],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.total_rules == 0


class FakeResult:
    def __init__(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._rows = rows

    def mappings(
        self,
    ) -> "FakeResult":
        return self

    def all(
        self,
    ) -> list[dict[str, Any]]:
        return self._rows


class FakeSession:
    def __init__(
        self,
        result_sets: list[list[dict[str, Any]]],
    ) -> None:
        self._result_sets = list(result_sets)
        self.statements: list[Any] = []

    async def execute(
        self,
        statement: Any,
    ) -> FakeResult:
        self.statements.append(statement)

        return FakeResult(self._result_sets.pop(0))


async def test_trace_snapshot_service_reads_steps_and_operations() -> None:
    session = FakeSession(
        [
            [
                {
                    "id": "step_001",
                    "parent_step_id": None,
                    "step_no": 1,
                    "step_type": "model",
                    "status": "succeeded",
                    "model_name": "scripted",
                    "tool_name": None,
                    "input_payload": {
                        "z": 2,
                        "a": 1,
                    },
                    "output_payload": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 10.0,
                    "error_type": None,
                    "error_message": None,
                    "started_at": CREATED_AT,
                    "finished_at": CREATED_AT,
                }
            ],
            [operation().model_dump(mode="python")],
        ]
    )

    snapshot = await TraceSnapshotService().capture_from_session(
        session=(
            session  # type: ignore[arg-type]
        ),
        run_id="run_001",
    )

    assert len(session.statements) == 2
    assert snapshot.steps[0].id == ("step_001")
    assert list(snapshot.steps[0].input_payload.keys()) == [
        "a",
        "z",
    ]
    assert snapshot.operations[0].operation_id == "op_001"
