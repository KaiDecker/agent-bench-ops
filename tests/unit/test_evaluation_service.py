from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.evaluation.budget_oracle import (
    BudgetOracleResult,
)
from app.evaluation.evaluator import (
    EvaluationService,
    benchmark_task_spec_from_record,
    build_evaluation_report,
    resolve_evaluation_state_source,
)
from app.evaluation.rules import (
    EvaluationViolation,
)
from app.evaluation.state_oracle import (
    FinalStateOracleResult,
)
from app.evaluation.temporal_oracle import (
    TemporalOracleResult,
)
from app.evaluation.trace_oracle import (
    TraceOracleResult,
)

EVALUATED_AT = datetime(
    2026,
    7,
    23,
    12,
    0,
    tzinfo=UTC,
)


def final_state_result(
    *,
    passed: bool = True,
    score: float = 1.0,
    violations: list[EvaluationViolation] | None = None,
) -> FinalStateOracleResult:
    return FinalStateOracleResult(
        passed=passed,
        score=score,
        passed_assertions=(1 if passed else 0),
        total_assertions=1,
        rule_results=[],
        violations=violations or [],
    )


def trace_result(
    *,
    passed: bool = True,
    score: float = 1.0,
    violations: list[EvaluationViolation] | None = None,
) -> TraceOracleResult:
    return TraceOracleResult(
        passed=passed,
        score=score,
        passed_rules=(1 if passed else 0),
        total_rules=1,
        rule_results=[],
        violations=violations or [],
    )


def temporal_result(
    *,
    passed: bool = True,
    score: float = 1.0,
    violations: list[EvaluationViolation] | None = None,
) -> TemporalOracleResult:
    return TemporalOracleResult(
        passed=passed,
        score=score,
        passed_rules=(1 if passed else 0),
        total_rules=1,
        rule_results=[],
        violations=violations or [],
    )


def budget_result(
    *,
    passed: bool = True,
    score: float = 1.0,
    violations: list[EvaluationViolation] | None = None,
) -> BudgetOracleResult:
    return BudgetOracleResult(
        passed=passed,
        score=score,
        passed_metrics=(4 if passed else 0),
        total_metrics=4,
        metric_results=[],
        violations=violations or [],
    )


def test_evaluation_report_passes_all_oracles() -> None:
    report = build_evaluation_report(
        run_id="run_001",
        task_key="task_001",
        task_version=1,
        evaluator_version="v1",
        run_status="succeeded",
        state_source="live",
        actual_final_state={
            "employees": [],
        },
        final_state=(final_state_result()),
        trace=trace_result(),
        temporal=temporal_result(),
        budget=budget_result(),
        evaluated_at=EVALUATED_AT,
    )

    assert report.passed is True
    assert report.runtime_passed is True
    assert report.overall_score == 1.0
    assert report.violations == []


def test_runtime_failure_gates_evaluation() -> None:
    report = build_evaluation_report(
        run_id="run_001",
        task_key="task_001",
        task_version=1,
        evaluator_version="v1",
        run_status="failed",
        state_source="live",
        actual_final_state={},
        final_state=(final_state_result()),
        trace=trace_result(),
        temporal=temporal_result(),
        budget=budget_result(),
        evaluated_at=EVALUATED_AT,
    )

    assert report.passed is False
    assert report.runtime_passed is False

    # Oracle 本身全部满分，
    # Runtime 失败只作为通过门槛。
    assert report.overall_score == 1.0

    assert report.violations[0].code == "agent_run_not_succeeded"


def test_evaluation_report_combines_scores_and_violations() -> None:
    state_violation = EvaluationViolation(
        oracle="state",
        code="state_failed",
        message="State failed.",
    )

    budget_violation = EvaluationViolation(
        oracle="budget",
        code="budget_exceeded",
        message="Budget failed.",
    )

    report = build_evaluation_report(
        run_id="run_001",
        task_key="task_001",
        task_version=1,
        evaluator_version="v1",
        run_status="succeeded",
        state_source="persisted",
        actual_final_state={},
        final_state=(
            final_state_result(
                passed=False,
                score=0.5,
                violations=[
                    state_violation,
                ],
            )
        ),
        trace=trace_result(score=1.0),
        temporal=temporal_result(score=0.5),
        budget=budget_result(
            passed=False,
            score=0.75,
            violations=[
                budget_violation,
            ],
        ),
        evaluated_at=EVALUATED_AT,
    )

    assert report.passed is False

    assert report.overall_score == (0.6875)

    assert [violation.code for violation in report.violations] == [
        "state_failed",
        "budget_exceeded",
    ]


def test_scores_payload_contains_all_components() -> None:
    report = build_evaluation_report(
        run_id="run_001",
        task_key="task_001",
        task_version=1,
        evaluator_version="v1",
        run_status="succeeded",
        state_source="persisted",
        actual_final_state={},
        final_state=(final_state_result()),
        trace=trace_result(),
        temporal=temporal_result(),
        budget=budget_result(),
        evaluated_at=EVALUATED_AT,
    )

    scores = report.to_scores_payload()

    assert scores["state_source"] == "persisted"

    assert scores["runtime"]["passed"] is True

    assert set(scores) == {
        "overall_score",
        "state_source",
        "runtime",
        "final_state",
        "trace",
        "temporal",
        "budget",
    }


def test_benchmark_task_record_conversion() -> None:
    task = SimpleNamespace(
        task_key="employee_lookup_001",
        version=1,
        dataset_version="v1",
        name="查询员工",
        category="single_tool",
        description=None,
        user_request="查询员工。",
        initial_state={
            "employees": [],
            "accounts": [],
            "permissions": [],
            "employee_permissions": [],
            "tickets": [],
        },
        available_tools=[
            "get_employee",
        ],
        expected_state=[],
        required_events=[],
        forbidden_events=[],
        temporal_rules=[],
        budget={
            "max_agent_steps": 5,
            "max_tool_calls": 2,
            "max_tokens": 3000,
            "timeout_seconds": 30,
        },
        metadata_json={
            "difficulty": "easy",
        },
        is_active=True,
    )

    spec = benchmark_task_spec_from_record(
        task  # type: ignore[arg-type]
    )

    assert spec.task_key == ("employee_lookup_001")

    assert spec.budget.max_tokens == (3000)


class FakeScalarResult:
    def __init__(
        self,
        value: Any,
    ) -> None:
        self._value = value

    def scalar_one_or_none(
        self,
    ) -> Any:
        return self._value


class FakeSession:
    def __init__(
        self,
        values: list[Any],
    ) -> None:
        self._values = list(values)
        self.statements: list[Any] = []

    async def execute(
        self,
        statement: Any,
    ) -> FakeScalarResult:
        self.statements.append(statement)

        return FakeScalarResult(self._values.pop(0))


async def test_service_rejects_non_terminal_run() -> None:
    session = FakeSession(
        [
            SimpleNamespace(
                id="run_001",
                task_id="task_db_001",
                status="running",
            )
        ]
    )

    with pytest.raises(
        RuntimeError,
        match=("not in a terminal state"),
    ):
        await EvaluationService().evaluate_run_from_session(
            session=(
                session  # type: ignore[arg-type]
            ),
            run_id="run_001",
        )

    assert len(session.statements) == 1


def test_evaluator_version_cannot_be_empty() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        EvaluationService(evaluator_version="   ")


def test_first_evaluation_requires_explicit_live_state() -> None:
    with pytest.raises(
        RuntimeError,
        match=("requires explicit live-state capture"),
    ):
        resolve_evaluation_state_source(
            run_id="run_001",
            has_persisted_state=False,
            capture_live_state=False,
        )


def test_first_evaluation_uses_live_state_when_explicit() -> None:
    source = resolve_evaluation_state_source(
        run_id="run_001",
        has_persisted_state=False,
        capture_live_state=True,
    )

    assert source == "live"


def test_re_evaluation_uses_persisted_state_by_default() -> None:
    source = resolve_evaluation_state_source(
        run_id="run_001",
        has_persisted_state=True,
        capture_live_state=False,
    )

    assert source == "persisted"
