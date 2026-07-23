from typing import Any

import pytest

from app.benchmark.schemas import (
    TaskBudget,
)
from app.evaluation.budget_oracle import (
    BudgetOracle,
    BudgetSnapshotService,
    RunBudgetSnapshot,
)


def default_budget() -> TaskBudget:
    return TaskBudget(
        max_agent_steps=5,
        max_tool_calls=2,
        max_tokens=3000,
        timeout_seconds=30,
    )


def test_budget_oracle_passes_within_limits() -> None:
    result = BudgetOracle().evaluate(
        snapshot=RunBudgetSnapshot(
            run_id="run_001",
            total_steps=2,
            total_tool_calls=1,
            input_tokens=500,
            output_tokens=250,
            latency_ms=1500,
        ),
        budget=default_budget(),
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.passed_metrics == 4
    assert result.total_metrics == 4
    assert result.violations == []


def test_budget_oracle_allows_exact_limits() -> None:
    result = BudgetOracle().evaluate(
        snapshot=RunBudgetSnapshot(
            run_id="run_001",
            total_steps=5,
            total_tool_calls=2,
            input_tokens=2000,
            output_tokens=1000,
            latency_ms=30_000,
        ),
        budget=default_budget(),
    )

    assert result.passed is True
    assert result.score == 1.0

    assert all(metric.passed for metric in result.metric_results)


def test_budget_oracle_reports_partial_score() -> None:
    result = BudgetOracle().evaluate(
        snapshot=RunBudgetSnapshot(
            run_id="run_001",
            total_steps=6,
            total_tool_calls=3,
            input_tokens=2000,
            output_tokens=1500,
            latency_ms=20_000,
        ),
        budget=default_budget(),
    )

    assert result.passed is False
    assert result.score == 0.25
    assert result.passed_metrics == 1

    exceeded_metrics = {violation.details["metric"] for violation in result.violations}

    assert exceeded_metrics == {
        "agent_steps",
        "tool_calls",
        "tokens",
    }


def test_budget_oracle_combines_token_counts() -> None:
    result = BudgetOracle().evaluate(
        snapshot=RunBudgetSnapshot(
            run_id="run_001",
            total_steps=1,
            total_tool_calls=0,
            input_tokens=2500,
            output_tokens=600,
            latency_ms=1000,
        ),
        budget=default_budget(),
    )

    token_result = next(metric for metric in result.metric_results if metric.metric == "tokens")

    assert token_result.actual_value == 3100
    assert token_result.passed is False


def test_budget_oracle_rejects_missing_latency() -> None:
    result = BudgetOracle().evaluate(
        snapshot=RunBudgetSnapshot(
            run_id="run_001",
            total_steps=1,
            total_tool_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=None,
        ),
        budget=default_budget(),
    )

    assert result.passed is False
    assert result.score == 0.75

    assert result.violations[0].code == "budget_metric_missing"

    assert result.violations[0].details["metric"] == "latency_ms"


class FakeMappings:
    def __init__(
        self,
        row: dict[str, Any] | None,
    ) -> None:
        self._row = row

    def one_or_none(
        self,
    ) -> dict[str, Any] | None:
        return self._row


class FakeResult:
    def __init__(
        self,
        row: dict[str, Any] | None,
    ) -> None:
        self._row = row

    def mappings(
        self,
    ) -> FakeMappings:
        return FakeMappings(self._row)


class FakeSession:
    def __init__(
        self,
        row: dict[str, Any] | None,
    ) -> None:
        self._row = row
        self.statements: list[Any] = []

    async def execute(
        self,
        statement: Any,
    ) -> FakeResult:
        self.statements.append(statement)

        return FakeResult(self._row)


async def test_budget_snapshot_service_reads_agent_run() -> None:
    session = FakeSession(
        {
            "run_id": "run_001",
            "total_steps": 2,
            "total_tool_calls": 1,
            "input_tokens": 100,
            "output_tokens": 50,
            "latency_ms": 1200.5,
        }
    )

    snapshot = await BudgetSnapshotService().capture_from_session(
        session=(
            session  # type: ignore[arg-type]
        ),
        run_id="run_001",
    )

    assert len(session.statements) == 1
    assert snapshot.run_id == "run_001"
    assert snapshot.total_tokens == 150
    assert snapshot.latency_ms == 1200.5


async def test_budget_snapshot_service_rejects_missing_run() -> None:
    session = FakeSession(None)

    with pytest.raises(
        RuntimeError,
        match="AgentRun does not exist",
    ):
        await BudgetSnapshotService().capture_from_session(
            session=(
                session  # type: ignore[arg-type]
            ),
            run_id="run_missing",
        )
