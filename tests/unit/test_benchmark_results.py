from datetime import (
    UTC,
    datetime,
    timedelta,
)
from typing import Any

import pytest

from app.benchmark.results import (
    ExperimentResultService,
    build_persisted_experiment,
)

BASE_TIME = datetime(
    2026,
    7,
    24,
    8,
    0,
    tzinfo=UTC,
)


def experiment_row(
    *,
    run_id: str,
    sequence_no: int | None,
    status: str = "succeeded",
    evaluation_passed: bool | None = True,
    planned_runs: int | None = 2,
    created_offset: int = 0,
) -> dict[str, Any]:
    evaluation_present = evaluation_passed is not None

    runner_metadata: dict[
        str,
        Any,
    ] = {
        "runner_version": "stage7-v1",
        "experiment_id": "exp-001",
        "repetition_index": 1,
        "evaluation_policy": "always",
    }

    if sequence_no is not None:
        runner_metadata["sequence_no"] = sequence_no

    if planned_runs is not None:
        runner_metadata["planned_runs"] = planned_runs

    created_at = BASE_TIME + timedelta(seconds=created_offset)

    return {
        "run_id": run_id,
        "experiment_id": "exp-001",
        "task_key": ("employee_lookup_001"),
        "task_version": 1,
        "random_seed": 101,
        "status": status,
        "configuration": {"benchmark_runner": (runner_metadata)},
        "total_steps": 2,
        "total_tool_calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.01,
        "latency_ms": 500.0,
        "created_at": created_at,
        "started_at": created_at,
        "finished_at": (
            created_at + timedelta(milliseconds=500)
            if status
            in {
                "succeeded",
                "failed",
                "cancelled",
                "timed_out",
            }
            else None
        ),
        "evaluation_id": (f"eval_{run_id}" if evaluation_present else None),
        "evaluation_passed": (evaluation_passed),
        "evaluator_version": ("v1" if evaluation_present else None),
        "final_state_score": (1.0 if evaluation_present else None),
        "trace_score": (1.0 if evaluation_present else None),
        "budget_score": (1.0 if evaluation_present else None),
        "evaluation_scores": (
            {
                "overall_score": (1.0 if evaluation_passed else 0.5),
                "state_source": ("live"),
                "temporal": {
                    "score": 1.0,
                },
            }
            if evaluation_present
            else None
        ),
        "evaluation_violations": (
            [] if evaluation_passed else [{"code": ("state_assertion_failed")}]
        )
        if evaluation_present
        else None,
        "evaluated_at": (created_at + timedelta(seconds=1) if evaluation_present else None),
    }


def test_builds_persisted_experiment_summary() -> None:
    result = build_persisted_experiment(
        experiment_id="exp-001",
        rows=[
            experiment_row(
                run_id="run_001",
                sequence_no=1,
            ),
            experiment_row(
                run_id="run_002",
                sequence_no=2,
                evaluation_passed=False,
                created_offset=1,
            ),
        ],
    )

    assert result.planned_runs == 2
    assert result.executed_runs == 2
    assert result.terminal_runs == 2
    assert result.succeeded_runs == 2
    assert result.evaluated_runs == 2
    assert result.passed_runs == 1
    assert result.failed_evaluations == 1
    assert result.completion_rate == 1.0
    assert result.pass_rate == 0.5
    assert result.evaluation_pass_rate == 0.5
    assert result.total_tokens == 300
    assert result.total_tool_calls == 2
    assert result.total_cost_usd == 0.02
    assert result.average_latency_ms == 500.0
    assert len(result.task_aggregates) == 1

    task_aggregate = result.task_aggregates[0]

    assert task_aggregate.task_key == ("employee_lookup_001")

    assert task_aggregate.executed_runs == 2

    assert task_aggregate.passed_runs == 1

    assert task_aggregate.end_to_end_pass_rate == 0.5


def test_runs_are_sorted_by_sequence_number() -> None:
    result = build_persisted_experiment(
        experiment_id="exp-001",
        rows=[
            experiment_row(
                run_id="run_002",
                sequence_no=2,
            ),
            experiment_row(
                run_id="run_001",
                sequence_no=1,
                created_offset=10,
            ),
        ],
    )

    assert [run.run_id for run in result.runs] == [
        "run_001",
        "run_002",
    ]


def test_tracks_incomplete_and_missing_evaluation() -> None:
    result = build_persisted_experiment(
        experiment_id="exp-001",
        rows=[
            experiment_row(
                run_id="run_001",
                sequence_no=1,
            ),
            experiment_row(
                run_id="run_002",
                sequence_no=2,
                status="running",
                evaluation_passed=None,
                created_offset=1,
            ),
        ],
    )

    assert result.terminal_runs == 1
    assert result.incomplete_runs == 1
    assert result.evaluated_runs == 1
    assert result.missing_evaluations == 1
    assert result.completion_rate == 0.5
    assert result.pass_rate == 0.5


def test_falls_back_to_executed_count_without_metadata() -> None:
    result = build_persisted_experiment(
        experiment_id="exp-001",
        rows=[
            experiment_row(
                run_id="run_001",
                sequence_no=None,
                planned_runs=None,
            )
        ],
    )

    assert result.planned_runs == 1
    assert result.executed_runs == 1


def test_rejects_inconsistent_planned_run_metadata() -> None:
    with pytest.raises(
        RuntimeError,
        match=("inconsistent planned_runs"),
    ):
        build_persisted_experiment(
            experiment_id="exp-001",
            rows=[
                experiment_row(
                    run_id="run_001",
                    sequence_no=1,
                    planned_runs=2,
                ),
                experiment_row(
                    run_id="run_002",
                    sequence_no=2,
                    planned_runs=3,
                ),
            ],
        )


class FakeMappings:
    def __init__(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._rows = rows

    def all(
        self,
    ) -> list[dict[str, Any]]:
        return self._rows


class FakeResult:
    def __init__(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._rows = rows

    def mappings(
        self,
    ) -> FakeMappings:
        return FakeMappings(self._rows)


class FakeSession:
    def __init__(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._rows = rows
        self.statements: list[Any] = []

    async def execute(
        self,
        statement: Any,
    ) -> FakeResult:
        self.statements.append(statement)

        return FakeResult(self._rows)


async def test_result_service_reads_experiment() -> None:
    session = FakeSession(
        [
            experiment_row(
                run_id="run_001",
                sequence_no=1,
                planned_runs=1,
            )
        ]
    )

    result = await ExperimentResultService().get_experiment_from_session(
        session=(
            session  # type: ignore[arg-type]
        ),
        experiment_id="exp-001",
    )

    assert len(session.statements) == 1
    assert result.experiment_id == ("exp-001")
    assert result.executed_runs == 1


async def test_result_service_rejects_missing_experiment() -> None:
    session = FakeSession([])

    with pytest.raises(
        LookupError,
        match=("experiment does not exist"),
    ):
        await ExperimentResultService().get_experiment_from_session(
            session=(
                session  # type: ignore[arg-type]
            ),
            experiment_id=("exp-missing"),
        )


def test_summary_dict_omits_runs_and_keeps_aggregates() -> None:
    result = build_persisted_experiment(
        experiment_id="exp-001",
        rows=[
            experiment_row(
                run_id="run_001",
                sequence_no=1,
                planned_runs=1,
            )
        ],
    )

    summary = result.to_summary_dict()

    assert summary["experiment_id"] == "exp-001"

    assert summary["planned_runs"] == 1

    assert summary["executed_runs"] == 1

    assert "runs" not in summary

    assert len(summary["task_aggregates"]) == 1

    assert summary["task_aggregates"][0]["task_key"] == ("employee_lookup_001")
