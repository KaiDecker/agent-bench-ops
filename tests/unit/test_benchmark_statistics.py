from datetime import (
    UTC,
    datetime,
    timedelta,
)

import pytest

from app.benchmark.results import (
    PersistedBenchmarkRun,
    build_task_aggregates,
    calculate_rate,
    summarize_numeric,
)

BASE_TIME = datetime(
    2026,
    7,
    24,
    9,
    0,
    tzinfo=UTC,
)


def persisted_run(
    *,
    run_id: str,
    task_key: str = ("employee_lookup_001"),
    task_version: int = 1,
    status: str = "succeeded",
    evaluation_passed: bool | None = True,
    overall_score: float | None = 1.0,
    latency_ms: float | None = 100.0,
    total_tokens: int = 150,
    total_tool_calls: int = 1,
    offset_seconds: int = 0,
) -> PersistedBenchmarkRun:
    evaluation_present = evaluation_passed is not None

    input_tokens = total_tokens // 2

    output_tokens = total_tokens - input_tokens

    created_at = BASE_TIME + timedelta(seconds=offset_seconds)

    return PersistedBenchmarkRun(
        sequence_no=(offset_seconds + 1),
        run_id=run_id,
        experiment_id="exp-statistics",
        task_key=task_key,
        task_version=task_version,
        repetition_index=(offset_seconds + 1),
        random_seed=100 + offset_seconds,
        status=status,
        paused=False,
        runtime_passed=(status == "succeeded"),
        evaluation_status=("completed" if evaluation_present else "missing"),
        evaluation_passed=(evaluation_passed),
        passed=(evaluation_passed is True),
        evaluator_version=("v1" if evaluation_present else None),
        state_source=("live" if evaluation_present else None),
        overall_score=(overall_score if evaluation_present else None),
        final_state_score=(overall_score if evaluation_present else None),
        trace_score=(overall_score if evaluation_present else None),
        temporal_score=(overall_score if evaluation_present else None),
        budget_score=(overall_score if evaluation_present else None),
        total_steps=2,
        total_tool_calls=(total_tool_calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        cost_usd=0.0,
        violation_codes=(),
        created_at=created_at,
        started_at=created_at,
        finished_at=(
            created_at + timedelta(milliseconds=(latency_ms or 0))
            if status
            in {
                "succeeded",
                "failed",
                "cancelled",
                "timed_out",
            }
            else None
        ),
        evaluated_at=(created_at + timedelta(seconds=1) if evaluation_present else None),
    )


def test_numeric_summary_uses_population_standard_deviation() -> None:
    summary = summarize_numeric(
        [
            100,
            300,
        ]
    )

    assert summary is not None
    assert summary.sample_count == 2
    assert summary.mean == 200.0
    assert summary.stddev == 100.0
    assert summary.minimum == 100.0
    assert summary.maximum == 300.0


def test_numeric_summary_single_sample_has_zero_deviation() -> None:
    summary = summarize_numeric([42])

    assert summary is not None
    assert summary.mean == 42.0
    assert summary.stddev == 0.0


def test_numeric_summary_returns_none_for_empty_values() -> None:
    assert summarize_numeric([]) is None


def test_task_aggregate_calculates_rates_and_metrics() -> None:
    aggregates = build_task_aggregates(
        [
            persisted_run(
                run_id="run_001",
                overall_score=1.0,
                latency_ms=100.0,
                total_tokens=100,
                offset_seconds=0,
            ),
            persisted_run(
                run_id="run_002",
                evaluation_passed=False,
                overall_score=0.5,
                latency_ms=300.0,
                total_tokens=300,
                offset_seconds=1,
            ),
        ]
    )

    aggregate = aggregates[0]

    assert aggregate.executed_runs == 2
    assert aggregate.succeeded_runs == 2
    assert aggregate.evaluated_runs == 2
    assert aggregate.passed_runs == 1

    assert aggregate.runtime_success_rate == 1.0

    assert aggregate.evaluation_coverage == 1.0

    assert aggregate.evaluation_pass_rate == 0.5

    assert aggregate.end_to_end_pass_rate == 0.5

    assert aggregate.overall_score is not None
    assert aggregate.overall_score.mean == 0.75
    assert aggregate.overall_score.stddev == 0.25

    assert aggregate.latency_ms is not None
    assert aggregate.latency_ms.mean == 200.0

    assert aggregate.total_tokens.mean == 200.0


def test_missing_evaluation_is_excluded_from_score_mean() -> None:
    aggregates = build_task_aggregates(
        [
            persisted_run(
                run_id="run_001",
                overall_score=1.0,
            ),
            persisted_run(
                run_id="run_002",
                evaluation_passed=None,
                overall_score=None,
                offset_seconds=1,
            ),
        ]
    )

    aggregate = aggregates[0]

    assert aggregate.executed_runs == 2
    assert aggregate.evaluated_runs == 1

    assert aggregate.evaluation_coverage == 0.5

    assert aggregate.overall_score is not None
    assert aggregate.overall_score.sample_count == 1
    assert aggregate.overall_score.mean == 1.0


def test_task_aggregates_are_grouped_and_sorted() -> None:
    aggregates = build_task_aggregates(
        [
            persisted_run(
                run_id="run_ticket",
                task_key=("create_ticket_001"),
            ),
            persisted_run(
                run_id="run_lookup",
                task_key=("employee_lookup_001"),
                offset_seconds=1,
            ),
        ]
    )

    assert [aggregate.task_key for aggregate in aggregates] == [
        "create_ticket_001",
        "employee_lookup_001",
    ]


def test_calculate_rate_rejects_invalid_values() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        calculate_rate(
            2,
            1,
        )

    assert calculate_rate(0, 0) == 0.0
