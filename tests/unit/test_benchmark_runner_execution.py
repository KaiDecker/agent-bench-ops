from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from app.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkRunPlan,
    BenchmarkTaskRunSpec,
)


@dataclass
class FakeRuntimeResult:
    run_id: str
    status: str
    total_steps: int = 2
    total_tool_calls: int = 1
    input_tokens: int = 10
    output_tokens: int = 5
    latency_ms: float = 100.0


@dataclass
class FakeViolation:
    code: str


@dataclass
class FakeEvaluationReport:
    passed: bool
    overall_score: float = 1.0
    final_state_score: float = 1.0
    trace_score: float = 1.0
    temporal_score: float = 1.0
    budget_score: float = 1.0
    violations: tuple[
        FakeViolation,
        ...,
    ] = ()


class FakeRuntime:
    def __init__(
        self,
        outcomes: list[FakeRuntimeResult | Exception],
        *,
        events: list[str] | None = None,
    ) -> None:
        self._outcomes = list(outcomes)
        self.events = events if events is not None else []
        self.calls: list[dict[str, Any]] = []

    async def run_benchmark_task(
        self,
        **arguments: Any,
    ) -> FakeRuntimeResult:
        self.calls.append(arguments)

        self.events.append(f"run:{arguments['task_key']}")

        outcome = self._outcomes.pop(0)

        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome

        return outcome


class FakeEvaluationService:
    def __init__(
        self,
        outcomes: list[FakeEvaluationReport | Exception],
        *,
        events: list[str] | None = None,
    ) -> None:
        self._outcomes = list(outcomes)
        self.events = events if events is not None else []
        self.calls: list[dict[str, Any]] = []

    async def evaluate_run(
        self,
        *,
        run_id: str,
        capture_live_state: bool = False,
    ) -> FakeEvaluationReport:
        self.calls.append(
            {
                "run_id": run_id,
                "capture_live_state": (capture_live_state),
            }
        )

        self.events.append(f"eval:{run_id}")

        outcome = self._outcomes.pop(0)

        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome

        return outcome


def task(
    task_key: str,
) -> BenchmarkTaskRunSpec:
    return BenchmarkTaskRunSpec(
        task_key=task_key,
        permissions=[],
    )


def plan(
    *,
    policy: str = "always",
    fail_fast: bool = False,
    two_tasks: bool = False,
) -> BenchmarkRunPlan:
    tasks = [task("employee_lookup_001")]

    if two_tasks:
        tasks.append(task("create_ticket_001"))

    return BenchmarkRunPlan.model_validate(
        {
            "experiment_id": ("exp-stage7-execution"),
            "tasks": [item.model_dump(mode="python") for item in tasks],
            "evaluation_policy": policy,
            "fail_fast": fail_fast,
        }
    )


async def test_runner_executes_and_evaluates_serially() -> None:
    events: list[str] = []

    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_001",
                status="succeeded",
            ),
            FakeRuntimeResult(
                run_id="run_002",
                status="succeeded",
            ),
        ],
        events=events,
    )

    evaluator = FakeEvaluationService(
        [
            FakeEvaluationReport(passed=True),
            FakeEvaluationReport(passed=True),
        ],
        events=events,
    )

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan(two_tasks=True))

    assert events == [
        "run:employee_lookup_001",
        "eval:run_001",
        "run:create_ticket_001",
        "eval:run_002",
    ]

    assert result.planned_runs == 2
    assert result.executed_runs == 2
    assert result.passed_runs == 2
    assert result.failed_runs == 0
    assert result.evaluated_runs == 2
    assert result.pass_rate == 1.0

    assert all(call["capture_live_state"] for call in evaluator.calls)


async def test_always_policy_evaluates_failed_runtime() -> None:
    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_failed",
                status="failed",
            )
        ]
    )

    evaluator = FakeEvaluationService(
        [
            FakeEvaluationReport(
                passed=False,
                overall_score=0.5,
                violations=(FakeViolation(code=("agent_run_not_succeeded")),),
            )
        ]
    )

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan())

    execution = result.runs[0]

    assert execution.outcome == "failed"
    assert execution.passed is False
    assert execution.evaluation_status == "completed"
    assert execution.overall_score == 0.5
    assert execution.violation_codes == ("agent_run_not_succeeded",)


async def test_succeeded_only_skips_failed_runtime() -> None:
    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_failed",
                status="failed",
            )
        ]
    )

    evaluator = FakeEvaluationService([])

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan(policy="succeeded_only"))

    execution = result.runs[0]

    assert execution.outcome == "failed"
    assert execution.evaluation_status == "skipped"
    assert evaluator.calls == []


async def test_disabled_policy_uses_runtime_result() -> None:
    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_001",
                status="succeeded",
            )
        ]
    )

    evaluator = FakeEvaluationService([])

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan(policy="disabled"))

    execution = result.runs[0]

    assert execution.passed is True
    assert execution.outcome == "passed"
    assert execution.evaluation_status == "skipped"


async def test_fail_fast_stops_after_first_failure() -> None:
    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_001",
                status="succeeded",
            ),
            FakeRuntimeResult(
                run_id="run_002",
                status="succeeded",
            ),
        ]
    )

    evaluator = FakeEvaluationService(
        [
            FakeEvaluationReport(
                passed=False,
                overall_score=0.0,
            ),
            FakeEvaluationReport(passed=True),
        ]
    )

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(
        plan(
            fail_fast=True,
            two_tasks=True,
        )
    )

    assert result.stopped_early is True
    assert result.planned_runs == 2
    assert result.executed_runs == 1
    assert len(runtime.calls) == 1
    assert result.failed_runs == 1


async def test_runtime_error_is_captured_and_execution_continues() -> None:
    runtime = FakeRuntime(
        [
            RuntimeError("runtime exploded"),
            FakeRuntimeResult(
                run_id="run_002",
                status="succeeded",
            ),
        ]
    )

    evaluator = FakeEvaluationService([FakeEvaluationReport(passed=True)])

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan(two_tasks=True))

    assert result.executed_runs == 2
    assert result.runtime_error_runs == 1
    assert result.passed_runs == 1

    first = result.runs[0]

    assert first.outcome == "runtime_error"
    assert first.error is not None
    assert first.error.error_type == "RuntimeError"


async def test_evaluation_error_is_captured_and_execution_continues() -> None:
    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_001",
                status="succeeded",
            ),
            FakeRuntimeResult(
                run_id="run_002",
                status="succeeded",
            ),
        ]
    )

    evaluator = FakeEvaluationService(
        [
            RuntimeError("evaluation exploded"),
            FakeEvaluationReport(passed=True),
        ]
    )

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan(two_tasks=True))

    assert result.evaluation_error_runs == 1
    assert result.passed_runs == 1

    first = result.runs[0]

    assert first.outcome == "evaluation_error"
    assert first.evaluation_status == "failed"


async def test_waiting_approval_runtime_is_not_evaluated() -> None:
    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_paused",
                status="waiting_approval",
            )
        ]
    )

    evaluator = FakeEvaluationService([])

    result = await BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
    ).run_plan(plan())

    execution = result.runs[0]

    assert execution.outcome == "paused"
    assert execution.passed is False
    assert execution.evaluation_status == "skipped"
    assert result.paused_runs == 1
    assert evaluator.calls == []


def fake_execution_lock_factory(
    events: list[str],
    *,
    enter_error: Exception | None = None,
):
    @asynccontextmanager
    async def lock() -> AsyncIterator[None]:
        events.append("lock:acquire")

        if enter_error is not None:
            raise enter_error

        try:
            yield

        finally:
            events.append("lock:release")

    return lock


async def test_runner_holds_lock_for_entire_plan() -> None:
    events: list[str] = []

    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_001",
                status="succeeded",
            ),
            FakeRuntimeResult(
                run_id="run_002",
                status="succeeded",
            ),
        ],
        events=events,
    )

    evaluator = FakeEvaluationService(
        [
            FakeEvaluationReport(passed=True),
            FakeEvaluationReport(passed=True),
        ],
        events=events,
    )

    runner = BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
        execution_lock_factory=(fake_execution_lock_factory(events)),
    )

    await runner.run_plan(plan(two_tasks=True))

    assert events == [
        "lock:acquire",
        "run:employee_lookup_001",
        "eval:run_001",
        "run:create_ticket_001",
        "eval:run_002",
        "lock:release",
    ]


async def test_lock_failure_prevents_runtime_execution() -> None:
    events: list[str] = []

    runtime = FakeRuntime(
        [
            FakeRuntimeResult(
                run_id="run_001",
                status="succeeded",
            )
        ],
        events=events,
    )

    evaluator = FakeEvaluationService(
        [FakeEvaluationReport(passed=True)],
        events=events,
    )

    runner = BenchmarkRunner(
        runtime=runtime,
        evaluation_service=evaluator,
        execution_lock_factory=(
            fake_execution_lock_factory(
                events,
                enter_error=RuntimeError("lock busy"),
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="lock busy",
    ):
        await runner.run_plan(plan())

    assert runtime.calls == []
    assert evaluator.calls == []

    assert events == [
        "lock:acquire",
    ]


class UnexpectedFailureRunner(BenchmarkRunner):
    async def _run_plan_locked(
        self,
        plan: BenchmarkRunPlan,
    ):
        raise RuntimeError("unexpected runner failure")


async def test_lock_releases_after_unexpected_runner_error() -> None:
    events: list[str] = []

    runner = UnexpectedFailureRunner(
        runtime=FakeRuntime([]),
        evaluation_service=(FakeEvaluationService([])),
        execution_lock_factory=(fake_execution_lock_factory(events)),
    )

    with pytest.raises(
        RuntimeError,
        match=("unexpected runner failure"),
    ):
        await runner.run_plan(plan())

    assert events == [
        "lock:acquire",
        "lock:release",
    ]
