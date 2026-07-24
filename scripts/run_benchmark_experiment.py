import argparse
import asyncio
import json
import os
import selectors
import sys
from collections import Counter
from typing import Any
from uuid import uuid4

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}


def assert_strict_msgpack_enabled() -> None:
    value = os.environ.get(
        "LANGGRAPH_STRICT_MSGPACK",
        "",
    )

    if value.strip().lower() not in _TRUE_VALUES:
        raise RuntimeError("LANGGRAPH_STRICT_MSGPACK must be set to true before starting Python.")


def positive_int(
    value: str,
) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc

    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")

    return number


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a deterministic serial AgentBenchOps benchmark experiment.")
    )

    parser.add_argument(
        "--experiment-id",
        default=None,
        help=("Unique experiment identifier. Generated automatically when omitted."),
    )

    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=2,
        help="Number of repetitions. Default: 2.",
    )

    parser.add_argument(
        "--seed-base",
        type=int,
        default=7000,
        help=("First random seed. Default: 7000."),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=("Stop after the first failed run."),
    )

    return parser.parse_args()


def resolve_experiment_id(
    provided: str | None,
) -> str:
    if provided is not None:
        normalized = provided.strip()

        if not normalized:
            raise ValueError("experiment_id cannot be empty")

        return normalized

    return f"stage7-scripted-{uuid4().hex[:12]}"


async def assert_experiment_id_unused(
    *,
    experiment_id: str,
) -> None:
    from sqlalchemy import func, select

    from app.persistence.database import (
        AsyncSessionFactory,
    )
    from app.persistence.platform_models import (
        AgentRun,
    )

    async with AsyncSessionFactory() as session:
        result = await session.execute(
            select(func.count(AgentRun.id)).where(AgentRun.experiment_id == experiment_id)
        )

        count = int(result.scalar_one())

    if count != 0:
        raise RuntimeError(f"experiment_id already exists: {experiment_id}")


async def read_database_verification(
    *,
    experiment_id: str,
) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.persistence.database import (
        AsyncSessionFactory,
    )
    from app.persistence.models import Ticket
    from app.persistence.platform_models import (
        AgentRun,
        EvaluationResult,
        ToolOperation,
    )

    async with AsyncSessionFactory() as session:
        run_count_result = await session.execute(
            select(func.count(AgentRun.id)).where(AgentRun.experiment_id == experiment_id)
        )

        evaluation_count_result = await session.execute(
            select(func.count(EvaluationResult.id))
            .join(
                AgentRun,
                AgentRun.id == EvaluationResult.run_id,
            )
            .where(AgentRun.experiment_id == experiment_id)
        )

        operation_result = await session.execute(
            select(
                ToolOperation.tool_name,
                ToolOperation.status,
                ToolOperation.run_id,
            )
            .join(
                AgentRun,
                AgentRun.id == ToolOperation.run_id,
            )
            .where(AgentRun.experiment_id == experiment_id)
            .order_by(
                AgentRun.created_at,
                ToolOperation.created_at,
                ToolOperation.operation_id,
            )
        )

        ticket_result = await session.execute(select(Ticket).order_by(Ticket.id))

        operations = list(operation_result.all())

        tickets = list(ticket_result.scalars().all())

    tool_counts = Counter(row.tool_name for row in operations)

    status_counts = Counter(row.status for row in operations)

    ticket = tickets[0] if len(tickets) == 1 else None

    return {
        "agent_run_count": int(run_count_result.scalar_one()),
        "evaluation_count": int(evaluation_count_result.scalar_one()),
        "tool_operation_count": len(operations),
        "tool_counts": dict(sorted(tool_counts.items())),
        "tool_status_counts": dict(sorted(status_counts.items())),
        "current_ticket_count": len(tickets),
        "current_ticket_title": (ticket.title if ticket is not None else None),
        "current_ticket_status": (ticket.status if ticket is not None else None),
    }


def verify_experiment(
    *,
    repetitions: int,
    immediate: Any,
    persisted: Any,
    database: dict[str, Any],
) -> None:
    expected_runs = repetitions * 2

    expected_task_order = [
        task_key
        for _ in range(repetitions)
        for task_key in (
            "employee_lookup_001",
            "create_ticket_001",
        )
    ]

    if immediate.planned_runs != expected_runs:
        raise RuntimeError("Immediate result has wrong planned run count.")

    if immediate.executed_runs != expected_runs:
        raise RuntimeError("Immediate result has wrong executed run count.")

    if immediate.passed_runs != expected_runs:
        raise RuntimeError("Not all immediate runs passed.")

    if immediate.evaluated_runs != expected_runs:
        raise RuntimeError("Not all immediate runs were evaluated.")

    if immediate.stopped_early:
        raise RuntimeError("Experiment stopped early.")

    if persisted.planned_runs != expected_runs:
        raise RuntimeError("Persisted result has wrong planned run count.")

    if persisted.executed_runs != expected_runs:
        raise RuntimeError("Persisted result has wrong executed run count.")

    if persisted.terminal_runs != expected_runs:
        raise RuntimeError("Not all persisted runs are terminal.")

    if persisted.evaluated_runs != expected_runs:
        raise RuntimeError("Persisted evaluations are missing.")

    if persisted.passed_runs != expected_runs:
        raise RuntimeError("Not all persisted runs passed.")

    if persisted.unexecuted_runs != 0:
        raise RuntimeError("Persisted experiment has unexecuted runs.")

    if persisted.incomplete_runs != 0:
        raise RuntimeError("Persisted experiment has incomplete runs.")

    if persisted.missing_evaluations != 0:
        raise RuntimeError("Persisted experiment has missing evaluations.")

    actual_task_order = [run.task_key for run in persisted.runs]

    if actual_task_order != expected_task_order:
        raise RuntimeError("Persisted task execution order is incorrect.")

    sequence_numbers = [run.sequence_no for run in persisted.runs]

    if sequence_numbers != list(
        range(
            1,
            expected_runs + 1,
        )
    ):
        raise RuntimeError("Persisted sequence numbers are incorrect.")

    if any(run.status != "succeeded" for run in persisted.runs):
        raise RuntimeError("A persisted AgentRun did not succeed.")

    if any(run.evaluation_status != "completed" for run in persisted.runs):
        raise RuntimeError("A persisted evaluation is missing.")

    if any(run.state_source != "live" for run in persisted.runs):
        raise RuntimeError("A Runner evaluation did not capture live state.")

    if any(run.overall_score != 1.0 for run in persisted.runs):
        raise RuntimeError("A persisted overall score is not 1.0.")

    if len({run.run_id for run in persisted.runs}) != expected_runs:
        raise RuntimeError("Experiment contains duplicate run IDs.")

    aggregates = {aggregate.task_key: aggregate for aggregate in persisted.task_aggregates}

    if set(aggregates) != {
        "employee_lookup_001",
        "create_ticket_001",
    }:
        raise RuntimeError("Unexpected task aggregates.")

    for aggregate in aggregates.values():
        if aggregate.executed_runs != repetitions:
            raise RuntimeError("Task aggregate has wrong execution count.")

        if aggregate.passed_runs != repetitions:
            raise RuntimeError("Task aggregate has failed runs.")

        if aggregate.end_to_end_pass_rate != 1.0:
            raise RuntimeError("Task aggregate pass rate is not 1.0.")

        if aggregate.overall_score is None:
            raise RuntimeError("Task aggregate has no overall score.")

        if aggregate.overall_score.mean != 1.0:
            raise RuntimeError("Task aggregate mean score is not 1.0.")

        if aggregate.total_tool_calls.mean != 1.0:
            raise RuntimeError("Task aggregate tool-call mean is not 1.0.")

    if database["agent_run_count"] != expected_runs:
        raise RuntimeError("Database AgentRun count is incorrect.")

    if database["evaluation_count"] != expected_runs:
        raise RuntimeError("Database EvaluationResult count is incorrect.")

    if database["tool_operation_count"] != expected_runs:
        raise RuntimeError("Database ToolOperation count is incorrect.")

    if database["tool_status_counts"] != {
        "succeeded": expected_runs,
    }:
        raise RuntimeError("A ToolOperation did not succeed.")

    if database["tool_counts"] != {
        "create_ticket": repetitions,
        "get_employee": repetitions,
    }:
        raise RuntimeError("ToolOperation distribution is incorrect.")

    if database["current_ticket_count"] != 1:
        raise RuntimeError("Final reset state should contain exactly one ticket.")

    if database["current_ticket_title"] != "数据平台访问问题":
        raise RuntimeError("Final ticket title is incorrect.")

    if database["current_ticket_status"] != "open":
        raise RuntimeError("Final ticket status is incorrect.")


async def async_main(
    arguments: argparse.Namespace,
) -> None:
    assert_strict_msgpack_enabled()

    from app.agent.checkpoint import (
        open_postgres_checkpointer,
    )
    from app.agent.model import (
        ScriptedBenchmarkModel,
    )
    from app.agent.runtime import AgentRuntime
    from app.benchmark.locking import (
        postgres_benchmark_execution_lock,
    )
    from app.benchmark.results import (
        ExperimentResultService,
    )
    from app.benchmark.runner import (
        BenchmarkRunner,
        BenchmarkRunPlan,
        BenchmarkTaskRunSpec,
    )
    from app.evaluation.evaluator import (
        EvaluationService,
    )
    from app.persistence.database import engine
    from app.tools.gateway import ToolGateway
    from app.tools.registry import (
        build_default_registry,
    )

    experiment_id = resolve_experiment_id(arguments.experiment_id)

    try:
        await assert_experiment_id_unused(experiment_id=experiment_id)

        registry = build_default_registry()
        gateway = ToolGateway(registry)

        runtime = AgentRuntime(
            model=ScriptedBenchmarkModel(),
            model_provider="scripted",
            model_name=("scripted-benchmark-suite-v1"),
            registry=registry,
            gateway=gateway,
            checkpointer_factory=(open_postgres_checkpointer),
        )

        plan = BenchmarkRunPlan(
            experiment_id=experiment_id,
            tasks=[
                BenchmarkTaskRunSpec(
                    task_key=("employee_lookup_001"),
                    task_version=1,
                    actor_id=("benchmark-agent"),
                    permissions=[
                        "employee.read",
                    ],
                    configuration={
                        "task_family": ("employee_lookup"),
                    },
                ),
                BenchmarkTaskRunSpec(
                    task_key=("create_ticket_001"),
                    task_version=1,
                    actor_id=("benchmark-agent"),
                    permissions=[
                        "ticket.write",
                    ],
                    configuration={
                        "task_family": ("ticket_creation"),
                    },
                ),
            ],
            repetitions=(arguments.repetitions),
            random_seeds=[arguments.seed_base + index for index in range(arguments.repetitions)],
            fail_fast=(arguments.fail_fast),
            evaluation_policy="always",
            prompt_version=("stage7-runner-v1"),
            agent_strategy=("langgraph-model-tool-loop"),
            memory_strategy=("messages-state"),
            configuration={
                "stage": "7D",
                "provider": "scripted",
                "checkpoint_backend": ("postgresql"),
                "execution_mode": "serial",
            },
        )

        runner = BenchmarkRunner(
            runtime=runtime,
            evaluation_service=(EvaluationService()),
            execution_lock_factory=(postgres_benchmark_execution_lock),
        )

        immediate_result = await runner.run_plan(plan)

        persisted_result = await ExperimentResultService().get_experiment(
            experiment_id=experiment_id
        )

        database_verification = await read_database_verification(experiment_id=experiment_id)

        verify_experiment(
            repetitions=(arguments.repetitions),
            immediate=immediate_result,
            persisted=persisted_result,
            database=(database_verification),
        )

        output = {
            "experiment_id": (experiment_id),
            "repetitions": (arguments.repetitions),
            "immediate_result": (immediate_result.to_dict()),
            "persisted_result": (persisted_result.to_dict()),
            "database_verification": (database_verification),
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    finally:
        await engine.dispose()


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
    arguments = parse_arguments()

    if sys.platform == "win32":
        asyncio.run(
            async_main(arguments),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(async_main(arguments))


if __name__ == "__main__":
    main()
