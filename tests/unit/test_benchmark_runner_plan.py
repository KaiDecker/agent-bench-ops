import pytest
from pydantic import ValidationError

from app.benchmark.runner import (
    BenchmarkRunPlan,
    BenchmarkTaskRunSpec,
    expand_benchmark_plan,
)


def task(
    task_key: str,
    *,
    permissions: list[str] | None = None,
    configuration: dict[
        str,
        object,
    ]
    | None = None,
) -> BenchmarkTaskRunSpec:
    return BenchmarkTaskRunSpec(
        task_key=task_key,
        permissions=permissions or [],
        configuration=(configuration or {}),
    )


def test_plan_expands_in_deterministic_order() -> None:
    plan = BenchmarkRunPlan(
        experiment_id="exp-stage7-001",
        tasks=[
            task(
                "employee_lookup_001",
                permissions=[
                    "employee.read",
                ],
            ),
            task(
                "create_ticket_001",
                permissions=[
                    "ticket.write",
                ],
            ),
        ],
        repetitions=2,
        random_seeds=[
            101,
            202,
        ],
    )

    runs = expand_benchmark_plan(plan)

    assert [
        (
            run.sequence_no,
            run.task_key,
            run.repetition_index,
            run.random_seed,
        )
        for run in runs
    ] == [
        (
            1,
            "employee_lookup_001",
            1,
            101,
        ),
        (
            2,
            "create_ticket_001",
            1,
            101,
        ),
        (
            3,
            "employee_lookup_001",
            2,
            202,
        ),
        (
            4,
            "create_ticket_001",
            2,
            202,
        ),
    ]


def test_planned_run_produces_runtime_arguments() -> None:
    plan = BenchmarkRunPlan(
        experiment_id="exp-stage7-002",
        tasks=[
            task(
                "employee_lookup_001",
                permissions=[
                    "employee.read",
                ],
            )
        ],
    )

    run = expand_benchmark_plan(plan)[0]

    arguments = run.to_runtime_kwargs()

    assert arguments["experiment_id"] == ("exp-stage7-002")

    assert arguments["task_key"] == ("employee_lookup_001")

    assert arguments["permissions"] == ["employee.read"]

    assert arguments["configuration"]["benchmark_runner"] == {
        "runner_version": "stage7-v1",
        "experiment_id": ("exp-stage7-002"),
        "sequence_no": 1,
        "repetition_index": 1,
        "repetitions": 1,
        "task_count": 1,
        "planned_runs": 1,
        "task_key": ("employee_lookup_001"),
        "task_version": 1,
        "execution_mode": "serial",
        "evaluation_policy": "always",
        "fail_fast": False,
    }


def test_task_configuration_overrides_plan_configuration() -> None:
    plan = BenchmarkRunPlan(
        experiment_id="exp-stage7-003",
        tasks=[
            task(
                "employee_lookup_001",
                configuration={
                    "thinking_mode": ("enabled"),
                },
            )
        ],
        configuration={
            "provider": "deepseek",
            "thinking_mode": ("disabled"),
        },
    )

    run = expand_benchmark_plan(plan)[0]

    assert run.configuration["provider"] == "deepseek"

    assert run.configuration["thinking_mode"] == "enabled"


def test_plan_rejects_seed_count_mismatch() -> None:
    with pytest.raises(
        ValidationError,
        match=("random_seeds length must equal repetitions"),
    ):
        BenchmarkRunPlan(
            experiment_id="exp-invalid",
            tasks=[task("employee_lookup_001")],
            repetitions=2,
            random_seeds=[101],
        )


def test_plan_rejects_duplicate_task_identity() -> None:
    with pytest.raises(
        ValidationError,
        match=("duplicate task identities"),
    ):
        BenchmarkRunPlan(
            experiment_id="exp-invalid",
            tasks=[
                task("employee_lookup_001"),
                task("employee_lookup_001"),
            ],
        )


def test_task_rejects_duplicate_permissions() -> None:
    with pytest.raises(
        ValidationError,
        match=("cannot contain duplicates"),
    ):
        task(
            "employee_lookup_001",
            permissions=[
                "employee.read",
                "employee.read",
            ],
        )


def test_task_rejects_empty_permission() -> None:
    with pytest.raises(
        ValidationError,
        match=("cannot contain empty"),
    ):
        task(
            "employee_lookup_001",
            permissions=[
                "employee.read",
                " ",
            ],
        )


@pytest.mark.parametrize(
    "configuration_owner",
    [
        "plan",
        "task",
    ],
)
def test_runner_configuration_key_is_reserved(
    configuration_owner: str,
) -> None:
    reserved = {
        "benchmark_runner": {
            "spoofed": True,
        }
    }

    with pytest.raises(
        ValidationError,
        match=("reserved for BenchmarkRunner"),
    ):
        if configuration_owner == "plan":
            BenchmarkRunPlan(
                experiment_id=("exp-invalid"),
                tasks=[task("employee_lookup_001")],
                configuration=reserved,
            )

        else:
            BenchmarkRunPlan(
                experiment_id=("exp-invalid"),
                tasks=[
                    task(
                        "employee_lookup_001",
                        configuration=reserved,
                    )
                ],
            )
