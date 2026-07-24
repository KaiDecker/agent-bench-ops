from collections.abc import (
    AsyncIterator,
    Callable,
    Sequence,
)
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
)
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

type EvaluationPolicy = Literal[
    "always",
    "succeeded_only",
    "disabled",
]


type BenchmarkExecutionLockFactory = Callable[
    [],
    AbstractAsyncContextManager[Any],
]


@asynccontextmanager
async def no_benchmark_execution_lock() -> AsyncIterator[None]:
    """
    默认空锁。

    数据库互斥由正式 Runner 装配时显式启用，
    使纯单元测试无需访问 PostgreSQL。
    """

    yield


RUNNER_CONFIGURATION_KEY = "benchmark_runner"


def validate_user_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    校验调用方提供的 Runtime configuration。

    benchmark_runner 是 Runner 保留字段，
    用来记录实验和重复运行信息。
    """

    if RUNNER_CONFIGURATION_KEY in configuration:
        raise ValueError(f"{RUNNER_CONFIGURATION_KEY} is reserved for BenchmarkRunner")

    return configuration


class BenchmarkTaskRunSpec(BaseModel):
    """实验计划中的一个 Benchmark Task。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    task_key: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )

    task_version: int = Field(
        default=1,
        gt=0,
    )

    actor_id: str = Field(
        default="benchmark-agent",
        min_length=1,
        max_length=100,
    )

    permissions: list[str] = Field(
        default_factory=list,
    )

    configuration: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    @field_validator(
        "task_key",
        "actor_id",
    )
    @classmethod
    def normalize_identity(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Identity fields cannot be empty")

        return normalized

    @field_validator("permissions")
    @classmethod
    def normalize_permissions(
        cls,
        permissions: list[str],
    ) -> list[str]:
        normalized = [permission.strip() for permission in permissions]

        if any(not permission for permission in normalized):
            raise ValueError("permissions cannot contain empty values")

        if len(normalized) != len(set(normalized)):
            raise ValueError("permissions cannot contain duplicates")

        return normalized

    @field_validator("configuration")
    @classmethod
    def validate_configuration(
        cls,
        configuration: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        return validate_user_configuration(configuration)


class BenchmarkRunPlan(BaseModel):
    """一组需要串行执行的 Benchmark 实验计划。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    experiment_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=(
            r"^[A-Za-z0-9]"
            r"[A-Za-z0-9_.:-]*$"
        ),
    )

    tasks: list[BenchmarkTaskRunSpec] = Field(
        min_length=1,
    )

    repetitions: int = Field(
        default=1,
        ge=1,
        le=1000,
    )

    random_seeds: list[int] | None = None

    fail_fast: bool = False

    evaluation_policy: EvaluationPolicy = "always"

    prompt_version: str = Field(
        default="v1",
        min_length=1,
        max_length=50,
    )

    agent_strategy: str = Field(
        default=("langgraph-model-tool-loop"),
        min_length=1,
        max_length=50,
    )

    memory_strategy: str = Field(
        default="messages-state",
        min_length=1,
        max_length=50,
    )

    configuration: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    @field_validator(
        "experiment_id",
        "prompt_version",
        "agent_strategy",
        "memory_strategy",
    )
    @classmethod
    def normalize_string_field(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Plan string fields cannot be empty")

        return normalized

    @field_validator("configuration")
    @classmethod
    def validate_configuration(
        cls,
        configuration: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        return validate_user_configuration(configuration)

    @model_validator(mode="after")
    def validate_plan(
        self,
    ) -> Self:
        task_identities = [
            (
                task.task_key,
                task.task_version,
            )
            for task in self.tasks
        ]

        if len(task_identities) != len(set(task_identities)):
            raise ValueError("BenchmarkRunPlan cannot contain duplicate task identities")

        if self.random_seeds is not None and len(self.random_seeds) != self.repetitions:
            raise ValueError("random_seeds length must equal repetitions")

        return self


class PlannedBenchmarkRun(BaseModel):
    """展开后的一次具体 AgentRun 请求。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    sequence_no: int = Field(
        gt=0,
    )

    experiment_id: str
    task_key: str
    task_version: int

    repetition_index: int = Field(
        gt=0,
    )

    actor_id: str

    permissions: tuple[
        str,
        ...,
    ] = ()

    random_seed: int | None = None

    prompt_version: str
    agent_strategy: str
    memory_strategy: str

    evaluation_policy: EvaluationPolicy

    configuration: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    def to_runtime_kwargs(
        self,
    ) -> dict[str, Any]:
        """转换成 AgentRuntime.run_benchmark_task 参数。"""

        return {
            "task_key": self.task_key,
            "task_version": (self.task_version),
            "actor_id": self.actor_id,
            "permissions": list(self.permissions),
            "prompt_version": (self.prompt_version),
            "agent_strategy": (self.agent_strategy),
            "memory_strategy": (self.memory_strategy),
            "experiment_id": (self.experiment_id),
            "random_seed": (self.random_seed),
            "configuration": dict(self.configuration),
        }


def expand_benchmark_plan(
    plan: BenchmarkRunPlan,
) -> tuple[
    PlannedBenchmarkRun,
    ...,
]:
    """
    将实验计划确定性展开成单次运行。

    顺序固定为：
    repetition 1 的全部任务，
    repetition 2 的全部任务……
    """

    planned_runs: list[PlannedBenchmarkRun] = []

    sequence_no = 0

    planned_run_count = len(plan.tasks) * plan.repetitions

    for repetition_index in range(
        1,
        plan.repetitions + 1,
    ):
        random_seed = (
            plan.random_seeds[repetition_index - 1] if plan.random_seeds is not None else None
        )

        for task in plan.tasks:
            sequence_no += 1

            configuration = {
                **plan.configuration,
                **task.configuration,
                RUNNER_CONFIGURATION_KEY: {
                    "runner_version": "stage7-v1",
                    "experiment_id": (plan.experiment_id),
                    "sequence_no": (sequence_no),
                    "repetition_index": (repetition_index),
                    "repetitions": (plan.repetitions),
                    "task_count": len(plan.tasks),
                    "planned_runs": (planned_run_count),
                    "task_key": (task.task_key),
                    "task_version": (task.task_version),
                    "execution_mode": "serial",
                    "evaluation_policy": (plan.evaluation_policy),
                    "fail_fast": (plan.fail_fast),
                },
            }

            planned_runs.append(
                PlannedBenchmarkRun(
                    sequence_no=sequence_no,
                    experiment_id=(plan.experiment_id),
                    task_key=task.task_key,
                    task_version=(task.task_version),
                    repetition_index=(repetition_index),
                    actor_id=task.actor_id,
                    permissions=tuple(task.permissions),
                    random_seed=random_seed,
                    prompt_version=(plan.prompt_version),
                    agent_strategy=(plan.agent_strategy),
                    memory_strategy=(plan.memory_strategy),
                    evaluation_policy=(plan.evaluation_policy),
                    configuration=(configuration),
                )
            )

    return tuple(planned_runs)


type EvaluationExecutionStatus = Literal[
    "completed",
    "skipped",
    "failed",
]


type BenchmarkRunOutcome = Literal[
    "passed",
    "failed",
    "paused",
    "runtime_error",
    "evaluation_error",
]


class RuntimeResultLike(Protocol):
    """BenchmarkRunner 依赖的 Runtime 返回契约。"""

    run_id: str
    status: str
    total_steps: int
    total_tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float | None


class EvaluationViolationLike(Protocol):
    """Runner 需要读取的最小违规项契约。"""

    code: str


class EvaluationReportLike(Protocol):
    """BenchmarkRunner 依赖的评估报告契约。"""

    passed: bool
    overall_score: float
    final_state_score: float
    trace_score: float
    temporal_score: float
    budget_score: float
    violations: Sequence[EvaluationViolationLike]


class BenchmarkRuntime(Protocol):
    """可供 BenchmarkRunner 调用的 Runtime。"""

    async def run_benchmark_task(
        self,
        *,
        task_key: str,
        task_version: int,
        actor_id: str,
        permissions: Sequence[str],
        prompt_version: str,
        agent_strategy: str,
        memory_strategy: str,
        experiment_id: str | None,
        random_seed: int | None,
        configuration: dict[str, Any],
    ) -> RuntimeResultLike:
        """运行一个 Benchmark Task。"""
        ...


class BenchmarkEvaluator(Protocol):
    """可供 BenchmarkRunner 调用的评估服务。"""

    async def evaluate_run(
        self,
        *,
        run_id: str,
        capture_live_state: bool = False,
    ) -> EvaluationReportLike:
        """评估一个终态 AgentRun。"""
        ...


class BenchmarkExecutionError(BaseModel):
    """一次 Runner 执行错误。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    stage: Literal[
        "runtime",
        "evaluation",
    ]

    error_type: str = Field(
        min_length=1,
        max_length=200,
    )

    message: str


class BenchmarkRunExecution(BaseModel):
    """实验中一次具体运行的执行结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    sequence_no: int = Field(
        gt=0,
    )

    experiment_id: str
    task_key: str
    task_version: int

    repetition_index: int = Field(
        gt=0,
    )

    random_seed: int | None = None

    run_id: str | None = None
    runtime_status: str | None = None

    outcome: BenchmarkRunOutcome
    passed: bool

    evaluation_status: EvaluationExecutionStatus

    evaluation_passed: bool | None = None

    overall_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    final_state_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    trace_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    temporal_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    budget_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    total_steps: int | None = Field(
        default=None,
        ge=0,
    )

    total_tool_calls: int | None = Field(
        default=None,
        ge=0,
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    violation_codes: tuple[
        str,
        ...,
    ] = ()

    error: BenchmarkExecutionError | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
        )


class BenchmarkExperimentResult(BaseModel):
    """一个完整串行实验的结果汇总。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    experiment_id: str

    planned_runs: int = Field(
        ge=0,
    )

    executed_runs: int = Field(
        ge=0,
    )

    passed_runs: int = Field(
        ge=0,
    )

    failed_runs: int = Field(
        ge=0,
    )

    evaluated_runs: int = Field(
        ge=0,
    )

    runtime_error_runs: int = Field(
        ge=0,
    )

    evaluation_error_runs: int = Field(
        ge=0,
    )

    paused_runs: int = Field(
        ge=0,
    )

    pass_rate: float = Field(
        ge=0,
        le=1,
    )

    stopped_early: bool

    started_at: datetime
    finished_at: datetime

    duration_ms: float = Field(
        ge=0,
    )

    runs: list[BenchmarkRunExecution] = Field(
        default_factory=list,
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
        )


def should_evaluate_runtime_result(
    *,
    policy: EvaluationPolicy,
    runtime_status: str,
) -> bool:
    """判断一次 Runtime 结果是否应当立即评估。"""

    if policy == "disabled":
        return False

    if policy == "succeeded_only":
        return runtime_status == "succeeded"

    return runtime_status in {
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    }


def build_experiment_result(
    *,
    plan: BenchmarkRunPlan,
    planned_run_count: int,
    runs: list[BenchmarkRunExecution],
    stopped_early: bool,
    started_at: datetime,
    finished_at: datetime,
    duration_ms: float,
) -> BenchmarkExperimentResult:
    """从单项执行结果生成实验汇总。"""

    executed_runs = len(runs)

    passed_runs = sum(int(run.passed) for run in runs)

    failed_runs = executed_runs - passed_runs

    evaluated_runs = sum(int(run.evaluation_status == "completed") for run in runs)

    runtime_error_runs = sum(int(run.outcome == "runtime_error") for run in runs)

    evaluation_error_runs = sum(int(run.outcome == "evaluation_error") for run in runs)

    paused_runs = sum(int(run.outcome == "paused") for run in runs)

    pass_rate = 0.0

    if executed_runs > 0:
        pass_rate = round(
            passed_runs / executed_runs,
            6,
        )

    return BenchmarkExperimentResult(
        experiment_id=(plan.experiment_id),
        planned_runs=planned_run_count,
        executed_runs=executed_runs,
        passed_runs=passed_runs,
        failed_runs=failed_runs,
        evaluated_runs=evaluated_runs,
        runtime_error_runs=(runtime_error_runs),
        evaluation_error_runs=(evaluation_error_runs),
        paused_runs=paused_runs,
        pass_rate=pass_rate,
        stopped_early=stopped_early,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        runs=runs,
    )


class BenchmarkRunner:
    """
    串行执行 BenchmarkRunPlan。

    每个 Runtime 完成后立即评估，确保业务状态
    在下一次 reset 前保存到 EvaluationResult。
    """

    def __init__(
        self,
        *,
        runtime: BenchmarkRuntime,
        evaluation_service: (BenchmarkEvaluator),
        execution_lock_factory: (BenchmarkExecutionLockFactory) = no_benchmark_execution_lock,
    ) -> None:
        self._runtime = runtime
        self._evaluation_service = evaluation_service
        self._execution_lock_factory = execution_lock_factory

    async def run_plan(
        self,
        plan: BenchmarkRunPlan,
    ) -> BenchmarkExperimentResult:
        async with self._execution_lock_factory():
            return await self._run_plan_locked(plan)

    async def _run_plan_locked(
        self,
        plan: BenchmarkRunPlan,
    ) -> BenchmarkExperimentResult:
        planned_runs = expand_benchmark_plan(plan)

        started_at = datetime.now(UTC)
        started_counter = perf_counter()

        executions: list[BenchmarkRunExecution] = []

        stopped_early = False

        for planned_run in planned_runs:
            execution = await self._execute_run(planned_run)

            executions.append(execution)

            if plan.fail_fast and not execution.passed:
                stopped_early = True
                break

        finished_at = datetime.now(UTC)

        duration_ms = round(
            (perf_counter() - started_counter) * 1000,
            2,
        )

        return build_experiment_result(
            plan=plan,
            planned_run_count=len(planned_runs),
            runs=executions,
            stopped_early=stopped_early,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    async def _execute_run(
        self,
        planned_run: PlannedBenchmarkRun,
    ) -> BenchmarkRunExecution:
        try:
            runtime_result = await self._runtime.run_benchmark_task(
                **planned_run.to_runtime_kwargs()
            )

        except Exception as exc:
            return BenchmarkRunExecution(
                sequence_no=(planned_run.sequence_no),
                experiment_id=(planned_run.experiment_id),
                task_key=(planned_run.task_key),
                task_version=(planned_run.task_version),
                repetition_index=(planned_run.repetition_index),
                random_seed=(planned_run.random_seed),
                outcome="runtime_error",
                passed=False,
                evaluation_status=("skipped"),
                error=(
                    BenchmarkExecutionError(
                        stage="runtime",
                        error_type=(type(exc).__name__),
                        message=str(exc),
                    )
                ),
            )

        runtime_status = runtime_result.status

        common_values: dict[str, Any] = {
            "sequence_no": (planned_run.sequence_no),
            "experiment_id": (planned_run.experiment_id),
            "task_key": (planned_run.task_key),
            "task_version": (planned_run.task_version),
            "repetition_index": (planned_run.repetition_index),
            "random_seed": (planned_run.random_seed),
            "run_id": runtime_result.run_id,
            "runtime_status": (runtime_status),
            "total_steps": (runtime_result.total_steps),
            "total_tool_calls": (runtime_result.total_tool_calls),
            "input_tokens": (runtime_result.input_tokens),
            "output_tokens": (runtime_result.output_tokens),
            "latency_ms": (runtime_result.latency_ms),
        }

        if runtime_status in {"paused", "waiting_approval"}:
            return BenchmarkRunExecution(
                **common_values,
                outcome="paused",
                passed=False,
                evaluation_status="skipped",
            )

        should_evaluate = should_evaluate_runtime_result(
            policy=(planned_run.evaluation_policy),
            runtime_status=(runtime_status),
        )

        if not should_evaluate:
            runtime_passed = runtime_status == "succeeded"

            return BenchmarkRunExecution(
                **common_values,
                outcome=("passed" if runtime_passed else "failed"),
                passed=runtime_passed,
                evaluation_status="skipped",
            )

        try:
            evaluation_report = await self._evaluation_service.evaluate_run(
                run_id=(runtime_result.run_id),
                capture_live_state=True,
            )

        except Exception as exc:
            return BenchmarkRunExecution(
                **common_values,
                outcome=("evaluation_error"),
                passed=False,
                evaluation_status="failed",
                error=(
                    BenchmarkExecutionError(
                        stage="evaluation",
                        error_type=(type(exc).__name__),
                        message=str(exc),
                    )
                ),
            )

        passed = runtime_status == "succeeded" and evaluation_report.passed

        return BenchmarkRunExecution(
            **common_values,
            outcome=("passed" if passed else "failed"),
            passed=passed,
            evaluation_status="completed",
            evaluation_passed=(evaluation_report.passed),
            overall_score=(evaluation_report.overall_score),
            final_state_score=(evaluation_report.final_state_score),
            trace_score=(evaluation_report.trace_score),
            temporal_score=(evaluation_report.temporal_score),
            budget_score=(evaluation_report.budget_score),
            violation_codes=tuple(violation.code for violation in evaluation_report.violations),
        )


__all__ = [
    "BenchmarkExecutionError",
    "BenchmarkExperimentResult",
    "BenchmarkRunExecution",
    "BenchmarkRunOutcome",
    "BenchmarkRunPlan",
    "BenchmarkRunner",
    "BenchmarkRuntime",
    "BenchmarkTaskRunSpec",
    "EvaluationExecutionStatus",
    "EvaluationPolicy",
    "EvaluationReportLike",
    "PlannedBenchmarkRun",
    "RUNNER_CONFIGURATION_KEY",
    "RuntimeResultLike",
    "build_experiment_result",
    "expand_benchmark_plan",
    "should_evaluate_runtime_result",
    "validate_user_configuration",
    "BenchmarkExecutionLockFactory",
    "no_benchmark_execution_lock",
]
