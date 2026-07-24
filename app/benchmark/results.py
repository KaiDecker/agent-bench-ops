from collections.abc import (
    Mapping,
    Sequence,
)
from datetime import datetime
from statistics import fmean, pstdev
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.benchmark.runner import (
    EvaluationPolicy,
)
from app.persistence.database import (
    AsyncSessionFactory,
)
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
    EvaluationResult,
)

type AgentRunStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "succeeded",
    "failed",
    "cancelled",
    "timed_out",
]


type PersistedEvaluationStatus = Literal[
    "completed",
    "missing",
]


type EvaluationStateSource = Literal[
    "live",
    "persisted",
]


TERMINAL_AGENT_RUN_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    }
)


class PersistedBenchmarkRun(BaseModel):
    """从数据库重建的一次实验运行。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    sequence_no: int | None = Field(
        default=None,
        gt=0,
    )

    run_id: str
    experiment_id: str

    task_key: str
    task_version: int

    repetition_index: int | None = Field(
        default=None,
        gt=0,
    )

    random_seed: int | None = None

    status: AgentRunStatus
    paused: bool
    runtime_passed: bool

    evaluation_status: PersistedEvaluationStatus

    evaluation_passed: bool | None = None
    passed: bool

    evaluator_version: str | None = None
    state_source: EvaluationStateSource | None = None

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

    total_steps: int = Field(
        ge=0,
    )

    total_tool_calls: int = Field(
        ge=0,
    )

    input_tokens: int = Field(
        ge=0,
    )

    output_tokens: int = Field(
        ge=0,
    )

    total_tokens: int = Field(
        ge=0,
    )

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    cost_usd: float = Field(
        ge=0,
    )

    violation_codes: tuple[
        str,
        ...,
    ] = ()

    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    evaluated_at: datetime | None


class NumericMetricSummary(BaseModel):
    """一组实验数值的描述性统计。"""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    sample_count: int = Field(
        gt=0,
    )

    mean: float
    stddev: float = Field(
        ge=0,
    )

    minimum: float
    maximum: float


class BenchmarkTaskAggregate(BaseModel):
    """同一任务和版本在一个实验中的聚合统计。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    task_key: str
    task_version: int = Field(
        gt=0,
    )

    executed_runs: int = Field(
        ge=0,
    )

    terminal_runs: int = Field(
        ge=0,
    )

    succeeded_runs: int = Field(
        ge=0,
    )

    evaluated_runs: int = Field(
        ge=0,
    )

    passed_runs: int = Field(
        ge=0,
    )

    runtime_success_rate: float = Field(
        ge=0,
        le=1,
    )

    evaluation_coverage: float = Field(
        ge=0,
        le=1,
    )

    evaluation_pass_rate: float = Field(
        ge=0,
        le=1,
    )

    end_to_end_pass_rate: float = Field(
        ge=0,
        le=1,
    )

    overall_score: NumericMetricSummary | None = None

    final_state_score: NumericMetricSummary | None = None

    trace_score: NumericMetricSummary | None = None

    temporal_score: NumericMetricSummary | None = None

    budget_score: NumericMetricSummary | None = None

    latency_ms: NumericMetricSummary | None = None

    total_tokens: NumericMetricSummary

    total_tool_calls: NumericMetricSummary


class PersistedBenchmarkExperiment(BaseModel):
    """由数据库事实重建的实验汇总。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    experiment_id: str

    runner_version: str | None = None
    evaluation_policy: EvaluationPolicy | None = None

    planned_runs: int = Field(
        ge=0,
    )

    executed_runs: int = Field(
        ge=0,
    )

    unexecuted_runs: int = Field(
        ge=0,
    )

    terminal_runs: int = Field(
        ge=0,
    )

    incomplete_runs: int = Field(
        ge=0,
    )

    succeeded_runs: int = Field(
        ge=0,
    )

    runtime_failed_runs: int = Field(
        ge=0,
    )

    evaluated_runs: int = Field(
        ge=0,
    )

    missing_evaluations: int = Field(
        ge=0,
    )

    passed_runs: int = Field(
        ge=0,
    )

    failed_evaluations: int = Field(
        ge=0,
    )

    completion_rate: float = Field(
        ge=0,
        le=1,
    )

    pass_rate: float = Field(
        ge=0,
        le=1,
    )

    evaluation_pass_rate: float = Field(
        ge=0,
        le=1,
    )

    total_tokens: int = Field(
        ge=0,
    )

    total_tool_calls: int = Field(
        ge=0,
    )

    total_cost_usd: float = Field(
        ge=0,
    )

    average_latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    first_run_at: datetime
    last_run_at: datetime

    task_aggregates: list[BenchmarkTaskAggregate] = Field(
        default_factory=list,
    )
    runs: list[PersistedBenchmarkRun] = Field(
        default_factory=list,
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
        )

    def to_summary_dict(
        self,
    ) -> dict[str, Any]:
        """
        输出不包含逐 Run 明细的实验摘要。

        保留实验级统计和按任务聚合，适合终端查看。
        """

        payload = self.to_dict()
        payload.pop(
            "runs",
            None,
        )

        return payload


def extract_runner_metadata(
    configuration: Any,
) -> dict[str, Any]:
    """读取 AgentRun.configuration 中的 Runner 元数据。"""

    if not isinstance(
        configuration,
        Mapping,
    ):
        return {}

    metadata = configuration.get("benchmark_runner")

    if not isinstance(
        metadata,
        Mapping,
    ):
        return {}

    return {str(key): value for key, value in metadata.items()}


def positive_int_or_none(
    value: Any,
) -> int | None:
    """将合法正整数提取出来。"""

    if isinstance(value, bool):
        return None

    if not isinstance(value, int):
        return None

    if value <= 0:
        return None

    return value


def optional_score(
    value: Any,
) -> float | None:
    """读取范围为 0..1 的可选分数。"""

    if isinstance(value, bool):
        return None

    if not isinstance(
        value,
        int | float,
    ):
        return None

    normalized = float(value)

    if not 0 <= normalized <= 1:
        return None

    return normalized


def nested_score(
    scores: Any,
    component: str,
) -> float | None:
    """读取 scores JSON 中某个 Oracle 的 score。"""

    if not isinstance(
        scores,
        Mapping,
    ):
        return None

    payload = scores.get(component)

    if not isinstance(
        payload,
        Mapping,
    ):
        return None

    return optional_score(payload.get("score"))


def evaluation_state_source(
    scores: Any,
) -> EvaluationStateSource | None:
    if not isinstance(
        scores,
        Mapping,
    ):
        return None

    value = scores.get("state_source")

    if value == "live":
        return "live"

    if value == "persisted":
        return "persisted"

    return None


def evaluation_policy_from_metadata(
    metadata: Mapping[str, Any],
) -> EvaluationPolicy | None:
    value = metadata.get("evaluation_policy")

    if value == "always":
        return "always"

    if value == "succeeded_only":
        return "succeeded_only"

    if value == "disabled":
        return "disabled"

    return None


def violation_codes_from_json(
    violations: Any,
) -> tuple[str, ...]:
    if not isinstance(
        violations,
        list,
    ):
        return ()

    codes: list[str] = []

    for violation in violations:
        if not isinstance(
            violation,
            Mapping,
        ):
            continue

        code = violation.get("code")

        if isinstance(code, str) and code.strip():
            codes.append(code.strip())

    return tuple(codes)


def calculate_rate(
    numerator: int,
    denominator: int,
) -> float:
    """安全计算 0..1 比率。"""

    if numerator < 0:
        raise ValueError("Rate numerator cannot be negative")

    if denominator < 0:
        raise ValueError("Rate denominator cannot be negative")

    if numerator > denominator:
        raise ValueError("Rate numerator cannot exceed denominator")

    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator,
        6,
    )


def summarize_numeric(
    values: Sequence[int | float],
) -> NumericMetricSummary | None:
    """
    计算均值和总体标准差。

    使用 population standard deviation，
    因为这些值代表当前实验已观测到的完整运行集合。
    """

    if not values:
        return None

    normalized: list[float] = []

    for value in values:
        if isinstance(value, bool):
            raise ValueError("Boolean values cannot be summarized as metrics")

        normalized.append(float(value))

    return NumericMetricSummary(
        sample_count=len(normalized),
        mean=round(
            fmean(normalized),
            6,
        ),
        stddev=round(
            pstdev(normalized),
            6,
        ),
        minimum=round(
            min(normalized),
            6,
        ),
        maximum=round(
            max(normalized),
            6,
        ),
    )


def present_scores(
    runs: Sequence[PersistedBenchmarkRun],
    field_name: Literal[
        "overall_score",
        "final_state_score",
        "trace_score",
        "temporal_score",
        "budget_score",
    ],
) -> list[float]:
    """提取实际存在的评估分数。"""

    values: list[float] = []

    for run in runs:
        value = getattr(
            run,
            field_name,
        )

        if value is not None:
            values.append(value)

    return values


def build_task_aggregates(
    runs: Sequence[PersistedBenchmarkRun],
) -> list[BenchmarkTaskAggregate]:
    """按 task_key 和 task_version 聚合运行结果。"""

    grouped: dict[
        tuple[str, int],
        list[PersistedBenchmarkRun],
    ] = {}

    for run in runs:
        identity = (
            run.task_key,
            run.task_version,
        )

        grouped.setdefault(
            identity,
            [],
        ).append(run)

    aggregates: list[BenchmarkTaskAggregate] = []

    for (
        task_key,
        task_version,
    ), task_runs in sorted(
        grouped.items(),
        key=lambda item: item[0],
    ):
        executed_runs = len(task_runs)

        terminal_runs = sum(int(run.status in TERMINAL_AGENT_RUN_STATUSES) for run in task_runs)

        succeeded_runs = sum(int(run.status == "succeeded") for run in task_runs)

        evaluated_runs = sum(int(run.evaluation_status == "completed") for run in task_runs)

        passed_runs = sum(int(run.passed) for run in task_runs)

        total_tokens = summarize_numeric([run.total_tokens for run in task_runs])

        total_tool_calls = summarize_numeric([run.total_tool_calls for run in task_runs])

        # 每个分组至少包含一个 Run，
        # 因此这两项不可能为 None。
        assert total_tokens is not None
        assert total_tool_calls is not None

        aggregates.append(
            BenchmarkTaskAggregate(
                task_key=task_key,
                task_version=task_version,
                executed_runs=executed_runs,
                terminal_runs=terminal_runs,
                succeeded_runs=(succeeded_runs),
                evaluated_runs=evaluated_runs,
                passed_runs=passed_runs,
                runtime_success_rate=(
                    calculate_rate(
                        succeeded_runs,
                        executed_runs,
                    )
                ),
                evaluation_coverage=(
                    calculate_rate(
                        evaluated_runs,
                        executed_runs,
                    )
                ),
                evaluation_pass_rate=(
                    calculate_rate(
                        passed_runs,
                        evaluated_runs,
                    )
                ),
                end_to_end_pass_rate=(
                    calculate_rate(
                        passed_runs,
                        executed_runs,
                    )
                ),
                overall_score=(
                    summarize_numeric(
                        present_scores(
                            task_runs,
                            "overall_score",
                        )
                    )
                ),
                final_state_score=(
                    summarize_numeric(
                        present_scores(
                            task_runs,
                            "final_state_score",
                        )
                    )
                ),
                trace_score=(
                    summarize_numeric(
                        present_scores(
                            task_runs,
                            "trace_score",
                        )
                    )
                ),
                temporal_score=(
                    summarize_numeric(
                        present_scores(
                            task_runs,
                            "temporal_score",
                        )
                    )
                ),
                budget_score=(
                    summarize_numeric(
                        present_scores(
                            task_runs,
                            "budget_score",
                        )
                    )
                ),
                latency_ms=(
                    summarize_numeric(
                        [run.latency_ms for run in task_runs if run.latency_ms is not None]
                    )
                ),
                total_tokens=total_tokens,
                total_tool_calls=(total_tool_calls),
            )
        )

    return aggregates


def persisted_run_from_row(
    row: Mapping[str, Any],
) -> tuple[
    PersistedBenchmarkRun,
    dict[str, Any],
]:
    """将数据库查询行转换成持久化 Run。"""

    metadata = extract_runner_metadata(row.get("configuration"))

    status = row["status"]

    if not isinstance(status, str):
        raise ValueError("AgentRun status must be a string")

    evaluation_present = row.get("evaluation_id") is not None

    raw_evaluation_passed = row.get("evaluation_passed")

    evaluation_passed = (
        raw_evaluation_passed
        if (
            evaluation_present
            and isinstance(
                raw_evaluation_passed,
                bool,
            )
        )
        else None
    )

    scores = row.get("evaluation_scores")

    input_tokens = int(row.get("input_tokens") or 0)

    output_tokens = int(row.get("output_tokens") or 0)

    configuration = row.get("configuration")

    paused = (
        status == "running"
        and isinstance(
            configuration,
            Mapping,
        )
        and configuration.get("paused") is True
    )

    run = PersistedBenchmarkRun(
        sequence_no=(positive_int_or_none(metadata.get("sequence_no"))),
        run_id=str(row["run_id"]),
        experiment_id=str(row["experiment_id"]),
        task_key=str(row["task_key"]),
        task_version=int(row["task_version"]),
        repetition_index=(positive_int_or_none(metadata.get("repetition_index"))),
        random_seed=row.get("random_seed"),
        status=status,
        paused=paused,
        runtime_passed=(status == "succeeded"),
        evaluation_status=("completed" if evaluation_present else "missing"),
        evaluation_passed=(evaluation_passed),
        passed=(evaluation_passed is True),
        evaluator_version=(row.get("evaluator_version") if evaluation_present else None),
        state_source=(evaluation_state_source(scores)),
        overall_score=(
            optional_score(scores.get("overall_score"))
            if isinstance(
                scores,
                Mapping,
            )
            else None
        ),
        final_state_score=(
            optional_score(row.get("final_state_score")) if evaluation_present else None
        ),
        trace_score=(optional_score(row.get("trace_score")) if evaluation_present else None),
        temporal_score=(
            nested_score(
                scores,
                "temporal",
            )
        ),
        budget_score=(optional_score(row.get("budget_score")) if evaluation_present else None),
        total_steps=int(row.get("total_steps") or 0),
        total_tool_calls=int(row.get("total_tool_calls") or 0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(input_tokens + output_tokens),
        latency_ms=(float(row["latency_ms"]) if row.get("latency_ms") is not None else None),
        cost_usd=float(row.get("cost_usd") or 0),
        violation_codes=(violation_codes_from_json(row.get("evaluation_violations"))),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        evaluated_at=(row.get("evaluated_at") if evaluation_present else None),
    )

    return run, metadata


def build_persisted_experiment(
    *,
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> PersistedBenchmarkExperiment:
    """根据数据库查询结果重建实验汇总。"""

    normalized_experiment_id = experiment_id.strip()

    if not normalized_experiment_id:
        raise ValueError("experiment_id cannot be empty")

    if not rows:
        raise LookupError(f"Benchmark experiment does not exist: {normalized_experiment_id}")

    converted = [persisted_run_from_row(row) for row in rows]

    runs = [item[0] for item in converted]

    metadata_items = [item[1] for item in converted]

    runs.sort(
        key=lambda run: (
            0 if run.sequence_no is not None else 1,
            run.sequence_no if run.sequence_no is not None else 2_147_483_647,
            run.created_at,
            run.run_id,
        )
    )

    planned_candidates = {
        value
        for metadata in metadata_items
        if (value := positive_int_or_none(metadata.get("planned_runs"))) is not None
    }

    if len(planned_candidates) > 1:
        raise RuntimeError("Experiment metadata contains inconsistent planned_runs values")

    if planned_candidates:
        planned_runs = next(iter(planned_candidates))
    else:
        sequence_numbers = [run.sequence_no for run in runs if run.sequence_no is not None]

        planned_runs = max(sequence_numbers) if sequence_numbers else len(runs)

    if len(runs) > planned_runs:
        raise RuntimeError("Experiment contains more runs than planned_runs metadata")

    runner_versions = {
        value
        for metadata in metadata_items
        if (
            isinstance(
                (value := metadata.get("runner_version")),
                str,
            )
            and value.strip()
        )
    }

    if len(runner_versions) > 1:
        raise RuntimeError("Experiment metadata contains inconsistent runner versions")

    policies = {
        policy
        for metadata in metadata_items
        if (policy := evaluation_policy_from_metadata(metadata)) is not None
    }

    if len(policies) > 1:
        raise RuntimeError("Experiment metadata contains inconsistent evaluation policies")

    executed_runs = len(runs)

    terminal_runs = sum(int(run.status in TERMINAL_AGENT_RUN_STATUSES) for run in runs)

    succeeded_runs = sum(int(run.status == "succeeded") for run in runs)

    runtime_failed_runs = sum(
        int(
            run.status
            in {
                "failed",
                "cancelled",
                "timed_out",
            }
        )
        for run in runs
    )

    evaluated_runs = sum(int(run.evaluation_status == "completed") for run in runs)

    passed_runs = sum(int(run.passed) for run in runs)

    failed_evaluations = sum(
        int(run.evaluation_status == "completed" and not run.passed) for run in runs
    )

    latencies = [run.latency_ms for run in runs if run.latency_ms is not None]

    average_latency_ms = None

    if latencies:
        average_latency_ms = round(
            sum(latencies) / len(latencies),
            6,
        )

    completion_rate = round(
        terminal_runs / planned_runs,
        6,
    )

    pass_rate = round(
        passed_runs / planned_runs,
        6,
    )

    evaluation_pass_rate = 0.0

    if evaluated_runs > 0:
        evaluation_pass_rate = round(
            passed_runs / evaluated_runs,
            6,
        )

    return PersistedBenchmarkExperiment(
        experiment_id=(normalized_experiment_id),
        runner_version=(next(iter(runner_versions)) if runner_versions else None),
        evaluation_policy=(next(iter(policies)) if policies else None),
        planned_runs=planned_runs,
        executed_runs=executed_runs,
        unexecuted_runs=(planned_runs - executed_runs),
        terminal_runs=terminal_runs,
        incomplete_runs=(executed_runs - terminal_runs),
        succeeded_runs=succeeded_runs,
        runtime_failed_runs=(runtime_failed_runs),
        evaluated_runs=evaluated_runs,
        missing_evaluations=(executed_runs - evaluated_runs),
        passed_runs=passed_runs,
        failed_evaluations=(failed_evaluations),
        completion_rate=(completion_rate),
        pass_rate=pass_rate,
        evaluation_pass_rate=(evaluation_pass_rate),
        total_tokens=sum(run.total_tokens for run in runs),
        total_tool_calls=sum(run.total_tool_calls for run in runs),
        total_cost_usd=round(
            sum(run.cost_usd for run in runs),
            6,
        ),
        average_latency_ms=(average_latency_ms),
        first_run_at=min(run.created_at for run in runs),
        last_run_at=max(run.created_at for run in runs),
        task_aggregates=(build_task_aggregates(runs)),
        runs=runs,
    )


class ExperimentResultService:
    """从 PostgreSQL 查询并重建实验结果。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
    ) -> None:
        self._session_factory = session_factory

    async def get_experiment(
        self,
        *,
        experiment_id: str,
    ) -> PersistedBenchmarkExperiment:
        async with self._session_factory() as session:
            return await self.get_experiment_from_session(
                session=session,
                experiment_id=(experiment_id),
            )

    async def get_experiment_from_session(
        self,
        *,
        session: AsyncSession,
        experiment_id: str,
    ) -> PersistedBenchmarkExperiment:
        normalized_experiment_id = experiment_id.strip()

        if not normalized_experiment_id:
            raise ValueError("experiment_id cannot be empty")

        result = await session.execute(
            select(
                AgentRun.id.label("run_id"),
                AgentRun.experiment_id,
                AgentRun.random_seed,
                AgentRun.status,
                AgentRun.configuration,
                AgentRun.total_steps,
                AgentRun.total_tool_calls,
                AgentRun.input_tokens,
                AgentRun.output_tokens,
                AgentRun.cost_usd,
                AgentRun.latency_ms,
                AgentRun.created_at,
                AgentRun.started_at,
                AgentRun.finished_at,
                BenchmarkTask.task_key,
                BenchmarkTask.version.label("task_version"),
                EvaluationResult.id.label("evaluation_id"),
                EvaluationResult.passed.label("evaluation_passed"),
                EvaluationResult.evaluator_version,
                EvaluationResult.final_state_score,
                EvaluationResult.trace_score,
                EvaluationResult.budget_score,
                EvaluationResult.scores.label("evaluation_scores"),
                EvaluationResult.violations.label("evaluation_violations"),
                EvaluationResult.evaluated_at,
            )
            .join(
                BenchmarkTask,
                BenchmarkTask.id == AgentRun.task_id,
            )
            .outerjoin(
                EvaluationResult,
                EvaluationResult.run_id == AgentRun.id,
            )
            .where(AgentRun.experiment_id == normalized_experiment_id)
            .order_by(
                AgentRun.created_at,
                AgentRun.id,
            )
        )

        rows = result.mappings().all()

        return build_persisted_experiment(
            experiment_id=(normalized_experiment_id),
            rows=rows,
        )


__all__ = [
    "AgentRunStatus",
    "EvaluationStateSource",
    "ExperimentResultService",
    "PersistedBenchmarkExperiment",
    "PersistedBenchmarkRun",
    "PersistedEvaluationStatus",
    "TERMINAL_AGENT_RUN_STATUSES",
    "build_persisted_experiment",
    "evaluation_policy_from_metadata",
    "evaluation_state_source",
    "extract_runner_metadata",
    "nested_score",
    "optional_score",
    "persisted_run_from_row",
    "positive_int_or_none",
    "violation_codes_from_json",
    "BenchmarkTaskAggregate",
    "NumericMetricSummary",
    "build_task_aggregates",
    "calculate_rate",
    "present_scores",
    "summarize_numeric",
]
