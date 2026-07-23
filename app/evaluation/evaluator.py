from datetime import UTC, datetime
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

from app.benchmark.schemas import (
    BenchmarkTaskSpec,
)
from app.evaluation.budget_oracle import (
    BudgetOracle,
    BudgetOracleResult,
    BudgetSnapshotService,
)
from app.evaluation.rules import (
    EvaluationViolation,
)
from app.evaluation.state_oracle import (
    BusinessStateSnapshot,
    BusinessStateSnapshotService,
    FinalStateOracle,
    FinalStateOracleResult,
)
from app.evaluation.temporal_oracle import (
    TemporalOracle,
    TemporalOracleResult,
)
from app.evaluation.trace_oracle import (
    TraceOracle,
    TraceOracleResult,
    TraceSnapshotService,
)
from app.persistence.database import (
    AsyncSessionFactory,
)
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
    EvaluationResult,
)

type EvaluationStateSource = Literal[
    "live",
    "persisted",
]


TERMINAL_RUN_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    }
)


def resolve_evaluation_state_source(
    *,
    run_id: str,
    has_persisted_state: bool,
    capture_live_state: bool,
) -> EvaluationStateSource:
    """
    决定本次评估使用实时状态还是持久化状态。

    首次评估必须显式允许采集当前业务状态，
    避免把其他 Benchmark Run 的状态错误归给历史 Run。
    """

    if has_persisted_state and not capture_live_state:
        return "persisted"

    if capture_live_state:
        return "live"

    raise RuntimeError(
        "AgentRun has no persisted final-state snapshot. "
        "First evaluation requires explicit live-state capture: "
        f"{run_id}"
    )


class EvaluationReport(BaseModel):
    """一次完整自动评估的结构化报告。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    run_id: str
    task_key: str
    task_version: int

    evaluator_version: str

    run_status: str
    runtime_passed: bool

    state_source: EvaluationStateSource

    passed: bool

    overall_score: float = Field(
        ge=0,
        le=1,
    )

    final_state_score: float = Field(
        ge=0,
        le=1,
    )

    trace_score: float = Field(
        ge=0,
        le=1,
    )

    temporal_score: float = Field(
        ge=0,
        le=1,
    )

    budget_score: float = Field(
        ge=0,
        le=1,
    )

    final_state: FinalStateOracleResult
    trace: TraceOracleResult
    temporal: TemporalOracleResult
    budget: BudgetOracleResult

    violations: list[EvaluationViolation] = Field(
        default_factory=list,
    )

    actual_final_state: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    evaluated_at: datetime

    def to_scores_payload(
        self,
    ) -> dict[str, Any]:
        """生成 evaluation_results.scores。"""

        return {
            "overall_score": self.overall_score,
            "state_source": self.state_source,
            "runtime": {
                "passed": self.runtime_passed,
                "status": self.run_status,
            },
            "final_state": (self.final_state.model_dump(mode="json")),
            "trace": self.trace.model_dump(mode="json"),
            "temporal": (self.temporal.model_dump(mode="json")),
            "budget": self.budget.model_dump(mode="json"),
        }

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
        )


def benchmark_task_spec_from_record(
    task: BenchmarkTask,
) -> BenchmarkTaskSpec:
    """将 BenchmarkTask ORM 转换成严格任务契约。"""

    return BenchmarkTaskSpec.model_validate(
        {
            "task_key": task.task_key,
            "version": task.version,
            "dataset_version": (task.dataset_version),
            "name": task.name,
            "category": task.category,
            "description": task.description,
            "user_request": task.user_request,
            "initial_state": (task.initial_state),
            "available_tools": (task.available_tools),
            "expected_state": (task.expected_state),
            "required_events": (task.required_events),
            "forbidden_events": (task.forbidden_events),
            "temporal_rules": (task.temporal_rules),
            "budget": task.budget,
            "metadata": (task.metadata_json),
            "is_active": task.is_active,
        }
    )


def build_evaluation_report(
    *,
    run_id: str,
    task_key: str,
    task_version: int,
    evaluator_version: str,
    run_status: str,
    state_source: EvaluationStateSource,
    actual_final_state: dict[str, Any],
    final_state: FinalStateOracleResult,
    trace: TraceOracleResult,
    temporal: TemporalOracleResult,
    budget: BudgetOracleResult,
    evaluated_at: datetime | None = None,
) -> EvaluationReport:
    """汇总 Runtime 与四个 Oracle 的结果。"""

    runtime_passed = run_status == "succeeded"

    violations: list[EvaluationViolation] = []

    if not runtime_passed:
        violations.append(
            EvaluationViolation(
                oracle="runtime",
                code=("agent_run_not_succeeded"),
                message=("The AgentRun did not finish with succeeded status."),
                details={
                    "run_status": run_status,
                },
            )
        )

    violations.extend(final_state.violations)
    violations.extend(trace.violations)
    violations.extend(temporal.violations)
    violations.extend(budget.violations)

    component_scores = (
        final_state.score,
        trace.score,
        temporal.score,
        budget.score,
    )

    overall_score = round(
        sum(component_scores) / len(component_scores),
        6,
    )

    passed = all(
        (
            runtime_passed,
            final_state.passed,
            trace.passed,
            temporal.passed,
            budget.passed,
        )
    )

    return EvaluationReport(
        run_id=run_id,
        task_key=task_key,
        task_version=task_version,
        evaluator_version=(evaluator_version),
        run_status=run_status,
        runtime_passed=runtime_passed,
        state_source=state_source,
        passed=passed,
        overall_score=overall_score,
        final_state_score=(final_state.score),
        trace_score=trace.score,
        temporal_score=temporal.score,
        budget_score=budget.score,
        final_state=final_state,
        trace=trace,
        temporal=temporal,
        budget=budget,
        violations=violations,
        actual_final_state=(actual_final_state),
        evaluated_at=(evaluated_at or datetime.now(UTC)),
    )


class EvaluationService:
    """AgentRun 自动评估与结果持久化服务。"""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
        evaluator_version: str = "v1",
        state_snapshot_service: (BusinessStateSnapshotService | None) = None,
        trace_snapshot_service: (TraceSnapshotService | None) = None,
        budget_snapshot_service: (BudgetSnapshotService | None) = None,
        final_state_oracle: (FinalStateOracle | None) = None,
        trace_oracle: TraceOracle | None = None,
        temporal_oracle: (TemporalOracle | None) = None,
        budget_oracle: (BudgetOracle | None) = None,
    ) -> None:
        normalized_version = evaluator_version.strip()

        if not normalized_version:
            raise ValueError("evaluator_version cannot be empty")

        self._session_factory = session_factory

        self._evaluator_version = normalized_version

        self._state_snapshot_service = state_snapshot_service or BusinessStateSnapshotService(
            session_factory
        )

        self._trace_snapshot_service = trace_snapshot_service or TraceSnapshotService(
            session_factory
        )

        self._budget_snapshot_service = budget_snapshot_service or BudgetSnapshotService(
            session_factory
        )

        self._final_state_oracle = final_state_oracle or FinalStateOracle()

        self._trace_oracle = trace_oracle or TraceOracle()

        self._temporal_oracle = temporal_oracle or TemporalOracle()

        self._budget_oracle = budget_oracle or BudgetOracle()

    async def evaluate_run(
        self,
        *,
        run_id: str,
        capture_live_state: bool = False,
    ) -> EvaluationReport:
        async with self._session_factory.begin() as session:
            return await self.evaluate_run_from_session(
                session=session,
                run_id=run_id,
                capture_live_state=capture_live_state,
            )

    async def evaluate_run_from_session(
        self,
        *,
        session: AsyncSession,
        run_id: str,
        capture_live_state: bool = False,
    ) -> EvaluationReport:
        """
        评估一个终态 AgentRun。

        同一 run_id 通过 AgentRun 行锁串行化，
        防止并发评估产生重复 EvaluationResult。
        """

        run_result = await session.execute(
            select(AgentRun).where(AgentRun.id == run_id).with_for_update()
        )

        run = run_result.scalar_one_or_none()

        if run is None:
            raise RuntimeError(f"AgentRun does not exist: {run_id}")

        if run.status not in TERMINAL_RUN_STATUSES:
            raise RuntimeError(f"AgentRun is not in a terminal state: {run_id} ({run.status})")

        task_result = await session.execute(
            select(BenchmarkTask).where(BenchmarkTask.id == run.task_id)
        )

        task = task_result.scalar_one_or_none()

        if task is None:
            raise RuntimeError(f"BenchmarkTask does not exist for AgentRun: {run_id}")

        existing_result = await session.execute(
            select(EvaluationResult).where(EvaluationResult.run_id == run_id)
        )

        existing = existing_result.scalar_one_or_none()

        state_source: EvaluationStateSource

        state_source = resolve_evaluation_state_source(
            run_id=run_id,
            has_persisted_state=(existing is not None),
            capture_live_state=(capture_live_state),
        )

        if state_source == "persisted":
            assert existing is not None

            state_snapshot = BusinessStateSnapshot.model_validate(existing.actual_final_state)

        else:
            state_snapshot = await self._state_snapshot_service.capture_from_session(session)

        trace_snapshot = await self._trace_snapshot_service.capture_from_session(
            session=session,
            run_id=run_id,
        )

        budget_snapshot = await self._budget_snapshot_service.capture_from_session(
            session=session,
            run_id=run_id,
        )

        task_spec = benchmark_task_spec_from_record(task)

        final_state_result = self._final_state_oracle.evaluate(
            snapshot=state_snapshot,
            expectations=(task_spec.expected_state),
        )

        trace_result = self._trace_oracle.evaluate(
            snapshot=trace_snapshot,
            required_events=(task_spec.required_events),
            forbidden_events=(task_spec.forbidden_events),
        )

        temporal_result = self._temporal_oracle.evaluate(
            snapshot=trace_snapshot,
            rules=(task_spec.temporal_rules),
        )

        budget_result = self._budget_oracle.evaluate(
            snapshot=budget_snapshot,
            budget=task_spec.budget,
        )

        report = build_evaluation_report(
            run_id=run.id,
            task_key=task.task_key,
            task_version=task.version,
            evaluator_version=(self._evaluator_version),
            run_status=run.status,
            state_source=state_source,
            actual_final_state=(state_snapshot.to_json_dict()),
            final_state=(final_state_result),
            trace=trace_result,
            temporal=temporal_result,
            budget=budget_result,
        )

        await self._persist_report(
            session=session,
            existing=existing,
            report=report,
        )

        return report

    @staticmethod
    async def _persist_report(
        *,
        session: AsyncSession,
        existing: EvaluationResult | None,
        report: EvaluationReport,
    ) -> None:
        scores = report.to_scores_payload()

        violations = [violation.model_dump(mode="json") for violation in report.violations]

        if existing is None:
            evaluation = EvaluationResult(
                run_id=report.run_id,
                passed=report.passed,
                evaluator_version=(report.evaluator_version),
                final_state_score=(report.final_state_score),
                trace_score=(report.trace_score),
                budget_score=(report.budget_score),
                scores=scores,
                violations=violations,
                actual_final_state=(report.actual_final_state),
                evaluated_at=(report.evaluated_at),
            )

            session.add(evaluation)

        else:
            existing.passed = report.passed

            existing.evaluator_version = report.evaluator_version

            existing.final_state_score = report.final_state_score

            existing.trace_score = report.trace_score

            existing.budget_score = report.budget_score

            existing.scores = scores
            existing.violations = violations

            existing.actual_final_state = report.actual_final_state

            existing.evaluated_at = report.evaluated_at

        await session.flush()


__all__ = [
    "EvaluationReport",
    "EvaluationService",
    "EvaluationStateSource",
    "TERMINAL_RUN_STATUSES",
    "benchmark_task_spec_from_record",
    "build_evaluation_report",
    "resolve_evaluation_state_source",
]
