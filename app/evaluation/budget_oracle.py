from typing import Literal

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

from app.benchmark.schemas import TaskBudget
from app.evaluation.rules import (
    EvaluationViolation,
)
from app.persistence.database import (
    AsyncSessionFactory,
)
from app.persistence.platform_models import (
    AgentRun,
)

type BudgetMetric = Literal[
    "agent_steps",
    "tool_calls",
    "tokens",
    "latency_ms",
]


class RunBudgetSnapshot(BaseModel):
    """一次 AgentRun 的预算事实快照。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    run_id: str

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

    latency_ms: float | None = Field(
        default=None,
        ge=0,
    )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BudgetMetricResult(BaseModel):
    """一项预算指标的判定结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    metric: BudgetMetric

    actual_value: int | float | None

    limit_value: int | float

    utilization: float | None

    passed: bool


class BudgetOracleResult(BaseModel):
    """Budget Oracle 的完整结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    passed: bool

    score: float = Field(
        ge=0,
        le=1,
    )

    passed_metrics: int = Field(
        ge=0,
    )

    total_metrics: int = Field(
        ge=0,
    )

    metric_results: list[BudgetMetricResult] = Field(
        default_factory=list,
    )

    violations: list[EvaluationViolation] = Field(
        default_factory=list,
    )


def calculate_utilization(
    *,
    actual_value: int | float,
    limit_value: int | float,
) -> float:
    """计算预算使用比例。"""

    if limit_value <= 0:
        raise ValueError("Budget limit must be positive")

    return round(
        float(actual_value) / float(limit_value),
        6,
    )


class BudgetOracle:
    """根据 AgentRun 统计判断任务预算。"""

    def evaluate(
        self,
        *,
        snapshot: RunBudgetSnapshot,
        budget: TaskBudget,
    ) -> BudgetOracleResult:
        metric_results: list[BudgetMetricResult] = []

        violations: list[EvaluationViolation] = []

        metrics: list[
            tuple[
                BudgetMetric,
                int | float | None,
                int | float,
            ]
        ] = [
            (
                "agent_steps",
                snapshot.total_steps,
                budget.max_agent_steps,
            ),
            (
                "tool_calls",
                snapshot.total_tool_calls,
                budget.max_tool_calls,
            ),
            (
                "tokens",
                snapshot.total_tokens,
                budget.max_tokens,
            ),
            (
                "latency_ms",
                snapshot.latency_ms,
                budget.timeout_seconds * 1000,
            ),
        ]

        passed_metrics = 0

        for (
            metric,
            actual_value,
            limit_value,
        ) in metrics:
            if actual_value is None:
                metric_results.append(
                    BudgetMetricResult(
                        metric=metric,
                        actual_value=None,
                        limit_value=limit_value,
                        utilization=None,
                        passed=False,
                    )
                )

                violations.append(
                    EvaluationViolation(
                        oracle="budget",
                        code=("budget_metric_missing"),
                        message=("A required budget metric was not recorded."),
                        details={
                            "metric": metric,
                            "limit_value": (limit_value),
                        },
                    )
                )

                continue

            utilization = calculate_utilization(
                actual_value=actual_value,
                limit_value=limit_value,
            )

            passed = actual_value <= limit_value

            if passed:
                passed_metrics += 1
            else:
                violations.append(
                    EvaluationViolation(
                        oracle="budget",
                        code="budget_exceeded",
                        message=("An AgentRun exceeded its configured budget."),
                        details={
                            "metric": metric,
                            "actual_value": (actual_value),
                            "limit_value": (limit_value),
                            "overage": round(
                                float(actual_value) - float(limit_value),
                                6,
                            ),
                            "utilization": (utilization),
                        },
                    )
                )

            metric_results.append(
                BudgetMetricResult(
                    metric=metric,
                    actual_value=actual_value,
                    limit_value=limit_value,
                    utilization=utilization,
                    passed=passed,
                )
            )

        total_metrics = len(metrics)

        score = round(
            passed_metrics / total_metrics,
            6,
        )

        return BudgetOracleResult(
            passed=not violations,
            score=score,
            passed_metrics=passed_metrics,
            total_metrics=total_metrics,
            metric_results=metric_results,
            violations=violations,
        )


class BudgetSnapshotService:
    """从 AgentRun 读取预算评估事实。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
    ) -> None:
        self._session_factory = session_factory

    async def capture(
        self,
        *,
        run_id: str,
    ) -> RunBudgetSnapshot:
        async with self._session_factory() as session:
            async with session.begin():
                return await self.capture_from_session(
                    session=session,
                    run_id=run_id,
                )

    async def capture_from_session(
        self,
        *,
        session: AsyncSession,
        run_id: str,
    ) -> RunBudgetSnapshot:
        result = await session.execute(
            select(
                AgentRun.id.label("run_id"),
                AgentRun.total_steps,
                AgentRun.total_tool_calls,
                AgentRun.input_tokens,
                AgentRun.output_tokens,
                AgentRun.latency_ms,
            ).where(AgentRun.id == run_id)
        )

        row = result.mappings().one_or_none()

        if row is None:
            raise RuntimeError(f"AgentRun does not exist: {run_id}")

        return RunBudgetSnapshot.model_validate(dict(row))


__all__ = [
    "BudgetMetric",
    "BudgetMetricResult",
    "BudgetOracle",
    "BudgetOracleResult",
    "BudgetSnapshotService",
    "RunBudgetSnapshot",
    "calculate_utilization",
]
