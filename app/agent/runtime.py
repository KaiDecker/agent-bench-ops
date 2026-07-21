from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.agent.graph import build_agent_graph
from app.agent.model import AgentDecisionModel
from app.agent.recorder import RunStepRecorder
from app.agent.serialization import (
    json_safe,
    serialize_messages,
)
from app.agent.state import build_initial_state
from app.benchmark.reset import reset_business_state
from app.benchmark.schemas import BusinessInitialState
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
    RunStep,
    generate_id,
)
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolRegistry

type RuntimeStatus = Literal[
    "succeeded",
    "failed",
]


@dataclass(frozen=True, slots=True)
class PreparedAgentRun:
    """已完成任务重置并创建 AgentRun 的运行信息。"""

    run_id: str
    task_id: str
    task_key: str
    task_version: int
    user_request: str
    available_tools: tuple[str, ...]
    max_steps: int
    max_tool_calls: int


@dataclass(frozen=True, slots=True)
class RunStatistics:
    """从 run_steps 聚合出的运行统计。"""

    persisted_step_count: int
    model_step_count: int
    tool_step_count: int
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class AgentRuntimeResult:
    """统一 Runtime 返回结果。"""

    run_id: str
    task_key: str
    task_version: int
    model_provider: str
    model_name: str
    status: RuntimeStatus
    total_steps: int
    total_tool_calls: int
    persisted_step_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    final_response: str | None
    error: dict[str, Any] | None
    messages: tuple[BaseMessage, ...]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """转换成适合 API 或 JSON 输出的结构。"""

        return {
            "run_id": self.run_id,
            "task_key": self.task_key,
            "task_version": self.task_version,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "status": self.status,
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
            "persisted_step_count": (self.persisted_step_count),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "final_response": self.final_response,
            "error": json_safe(self.error),
            "messages": serialize_messages(list(self.messages)),
        }


class AgentRuntime:
    """
    Benchmark Agent 的统一运行入口。

    负责：
    1. 加载 BenchmarkTask
    2. 重置业务初始状态
    3. 创建 AgentRun
    4. 构建并执行 LangGraph
    5. 持久化 RunStep
    6. 汇总 Token、步骤和延迟
    7. 完成 AgentRun 生命周期
    """

    def __init__(
        self,
        *,
        model: AgentDecisionModel,
        model_provider: str,
        model_name: str,
        registry: ToolRegistry,
        gateway: ToolGateway,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionFactory,
    ) -> None:
        normalized_provider = model_provider.strip()
        normalized_model_name = model_name.strip()

        if not normalized_provider:
            raise ValueError("model_provider cannot be empty")

        if not normalized_model_name:
            raise ValueError("model_name cannot be empty")

        self._model_provider = normalized_provider
        self._model_name = normalized_model_name
        self._session_factory = session_factory

        self._recorder = RunStepRecorder(session_factory=session_factory)

        self._graph = build_agent_graph(
            model=model,
            registry=registry,
            gateway=gateway,
            session_factory=session_factory,
            recorder=self._recorder,
        )

    async def run_benchmark_task(
        self,
        *,
        task_key: str,
        task_version: int,
        actor_id: str,
        permissions: Sequence[str],
        prompt_version: str = "v1",
        agent_strategy: str = ("langgraph-model-tool-loop"),
        memory_strategy: str = "messages-state",
        experiment_id: str | None = None,
        random_seed: int | None = None,
        configuration: Mapping[str, Any] | None = None,
        recursion_limit: int | None = None,
    ) -> AgentRuntimeResult:
        """
        重置并执行一个 Benchmark Task。

        图内部可处理的 Agent 错误会以结构化失败结果返回。
        数据库、网络客户端或 Runtime 本身的未捕获异常会在
        AgentRun 标记失败后继续向上抛出。
        """

        prepared = await self._prepare_run(
            task_key=task_key,
            task_version=task_version,
            prompt_version=prompt_version,
            agent_strategy=agent_strategy,
            memory_strategy=memory_strategy,
            experiment_id=experiment_id,
            random_seed=random_seed,
            configuration=configuration,
        )

        initial_state = build_initial_state(
            user_request=prepared.user_request,
            run_id=prepared.run_id,
            actor_id=actor_id,
            available_tools=list(prepared.available_tools),
            permissions=list(permissions),
            max_steps=prepared.max_steps,
            max_tool_calls=prepared.max_tool_calls,
        )

        resolved_recursion_limit = (
            recursion_limit
            if recursion_limit is not None
            else max(
                12,
                prepared.max_steps * 2 + 4,
            )
        )

        started_at = perf_counter()

        try:
            graph_result = await self._graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": (resolved_recursion_limit),
                },
            )
        except Exception as exc:
            latency_ms = round(
                (perf_counter() - started_at) * 1000,
                2,
            )

            await self._finalize_exception(
                run_id=prepared.run_id,
                error=exc,
                latency_ms=latency_ms,
            )

            raise

        latency_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        error = graph_result.get("error")

        status: RuntimeStatus = "failed" if error is not None else "succeeded"

        total_steps = int(graph_result.get("step_count", 0))

        total_tool_calls = int(graph_result.get("tool_call_count", 0))

        final_response = graph_result.get("final_response")

        statistics = await self._finalize_result(
            run_id=prepared.run_id,
            status=status,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            latency_ms=latency_ms,
            final_response=final_response,
            error=error,
        )

        messages = tuple(graph_result.get("messages", []))

        return AgentRuntimeResult(
            run_id=prepared.run_id,
            task_key=prepared.task_key,
            task_version=prepared.task_version,
            model_provider=self._model_provider,
            model_name=self._model_name,
            status=status,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            persisted_step_count=(statistics.persisted_step_count),
            input_tokens=statistics.input_tokens,
            output_tokens=statistics.output_tokens,
            latency_ms=latency_ms,
            final_response=final_response,
            error=json_safe(error),
            messages=messages,
        )

    async def _prepare_run(
        self,
        *,
        task_key: str,
        task_version: int,
        prompt_version: str,
        agent_strategy: str,
        memory_strategy: str,
        experiment_id: str | None,
        random_seed: int | None,
        configuration: Mapping[str, Any] | None,
    ) -> PreparedAgentRun:
        normalized_task_key = task_key.strip()

        if not normalized_task_key:
            raise ValueError("task_key cannot be empty")

        if task_version < 1:
            raise ValueError("task_version must be positive")

        async with self._session_factory.begin() as session:
            result = await session.execute(
                select(BenchmarkTask).where(
                    BenchmarkTask.task_key == normalized_task_key,
                    BenchmarkTask.version == task_version,
                )
            )

            task = result.scalar_one_or_none()

            if task is None:
                raise LookupError(
                    f"Benchmark task does not exist: {normalized_task_key} v{task_version}"
                )

            initial_business_state = BusinessInitialState.model_validate(task.initial_state)

            await reset_business_state(
                session,
                initial_business_state,
            )

            budget = task.budget or {}

            max_steps = int(budget.get("max_agent_steps", 5))

            max_tool_calls = int(budget.get("max_tool_calls", 5))

            if max_steps < 1:
                raise ValueError("Task max_agent_steps must be positive")

            if max_tool_calls < 0:
                raise ValueError("Task max_tool_calls cannot be negative")

            run_id = generate_id()

            runtime_configuration = {
                "runtime_version": "stage5c-v1",
                **dict(configuration or {}),
            }

            run = AgentRun(
                id=run_id,
                task_id=task.id,
                experiment_id=experiment_id,
                status="running",
                model_provider=self._model_provider,
                model_name=self._model_name,
                prompt_version=prompt_version,
                agent_strategy=agent_strategy,
                memory_strategy=memory_strategy,
                input_payload={
                    "task_key": task.task_key,
                    "task_version": task.version,
                    "user_request": task.user_request,
                },
                configuration=json_safe(runtime_configuration),
                random_seed=random_seed,
                resume_count=0,
                total_steps=0,
                total_tool_calls=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                started_at=datetime.now(UTC),
            )

            session.add(run)
            await session.flush()

            return PreparedAgentRun(
                run_id=run_id,
                task_id=task.id,
                task_key=task.task_key,
                task_version=task.version,
                user_request=task.user_request,
                available_tools=tuple(task.available_tools or []),
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
            )

    async def _read_statistics(
        self,
        session: AsyncSession,
        *,
        run_id: str,
    ) -> RunStatistics:
        result = await session.execute(
            select(
                func.count(RunStep.id),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                RunStep.step_type == "model",
                                1,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                RunStep.step_type == "tool",
                                1,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(RunStep.input_tokens),
                    0,
                ),
                func.coalesce(
                    func.sum(RunStep.output_tokens),
                    0,
                ),
            ).where(RunStep.run_id == run_id)
        )

        row = result.one()

        return RunStatistics(
            persisted_step_count=int(row[0] or 0),
            model_step_count=int(row[1] or 0),
            tool_step_count=int(row[2] or 0),
            input_tokens=int(row[3] or 0),
            output_tokens=int(row[4] or 0),
        )

    async def _lock_run(
        self,
        session: AsyncSession,
        *,
        run_id: str,
    ) -> AgentRun:
        result = await session.execute(
            select(AgentRun).where(AgentRun.id == run_id).with_for_update()
        )

        run = result.scalar_one_or_none()

        if run is None:
            raise RuntimeError(f"AgentRun does not exist: {run_id}")

        return run

    async def _finalize_result(
        self,
        *,
        run_id: str,
        status: RuntimeStatus,
        total_steps: int,
        total_tool_calls: int,
        latency_ms: float,
        final_response: str | None,
        error: dict[str, Any] | None,
    ) -> RunStatistics:
        async with self._session_factory.begin() as session:
            run = await self._lock_run(
                session,
                run_id=run_id,
            )

            statistics = await self._read_statistics(
                session,
                run_id=run_id,
            )

            run.status = status
            run.total_steps = total_steps
            run.total_tool_calls = total_tool_calls
            run.input_tokens = statistics.input_tokens
            run.output_tokens = statistics.output_tokens
            run.latency_ms = latency_ms
            run.final_response = final_response

            if error is not None:
                run.error_type = str(
                    error.get(
                        "code",
                        "agent_runtime_error",
                    )
                )
                run.error_message = str(
                    error.get(
                        "message",
                        "Agent execution failed.",
                    )
                )
            else:
                run.error_type = None
                run.error_message = None

            run.finished_at = datetime.now(UTC)

            return statistics

    async def _finalize_exception(
        self,
        *,
        run_id: str,
        error: Exception,
        latency_ms: float,
    ) -> None:
        async with self._session_factory.begin() as session:
            run = await self._lock_run(
                session,
                run_id=run_id,
            )

            statistics = await self._read_statistics(
                session,
                run_id=run_id,
            )

            run.status = "failed"
            run.total_steps = statistics.model_step_count
            run.total_tool_calls = statistics.tool_step_count
            run.input_tokens = statistics.input_tokens
            run.output_tokens = statistics.output_tokens
            run.latency_ms = latency_ms
            run.final_response = None
            run.error_type = type(error).__name__
            run.error_message = str(error)
            run.finished_at = datetime.now(UTC)


__all__ = [
    "AgentRuntime",
    "AgentRuntimeResult",
    "PreparedAgentRun",
    "RunStatistics",
]
