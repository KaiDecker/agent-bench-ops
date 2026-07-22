from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
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
from app.agent.state import (
    AgentState,
    build_initial_state,
)
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

type CheckpointerFactory = Callable[
    [],
    AbstractAsyncContextManager[BaseCheckpointSaver],
]

type RuntimeStatus = Literal[
    "paused",
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
    checkpoint_ref: str | None


@dataclass(frozen=True, slots=True)
class PreparedResumeRun:
    """已经取得恢复执行权的 AgentRun。"""

    run_id: str
    checkpoint_ref: str
    task_key: str
    task_version: int
    previous_latency_ms: float
    breakpoint_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphInvocationOutcome:
    """一次 LangGraph 调用的结果和当前调度位置。"""

    state: dict[str, Any]
    next_nodes: tuple[str, ...]

    @property
    def is_paused(self) -> bool:
        return bool(self.next_nodes)


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
    checkpoint_ref: str | None
    task_key: str
    task_version: int
    model_provider: str
    model_name: str
    status: RuntimeStatus
    next_nodes: tuple[str, ...]
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
            "next_nodes": list(self.next_nodes),
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
            "checkpoint_ref": self.checkpoint_ref,
        }


def validate_resume_configuration(
    *,
    run_id: str,
    configuration: Mapping[str, Any],
) -> tuple[str, ...]:
    """
    校验 AgentRun 是否处于可恢复状态，并返回待执行节点。

    resume_in_progress 优先于 paused 检查，使并发恢复请求
    能得到准确的冲突原因。
    """

    if (
        configuration.get(
            "resume_in_progress",
            False,
        )
        is True
    ):
        raise RuntimeError(f"AgentRun resume is already in progress: {run_id}")

    if (
        configuration.get(
            "paused",
            False,
        )
        is not True
    ):
        raise RuntimeError(f"AgentRun is not marked as paused: {run_id}")

    raw_nodes = configuration.get("next_nodes")

    if (
        not isinstance(
            raw_nodes,
            list,
        )
        or not raw_nodes
    ):
        raise RuntimeError(f"Paused AgentRun has no pending nodes: {run_id}")

    if not all(isinstance(node, str) and bool(node.strip()) for node in raw_nodes):
        raise RuntimeError(f"Paused AgentRun contains invalid pending nodes: {run_id}")

    return tuple(node.strip() for node in raw_nodes)


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
        checkpointer_factory: CheckpointerFactory | None = None,
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
        self._model = model
        self._registry = registry
        self._gateway = gateway
        self._checkpointer_factory = checkpointer_factory

        self._recorder = RunStepRecorder(session_factory=session_factory)

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
        pause_before_tools: bool = False,
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
            graph_config: dict[str, Any] = {
                "recursion_limit": resolved_recursion_limit,
            }

            if prepared.checkpoint_ref is not None:
                graph_config["configurable"] = {
                    "thread_id": prepared.checkpoint_ref,
                }

            outcome = await self._invoke_graph(
                initial_state=initial_state,
                config=graph_config,
                interrupt_before=(("tools",) if pause_before_tools else None),
            )

            graph_result = outcome.state
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

        if outcome.is_paused:
            statistics = await self._record_pause(
                run_id=prepared.run_id,
                total_steps=total_steps,
                total_tool_calls=total_tool_calls,
                latency_ms=latency_ms,
                next_nodes=outcome.next_nodes,
            )

            messages = tuple(graph_result.get("messages", []))

            return AgentRuntimeResult(
                run_id=prepared.run_id,
                checkpoint_ref=prepared.checkpoint_ref,
                task_key=prepared.task_key,
                task_version=prepared.task_version,
                model_provider=self._model_provider,
                model_name=self._model_name,
                status="paused",
                next_nodes=outcome.next_nodes,
                total_steps=total_steps,
                total_tool_calls=total_tool_calls,
                persisted_step_count=(statistics.persisted_step_count),
                input_tokens=statistics.input_tokens,
                output_tokens=statistics.output_tokens,
                latency_ms=latency_ms,
                final_response=None,
                error=None,
                messages=messages,
            )

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
            checkpoint_ref=prepared.checkpoint_ref,
            task_key=prepared.task_key,
            task_version=prepared.task_version,
            model_provider=self._model_provider,
            model_name=self._model_name,
            status=status,
            next_nodes=outcome.next_nodes,
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

    async def resume_benchmark_run(
        self,
        *,
        run_id: str,
        recursion_limit: int = 12,
    ) -> AgentRuntimeResult:
        """
        从 PostgreSQL checkpoint 恢复一个暂停的 AgentRun。

        本方法不会创建新的 AgentRun，也不会重置 Benchmark 数据。
        """

        if self._checkpointer_factory is None:
            raise RuntimeError("Checkpoint recovery requires a checkpointer.")

        prepared = await self._claim_resume(
            run_id=run_id,
        )

        graph_config: dict[str, Any] = {
            "recursion_limit": recursion_limit,
            "configurable": {
                "thread_id": (prepared.checkpoint_ref),
            },
        }

        started_at = perf_counter()

        try:
            outcome = await self._invoke_graph(
                initial_state=None,
                config=graph_config,
                interrupt_before=(prepared.breakpoint_nodes),
            )

            graph_result = outcome.state
        except Exception as exc:
            segment_latency_ms = round(
                (perf_counter() - started_at) * 1000,
                2,
            )

            total_latency_ms = round(
                prepared.previous_latency_ms + segment_latency_ms,
                2,
            )

            await self._finalize_exception(
                run_id=prepared.run_id,
                error=exc,
                latency_ms=total_latency_ms,
            )

            raise

        segment_latency_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        total_latency_ms = round(
            prepared.previous_latency_ms + segment_latency_ms,
            2,
        )

        error = graph_result.get("error")

        status: RuntimeStatus = "failed" if error is not None else "succeeded"

        total_steps = int(
            graph_result.get(
                "step_count",
                0,
            )
        )

        total_tool_calls = int(
            graph_result.get(
                "tool_call_count",
                0,
            )
        )

        final_response = graph_result.get("final_response")

        messages = tuple(
            graph_result.get(
                "messages",
                [],
            )
        )

        if outcome.is_paused:
            statistics = await self._record_pause(
                run_id=prepared.run_id,
                total_steps=total_steps,
                total_tool_calls=total_tool_calls,
                latency_ms=total_latency_ms,
                next_nodes=outcome.next_nodes,
            )

            return AgentRuntimeResult(
                run_id=prepared.run_id,
                checkpoint_ref=(prepared.checkpoint_ref),
                task_key=prepared.task_key,
                task_version=(prepared.task_version),
                model_provider=(self._model_provider),
                model_name=self._model_name,
                status="paused",
                next_nodes=outcome.next_nodes,
                total_steps=total_steps,
                total_tool_calls=total_tool_calls,
                persisted_step_count=(statistics.persisted_step_count),
                input_tokens=(statistics.input_tokens),
                output_tokens=(statistics.output_tokens),
                latency_ms=total_latency_ms,
                final_response=None,
                error=None,
                messages=messages,
            )

        statistics = await self._finalize_result(
            run_id=prepared.run_id,
            status=status,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            latency_ms=total_latency_ms,
            final_response=final_response,
            error=error,
        )

        return AgentRuntimeResult(
            run_id=prepared.run_id,
            checkpoint_ref=(prepared.checkpoint_ref),
            task_key=prepared.task_key,
            task_version=prepared.task_version,
            model_provider=self._model_provider,
            model_name=self._model_name,
            status=status,
            next_nodes=outcome.next_nodes,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            persisted_step_count=(statistics.persisted_step_count),
            input_tokens=statistics.input_tokens,
            output_tokens=statistics.output_tokens,
            latency_ms=total_latency_ms,
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

            checkpoint_ref = run_id if self._checkpointer_factory is not None else None

            runtime_configuration = {
                **dict(configuration or {}),
                "runtime_version": "stage5d-v1",
                "checkpoint_enabled": (checkpoint_ref is not None),
                "checkpoint_thread_id": checkpoint_ref,
            }

            run = AgentRun(
                id=run_id,
                checkpoint_ref=checkpoint_ref,
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
                checkpoint_ref=checkpoint_ref,
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

            configuration = dict(run.configuration or {})

            if "paused" in configuration or "resume_in_progress" in configuration:
                configuration.pop(
                    "resume_started_at",
                    None,
                )
                configuration.pop(
                    "pause_reason",
                    None,
                )

                configuration.update(
                    {
                        "paused": False,
                        "next_nodes": [],
                        "resume_in_progress": False,
                        "resume_status": ("completed" if status == "succeeded" else "failed"),
                    }
                )

                run.configuration = configuration

            run.finished_at = datetime.now(UTC)

            return statistics

    async def _record_pause(
        self,
        *,
        run_id: str,
        total_steps: int,
        total_tool_calls: int,
        latency_ms: float,
        next_nodes: tuple[str, ...],
    ) -> RunStatistics:
        """保存暂停时已经完成的部分运行统计。"""

        async with self._session_factory.begin() as session:
            run = await self._lock_run(
                session,
                run_id=run_id,
            )

            statistics = await self._read_statistics(
                session,
                run_id=run_id,
            )

            run.status = "running"
            run.total_steps = total_steps
            run.total_tool_calls = total_tool_calls
            run.input_tokens = statistics.input_tokens
            run.output_tokens = statistics.output_tokens
            run.latency_ms = latency_ms
            run.final_response = None
            run.error_type = None
            run.error_message = None
            run.finished_at = None

            configuration = dict(run.configuration or {})

            configuration.pop(
                "resume_started_at",
                None,
            )

            run.configuration = {
                **configuration,
                "paused": True,
                "pause_reason": "static_breakpoint",
                "next_nodes": list(next_nodes),
                "resume_in_progress": False,
                "resume_status": "paused",
            }

            return statistics

    async def _claim_resume(
        self,
        *,
        run_id: str,
    ) -> PreparedResumeRun:
        """
        原子取得一个暂停运行的恢复执行权。

        数据库行锁用于保证两个进程不能同时声明同一个恢复任务；
        resume_in_progress 用于在事务提交后继续阻止重复恢复。
        """

        async with self._session_factory.begin() as session:
            run = await self._lock_run(
                session,
                run_id=run_id,
            )

            if run.status != "running":
                raise RuntimeError(
                    f"Only running AgentRuns can be resumed: run_id={run_id}, status={run.status}"
                )

            if run.checkpoint_ref is None:
                raise RuntimeError(f"AgentRun has no checkpoint_ref: {run_id}")

            configuration = dict(run.configuration or {})

            breakpoint_nodes = validate_resume_configuration(
                run_id=run_id,
                configuration=configuration,
            )

            if run.model_provider != self._model_provider:
                raise RuntimeError(
                    "Runtime model provider does not match "
                    "the paused AgentRun: "
                    f"expected={run.model_provider}, "
                    f"actual={self._model_provider}"
                )

            if run.model_name != self._model_name:
                raise RuntimeError(
                    "Runtime model name does not match "
                    "the paused AgentRun: "
                    f"expected={run.model_name}, "
                    f"actual={self._model_name}"
                )

            task = await session.get(
                BenchmarkTask,
                run.task_id,
            )

            if task is None:
                raise RuntimeError(f"BenchmarkTask does not exist for AgentRun: {run_id}")

            run.resume_count = int(run.resume_count or 0) + 1

            run.configuration = {
                **configuration,
                "paused": False,
                "resume_in_progress": True,
                "resume_status": "in_progress",
                "resume_started_at": (datetime.now(UTC).isoformat()),
            }

            await session.flush()

            return PreparedResumeRun(
                run_id=run.id,
                checkpoint_ref=run.checkpoint_ref,
                task_key=task.task_key,
                task_version=task.version,
                previous_latency_ms=float(run.latency_ms or 0.0),
                breakpoint_nodes=breakpoint_nodes,
            )

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

            configuration = dict(run.configuration or {})

            if "paused" in configuration or "resume_in_progress" in configuration:
                configuration.pop(
                    "resume_started_at",
                    None,
                )

                configuration.update(
                    {
                        "paused": False,
                        "next_nodes": [],
                        "resume_in_progress": False,
                        "resume_status": "failed",
                    }
                )

                run.configuration = configuration

            run.finished_at = datetime.now(UTC)

    async def _invoke_graph(
        self,
        *,
        initial_state: AgentState | None,
        config: dict[str, Any],
        interrupt_before: Sequence[str] | None = None,
    ) -> GraphInvocationOutcome:
        """
        在 Checkpointer 资源有效期内编译、执行并读取图状态。
        """

        if interrupt_before and self._checkpointer_factory is None:
            raise ValueError("Graph breakpoints require a checkpointer.")

        if self._checkpointer_factory is None:
            graph = build_agent_graph(
                model=self._model,
                registry=self._registry,
                gateway=self._gateway,
                session_factory=self._session_factory,
                recorder=self._recorder,
                checkpointer=None,
                interrupt_before=None,
            )

            state = await graph.ainvoke(
                initial_state,
                config=config,
            )

            return GraphInvocationOutcome(
                state=state,
                next_nodes=(),
            )

        async with self._checkpointer_factory() as checkpointer:
            graph = build_agent_graph(
                model=self._model,
                registry=self._registry,
                gateway=self._gateway,
                session_factory=self._session_factory,
                recorder=self._recorder,
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
            )

            state = await graph.ainvoke(
                initial_state,
                config=config,
            )

            snapshot = await graph.aget_state(config)

            return GraphInvocationOutcome(
                state=state,
                next_nodes=tuple(snapshot.next),
            )


__all__ = [
    "AgentRuntime",
    "AgentRuntimeResult",
    "GraphInvocationOutcome",
    "PreparedAgentRun",
    "PreparedResumeRun",
    "RunStatistics",
    "validate_resume_configuration",
]
