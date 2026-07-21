from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    RunStep,
    generate_id,
)

type RunStepType = Literal[
    "model",
    "tool",
]

type RunStepStatus = Literal[
    "succeeded",
    "failed",
]


@dataclass(frozen=True, slots=True)
class StartedRunStep:
    """已创建的运行步骤标识。"""

    id: str
    step_number: int


class RunStepRecorderProtocol(Protocol):
    """节点所需的最小步骤记录接口。"""

    async def start_step(
        self,
        *,
        run_id: str,
        parent_step_id: str | None,
        step_number: int,
        step_type: RunStepType,
        model_name: str | None,
        tool_name: str | None,
        input_payload: dict[str, Any],
    ) -> StartedRunStep:
        """创建 running 状态步骤。"""
        ...

    async def finish_step(
        self,
        *,
        step_id: str,
        status: RunStepStatus,
        output_payload: dict[str, Any] | None,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        error_type: str | None,
        error_message: str | None,
    ) -> None:
        """完成一个步骤。"""
        ...


class RunStepRecorder:
    """将 LangGraph 节点执行写入 run_steps。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = (AsyncSessionFactory),
    ) -> None:
        self._session_factory = session_factory

    async def start_step(
        self,
        *,
        run_id: str,
        parent_step_id: str | None,
        step_number: int,
        step_type: RunStepType,
        model_name: str | None,
        tool_name: str | None,
        input_payload: dict[str, Any],
    ) -> StartedRunStep:
        if step_number < 1:
            raise ValueError("step_number must be positive")

        step_id = generate_id()

        async with self._session_factory.begin() as session:
            step = RunStep(
                id=step_id,
                run_id=run_id,
                parent_step_id=parent_step_id,
                step_no=step_number,
                step_type=step_type,
                status="running",
                model_name=model_name,
                tool_name=tool_name,
                input_payload=input_payload,
                output_payload=None,
                input_tokens=0,
                output_tokens=0,
                latency_ms=None,
                error_type=None,
                error_message=None,
                started_at=datetime.now(UTC),
                finished_at=None,
            )

            session.add(step)
            await session.flush()

        return StartedRunStep(
            id=step_id,
            step_number=step_number,
        )

    async def finish_step(
        self,
        *,
        step_id: str,
        status: RunStepStatus,
        output_payload: dict[str, Any] | None,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        error_type: str | None,
        error_message: str | None,
    ) -> None:
        async with self._session_factory.begin() as session:
            result = await session.execute(
                select(RunStep).where(RunStep.id == step_id).with_for_update()
            )

            step = result.scalar_one_or_none()

            if step is None:
                raise RuntimeError(f"RunStep does not exist: {step_id}")

            step.status = status
            step.output_payload = output_payload
            step.input_tokens = input_tokens
            step.output_tokens = output_tokens
            step.latency_ms = latency_ms
            step.error_type = error_type
            step.error_message = error_message
            step.finished_at = datetime.now(UTC)
