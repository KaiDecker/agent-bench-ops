import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from sqlalchemy import select

from app.agent.graph import build_agent_graph
from app.agent.model import OpenAIToolCallingModel
from app.agent.state import build_initial_state
from app.benchmark.reset import reset_business_state
from app.benchmark.schemas import BusinessInitialState
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
)
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry

TASK_KEY = "employee_lookup_001"
TASK_VERSION = 1


def serialize_message(
    message: BaseMessage,
) -> dict[str, Any]:
    serialized: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }

    if isinstance(message, AIMessage):
        serialized["tool_calls"] = message.tool_calls
        serialized["usage_metadata"] = message.usage_metadata
        serialized["response_metadata"] = message.response_metadata

    if isinstance(message, ToolMessage):
        serialized["tool_call_id"] = message.tool_call_id

        if isinstance(message.content, str):
            try:
                serialized["parsed_content"] = json.loads(message.content)
            except json.JSONDecodeError:
                pass

    return serialized


async def prepare_run(
    *,
    model_name: str,
) -> tuple[str, BenchmarkTask]:
    async with AsyncSessionFactory.begin() as session:
        result = await session.execute(
            select(BenchmarkTask).where(
                BenchmarkTask.task_key == TASK_KEY,
                BenchmarkTask.version == TASK_VERSION,
            )
        )

        task = result.scalar_one_or_none()

        if task is None:
            raise RuntimeError(f"Task not found: {TASK_KEY} v{TASK_VERSION}")

        initial_state = BusinessInitialState.model_validate(task.initial_state)

        await reset_business_state(
            session,
            initial_state,
        )

        run = AgentRun(
            task_id=task.id,
            status="running",
            model_provider="openai",
            model_name=model_name,
            prompt_version="stage5b-v1",
            agent_strategy=("langgraph-model-tool-loop"),
            memory_strategy="messages-state",
            input_payload={
                "user_request": task.user_request,
            },
            configuration={
                "stage": "5B",
                "parallel_tool_calls": False,
            },
            started_at=datetime.now(UTC),
        )

        session.add(run)
        await session.flush()

        return run.id, task


async def async_main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get(
        "OPENAI_MODEL",
        "gpt-5-mini",
    )

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the current environment.")

    run_id, task = await prepare_run(
        model_name=model_name,
    )

    registry = build_default_registry()
    gateway = ToolGateway(registry)

    model = OpenAIToolCallingModel(
        model_name=model_name,
        api_key=api_key,
        timeout_seconds=60.0,
        max_retries=2,
    )

    graph = build_agent_graph(
        model=model,
        registry=registry,
        gateway=gateway,
    )

    budget = task.budget or {}

    initial_state = build_initial_state(
        user_request=task.user_request,
        run_id=run_id,
        actor_id="benchmark-agent",
        available_tools=list(task.available_tools),
        permissions=[
            "employee.read",
        ],
        max_steps=int(budget.get("max_agent_steps", 5)),
        max_tool_calls=int(budget.get("max_tool_calls", 2)),
    )

    try:
        result = await graph.ainvoke(
            initial_state,
            config={
                "recursion_limit": 12,
            },
        )
    except Exception as exc:
        async with AsyncSessionFactory.begin() as session:
            run = await session.get(AgentRun, run_id)

            if run is not None:
                run.status = "failed"
                run.error_type = type(exc).__name__
                run.error_message = str(exc)
                run.finished_at = datetime.now(UTC)

        raise

    async with AsyncSessionFactory.begin() as session:
        run = await session.get(AgentRun, run_id)

        if run is None:
            raise RuntimeError("AgentRun disappeared")

        run.status = "failed" if result["error"] is not None else "succeeded"

        run.total_steps = result["step_count"]
        run.total_tool_calls = result["tool_call_count"]
        run.final_response = result["final_response"]

        if result["error"] is not None:
            run.error_type = result["error"]["code"]
            run.error_message = result["error"]["message"]

        run.finished_at = datetime.now(UTC)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "task_key": TASK_KEY,
                "model_provider": "openai",
                "model_name": model_name,
                "status": ("failed" if result["error"] is not None else "succeeded"),
                "step_count": result["step_count"],
                "tool_call_count": (result["tool_call_count"]),
                "final_response": (result["final_response"]),
                "error": result["error"],
                "messages": [serialize_message(message) for message in result["messages"]],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
