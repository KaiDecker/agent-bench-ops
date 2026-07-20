import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import select

from app.benchmark.reset import reset_business_state
from app.benchmark.schemas import BusinessInitialState
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
    ToolOperation,
)
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolExecutionContext

TASK_KEY = "employee_lookup_001"
TASK_VERSION = 1
IDEMPOTENCY_KEY = "demo:get_employee:emp_001"


async def prepare_run() -> str:
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
            model_provider="manual",
            model_name="manual-tool-ledger-demo",
            prompt_version="none",
            agent_strategy="manual",
            memory_strategy="none",
            input_payload={
                "user_request": task.user_request,
            },
            configuration={
                "demo": "tool_operation_ledger",
            },
            started_at=datetime.now(UTC),
        )

        session.add(run)
        await session.flush()

        return run.id


async def async_main() -> None:
    run_id = await prepare_run()

    gateway = ToolGateway(build_default_registry())

    context = ToolExecutionContext(
        run_id=run_id,
        actor_id="local-demo",
        available_tools={"get_employee"},
        permissions={"employee.read"},
    )

    async with AsyncSessionFactory() as session:
        first_response = await gateway.execute(
            session=session,
            tool_name="get_employee",
            arguments={"employee_id": "emp_001"},
            context=context,
            idempotency_key=IDEMPOTENCY_KEY,
        )

        replay_response = await gateway.execute(
            session=session,
            tool_name="get_employee",
            arguments={"employee_id": "emp_001"},
            context=context,
            idempotency_key=IDEMPOTENCY_KEY,
        )

    async with AsyncSessionFactory.begin() as session:
        run = await session.get(AgentRun, run_id)

        if run is None:
            raise RuntimeError("Demo AgentRun disappeared")

        run.status = "succeeded"
        run.total_tool_calls = 2
        run.finished_at = datetime.now(UTC)

        result = await session.execute(select(ToolOperation).where(ToolOperation.run_id == run_id))

        operations = result.scalars().all()

        ledger_summary = [
            {
                "operation_id": item.operation_id,
                "tool_name": item.tool_name,
                "status": item.status,
                "arguments_hash": item.arguments_hash,
                "idempotency_key": item.idempotency_key,
                "retry_count": item.retry_count,
                "latency_ms": item.latency_ms,
            }
            for item in operations
        ]

    output = {
        "run_id": run_id,
        "first_response": first_response.model_dump(mode="json"),
        "replay_response": replay_response.model_dump(mode="json"),
        "ledger": ledger_summary,
    }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
