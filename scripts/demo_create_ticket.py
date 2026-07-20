import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import func, select

from app.benchmark.reset import reset_business_state
from app.benchmark.schemas import BusinessInitialState
from app.domain.tickets import Ticket
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
    ToolOperation,
)
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolExecutionContext

TASK_KEY = "create_ticket_001"
TASK_VERSION = 1


async def prepare_run() -> str:
    async with AsyncSessionFactory.begin() as session:
        result = await session.execute(
            select(BenchmarkTask).where(
                BenchmarkTask.task_key == TASK_KEY,
                BenchmarkTask.version == TASK_VERSION,
            )
        )

        task = result.scalar_one()

        initial_state = BusinessInitialState.model_validate(task.initial_state)

        await reset_business_state(session, initial_state)

        run = AgentRun(
            task_id=task.id,
            status="running",
            model_provider="manual",
            model_name="manual-create-ticket-demo",
            prompt_version="none",
            agent_strategy="manual",
            memory_strategy="none",
            input_payload={
                "user_request": task.user_request,
            },
            configuration={
                "demo": "create_ticket",
            },
            started_at=datetime.now(UTC),
        )

        session.add(run)
        await session.flush()

        return run.id


async def execute_ticket(
    run_id: str,
    *,
    fault_injection: str = "none",
) -> dict[str, object]:
    gateway = ToolGateway(build_default_registry())

    context = ToolExecutionContext(
        run_id=run_id,
        actor_id="local-demo",
        available_tools={"create_ticket"},
        permissions={"ticket.write"},
        fault_injection=fault_injection,
    )

    arguments = {
        "requester_employee_id": "emp_001",
        "target_employee_id": "emp_002",
        "ticket_type": "general",
        "risk_level": "medium",
        "title": "数据平台访问问题",
        "description": ("李四无法访问数据平台，请协助检查账号权限"),
    }

    async with AsyncSessionFactory() as session:
        first = await gateway.execute(
            session=session,
            tool_name="create_ticket",
            arguments=arguments,
            context=context,
            idempotency_key="create-ticket-demo-001",
        )

    async with AsyncSessionFactory() as session:
        second = await gateway.execute(
            session=session,
            tool_name="create_ticket",
            arguments=arguments,
            context=context,
            idempotency_key="create-ticket-demo-001",
        )

    return {
        "first": first.model_dump(mode="json"),
        "second": second.model_dump(mode="json"),
    }


async def read_state(
    run_id: str,
) -> dict[str, object]:
    async with AsyncSessionFactory() as session:
        ticket_count = await session.scalar(select(func.count()).select_from(Ticket))

        result = await session.execute(select(ToolOperation).where(ToolOperation.run_id == run_id))

        operations = result.scalars().all()

        return {
            "ticket_count": ticket_count,
            "operations": [
                {
                    "operation_id": item.operation_id,
                    "status": item.status,
                    "retry_count": item.retry_count,
                    "error_type": item.error_type,
                }
                for item in operations
            ],
        }


async def async_main() -> None:
    normal_run_id = await prepare_run()

    normal_responses = await execute_ticket(normal_run_id)
    normal_state = await read_state(normal_run_id)

    unknown_run_id = await prepare_run()

    unknown_responses = await execute_ticket(
        unknown_run_id,
        fault_injection="drop_response_after_commit",
    )
    unknown_state = await read_state(unknown_run_id)

    print(
        json.dumps(
            {
                "normal": {
                    "run_id": normal_run_id,
                    "responses": normal_responses,
                    "state": normal_state,
                },
                "response_loss": {
                    "run_id": unknown_run_id,
                    "responses": unknown_responses,
                    "state": unknown_state,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
