import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.benchmark.reset import reset_business_state
from app.benchmark.schemas import BusinessInitialState
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
)
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolExecutionContext

TASK_KEY = "employee_lookup_001"
TASK_VERSION = 1


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
            model_name="manual-read-tools-demo",
            prompt_version="none",
            agent_strategy="manual",
            memory_strategy="none",
            input_payload={
                "user_request": task.user_request,
            },
            configuration={
                "demo": "read_business_tools",
            },
            started_at=datetime.now(UTC),
        )

        session.add(run)
        await session.flush()

        return run.id


async def invoke(
    gateway: ToolGateway,
    *,
    context: ToolExecutionContext,
    tool_name: str,
    arguments: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    async with AsyncSessionFactory() as session:
        response = await gateway.execute(
            session=session,
            tool_name=tool_name,
            arguments=arguments,
            context=context,
            idempotency_key=idempotency_key,
        )

    return response.model_dump(mode="json")


async def async_main() -> None:
    run_id = await prepare_run()
    gateway = ToolGateway(build_default_registry())

    context = ToolExecutionContext(
        run_id=run_id,
        actor_id="local-demo",
        available_tools={
            "get_account",
            "list_employee_permissions",
            "create_ticket",
            "get_ticket",
        },
        permissions={
            "account.read",
            "permission.read",
            "ticket.write",
            "ticket.read",
        },
    )

    account_response = await invoke(
        gateway,
        context=context,
        tool_name="get_account",
        arguments={
            "employee_id": "emp_001",
        },
        idempotency_key="read-demo:get-account",
    )

    permissions_response = await invoke(
        gateway,
        context=context,
        tool_name="list_employee_permissions",
        arguments={
            "employee_id": "emp_001",
        },
        idempotency_key="read-demo:list-permissions",
    )

    create_response = await invoke(
        gateway,
        context=context,
        tool_name="create_ticket",
        arguments={
            "requester_employee_id": "emp_001",
            "target_employee_id": "emp_001",
            "ticket_type": "general",
            "risk_level": "low",
            "title": "只读工具演示工单",
            "description": ("用于验证 get_ticket 工具的本地演示工单。"),
        },
        idempotency_key="read-demo:create-ticket",
    )

    create_output = create_response.get("output")

    if not isinstance(create_output, dict):
        raise RuntimeError("create_ticket did not return an output")

    ticket_data = create_output.get("ticket")

    if not isinstance(ticket_data, dict):
        raise RuntimeError("create_ticket did not return ticket data")

    ticket_id = ticket_data.get("id")

    if not isinstance(ticket_id, str):
        raise RuntimeError("create_ticket returned an invalid ticket ID")

    ticket_response = await invoke(
        gateway,
        context=context,
        tool_name="get_ticket",
        arguments={
            "ticket_id": ticket_id,
        },
        idempotency_key="read-demo:get-ticket",
    )

    async with AsyncSessionFactory.begin() as session:
        run = await session.get(AgentRun, run_id)

        if run is not None:
            run.status = "succeeded"
            run.total_tool_calls = 4
            run.finished_at = datetime.now(UTC)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "account": account_response,
                "permissions": permissions_response,
                "created_ticket": create_response,
                "ticket": ticket_response,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
