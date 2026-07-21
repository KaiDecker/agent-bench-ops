import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import func, select

from app.benchmark.reset import reset_business_state
from app.benchmark.schemas import BusinessInitialState
from app.domain.tickets import Ticket, TicketMutation
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    BenchmarkTask,
    ToolOperation,
)
from app.tools.gateway import ToolGateway
from app.tools.recovery import (
    UnknownOperationRecoveryService,
)
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

        await reset_business_state(
            session,
            initial_state,
        )

        run = AgentRun(
            task_id=task.id,
            status="running",
            model_provider="manual",
            model_name="update-ticket-recovery-demo",
            prompt_version="none",
            agent_strategy="manual",
            memory_strategy="none",
            input_payload={
                "user_request": task.user_request,
            },
            configuration={
                "demo": "update_ticket_recovery",
            },
            started_at=datetime.now(UTC),
        )

        session.add(run)
        await session.flush()

        return run.id


async def async_main() -> None:
    run_id = await prepare_run()
    gateway = ToolGateway(build_default_registry())

    normal_context = ToolExecutionContext(
        run_id=run_id,
        actor_id="local-demo",
        available_tools={
            "create_ticket",
            "update_ticket",
        },
        permissions={
            "ticket.write",
        },
    )

    create_arguments = {
        "requester_employee_id": "emp_001",
        "target_employee_id": "emp_002",
        "ticket_type": "general",
        "risk_level": "medium",
        "title": "数据平台访问问题",
        "description": ("李四无法访问数据平台，请协助检查账号权限"),
    }

    async with AsyncSessionFactory() as session:
        create_response = await gateway.execute(
            session=session,
            tool_name="create_ticket",
            arguments=create_arguments,
            context=normal_context,
            idempotency_key=("update-ticket-demo:create"),
        )

    create_output = create_response.output

    if create_output is None:
        raise RuntimeError("create_ticket returned no output")

    ticket_data = create_output.get("ticket")

    if not isinstance(ticket_data, dict):
        raise RuntimeError("create_ticket returned invalid ticket data")

    ticket_id = ticket_data.get("id")
    ticket_version = ticket_data.get("version")

    if not isinstance(ticket_id, str):
        raise RuntimeError("Invalid ticket ID")

    if not isinstance(ticket_version, int):
        raise RuntimeError("Invalid ticket version")

    failure_context = normal_context.model_copy(
        update={"fault_injection": ("drop_response_after_commit")}
    )

    update_arguments = {
        "ticket_id": ticket_id,
        "expected_version": ticket_version,
        "title": "数据平台访问问题（处理中）",
        "description": ("已确认李四无法访问数据平台，正在检查账号权限。"),
    }

    async with AsyncSessionFactory() as session:
        first_update_response = await gateway.execute(
            session=session,
            tool_name="update_ticket",
            arguments=update_arguments,
            context=failure_context,
            idempotency_key=("update-ticket-demo:update"),
        )

    if first_update_response.operation_id is None:
        raise RuntimeError("Timed-out update has no operation ID")

    recovery_service = UnknownOperationRecoveryService()

    recovery_response = await recovery_service.recover(first_update_response.operation_id)

    async with AsyncSessionFactory() as session:
        replay_response = await gateway.execute(
            session=session,
            tool_name="update_ticket",
            arguments=update_arguments,
            context=normal_context,
            idempotency_key=("update-ticket-demo:update"),
        )

    async with AsyncSessionFactory.begin() as session:
        ticket_result = await session.execute(select(Ticket).where(Ticket.id == ticket_id))

        ticket = ticket_result.scalar_one()

        mutation_count = await session.scalar(
            select(func.count())
            .select_from(TicketMutation)
            .where(TicketMutation.ticket_id == ticket_id)
        )

        operation_result = await session.execute(
            select(ToolOperation).where(
                ToolOperation.operation_id == first_update_response.operation_id
            )
        )

        operation = operation_result.scalar_one()

        run = await session.get(AgentRun, run_id)

        if run is not None:
            run.status = "succeeded"
            run.total_tool_calls = 3
            run.finished_at = datetime.now(UTC)

        final_state = {
            "ticket_id": ticket.id,
            "title": ticket.title,
            "description": ticket.description,
            "status": ticket.status,
            "version": ticket.version,
            "mutation_count": mutation_count,
            "operation_status": operation.status,
            "retry_count": operation.retry_count,
            "recovery_count": (operation.recovery_count),
            "external_reference": (operation.external_reference),
        }

    print(
        json.dumps(
            {
                "run_id": run_id,
                "create_response": (create_response.model_dump(mode="json")),
                "first_update_response": (first_update_response.model_dump(mode="json")),
                "recovery_response": (recovery_response.model_dump(mode="json")),
                "replay_response": (replay_response.model_dump(mode="json")),
                "final_state": final_state,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
