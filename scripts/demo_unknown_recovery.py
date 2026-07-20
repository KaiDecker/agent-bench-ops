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
from app.tools.recovery import (
    UnknownOperationRecoveryService,
)
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolExecutionContext

TASK_KEY = "create_ticket_001"
TASK_VERSION = 1
IDEMPOTENCY_KEY = "unknown-recovery-demo-001"


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
            model_name="unknown-recovery-demo",
            prompt_version="none",
            agent_strategy="manual",
            memory_strategy="none",
            input_payload={
                "user_request": task.user_request,
            },
            configuration={
                "demo": "unknown_operation_recovery",
            },
            started_at=datetime.now(UTC),
        )

        session.add(run)
        await session.flush()

        return run.id


async def async_main() -> None:
    run_id = await prepare_run()
    gateway = ToolGateway(build_default_registry())

    arguments = {
        "requester_employee_id": "emp_001",
        "target_employee_id": "emp_002",
        "ticket_type": "general",
        "risk_level": "medium",
        "title": "数据平台访问问题",
        "description": ("李四无法访问数据平台，请协助检查账号权限"),
    }

    failure_context = ToolExecutionContext(
        run_id=run_id,
        actor_id="local-demo",
        available_tools={"create_ticket"},
        permissions={"ticket.write"},
        fault_injection="drop_response_after_commit",
    )

    async with AsyncSessionFactory() as session:
        first_response = await gateway.execute(
            session=session,
            tool_name="create_ticket",
            arguments=arguments,
            context=failure_context,
            idempotency_key=IDEMPOTENCY_KEY,
        )

    if first_response.operation_id is None:
        raise RuntimeError("The timed-out response has no operation ID.")

    recovery_service = UnknownOperationRecoveryService()

    recovery_response = await recovery_service.recover(first_response.operation_id)

    replay_context = ToolExecutionContext(
        run_id=run_id,
        actor_id="local-demo",
        available_tools={"create_ticket"},
        permissions={"ticket.write"},
    )

    async with AsyncSessionFactory() as session:
        replay_response = await gateway.execute(
            session=session,
            tool_name="create_ticket",
            arguments=arguments,
            context=replay_context,
            idempotency_key=IDEMPOTENCY_KEY,
        )

    async with AsyncSessionFactory.begin() as session:
        ticket_count = await session.scalar(select(func.count()).select_from(Ticket))

        operation_result = await session.execute(
            select(ToolOperation).where(ToolOperation.operation_id == first_response.operation_id)
        )

        operation = operation_result.scalar_one()

        run = await session.get(AgentRun, run_id)

        if run is not None:
            run.status = "succeeded"
            run.total_tool_calls = 2
            run.finished_at = datetime.now(UTC)

        final_ledger = {
            "operation_id": operation.operation_id,
            "status": operation.status,
            "external_reference": (operation.external_reference),
            "retry_count": operation.retry_count,
            "recovery_count": operation.recovery_count,
            "recovered_at": (
                operation.recovered_at.isoformat() if operation.recovered_at is not None else None
            ),
            "recovery_details": (operation.recovery_details),
        }

    print(
        json.dumps(
            {
                "run_id": run_id,
                "first_response": (first_response.model_dump(mode="json")),
                "recovery_response": (recovery_response.model_dump(mode="json")),
                "replay_response": (replay_response.model_dump(mode="json")),
                "ticket_count": ticket_count,
                "ledger": final_ledger,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
