import asyncio
import json
import os
import selectors
import sys
from typing import Any

_TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}


def assert_strict_msgpack_enabled() -> None:
    value = os.environ.get(
        "LANGGRAPH_STRICT_MSGPACK",
        "",
    )

    if value.strip().lower() not in _TRUE_VALUES:
        raise RuntimeError("LANGGRAPH_STRICT_MSGPACK must be set to true before starting Python.")


def compact_evaluation(
    report: Any,
) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "state_source": report.state_source,
        "passed": report.passed,
        "overall_score": report.overall_score,
        "final_state_score": (report.final_state_score),
        "trace_score": report.trace_score,
        "temporal_score": (report.temporal_score),
        "budget_score": report.budget_score,
        "violation_codes": [violation.code for violation in report.violations],
    }


async def read_verification(
    *,
    run_id: str,
) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.persistence.database import (
        AsyncSessionFactory,
    )
    from app.persistence.models import Ticket
    from app.persistence.platform_models import (
        EvaluationResult,
        ToolOperation,
    )

    async with AsyncSessionFactory() as session:
        ticket_result = await session.execute(
            select(Ticket)
            .where(
                Ticket.requester_employee_id == "emp_001",
                Ticket.target_employee_id == "emp_002",
                Ticket.ticket_type == "general",
                Ticket.title == "数据平台访问问题",
            )
            .order_by(Ticket.id)
        )

        tickets = list(ticket_result.scalars().all())

        operation_result = await session.execute(
            select(ToolOperation)
            .where(
                ToolOperation.run_id == run_id,
                ToolOperation.tool_name == "create_ticket",
            )
            .order_by(
                ToolOperation.created_at,
                ToolOperation.operation_id,
            )
        )

        operations = list(operation_result.scalars().all())

        evaluation_result = await session.execute(
            select(EvaluationResult).where(EvaluationResult.run_id == run_id)
        )

        evaluation = evaluation_result.scalar_one_or_none()

        evaluation_count_result = await session.execute(
            select(func.count(EvaluationResult.id)).where(EvaluationResult.run_id == run_id)
        )

        evaluation_count = int(evaluation_count_result.scalar_one())

        if evaluation is None:
            raise RuntimeError("EvaluationResult was not persisted.")

        ticket = tickets[0] if len(tickets) == 1 else None

        operation = operations[0] if len(operations) == 1 else None

        return {
            "ticket_count": len(tickets),
            "ticket_id": (ticket.id if ticket is not None else None),
            "ticket_title": (ticket.title if ticket is not None else None),
            "ticket_status": (ticket.status if ticket is not None else None),
            "ticket_source_operation_id": (
                ticket.source_operation_id if ticket is not None else None
            ),
            "tool_operation_count": (len(operations)),
            "tool_operation_ids": [item.operation_id for item in operations],
            "tool_operation_statuses": [item.status for item in operations],
            "source_operation_matches": (
                ticket is not None
                and operation is not None
                and ticket.source_operation_id == operation.operation_id
            ),
            "evaluation_count": (evaluation_count),
            "evaluation_passed": (evaluation.passed),
            "database_state_source": (evaluation.scores.get("state_source")),
        }


async def async_main() -> None:
    assert_strict_msgpack_enabled()

    from app.agent.checkpoint import (
        open_postgres_checkpointer,
    )
    from app.agent.model import (
        ScriptedCreateTicketModel,
    )
    from app.agent.runtime import AgentRuntime
    from app.evaluation.evaluator import (
        EvaluationService,
    )
    from app.persistence.database import engine
    from app.tools.gateway import ToolGateway
    from app.tools.registry import (
        build_default_registry,
    )

    try:
        registry = build_default_registry()
        gateway = ToolGateway(registry)

        runtime = AgentRuntime(
            model=ScriptedCreateTicketModel(),
            model_provider="scripted",
            model_name=("scripted-create-ticket-evaluation-v1"),
            registry=registry,
            gateway=gateway,
            checkpointer_factory=(open_postgres_checkpointer),
        )

        runtime_result = await runtime.run_benchmark_task(
            task_key="create_ticket_001",
            task_version=1,
            actor_id="benchmark-agent",
            permissions=[
                "ticket.write",
            ],
            prompt_version=("stage6-evaluation-v1"),
            agent_strategy=("langgraph-model-tool-loop"),
            memory_strategy=("messages-state"),
            configuration={
                "stage": "6F",
                "test_case": ("run_and_evaluate_create_ticket"),
                "checkpoint_backend": ("postgresql"),
            },
        )

        if runtime_result.status != "succeeded":
            raise RuntimeError(f"AgentRuntime did not succeed: {runtime_result.status}")

        evaluation_service = EvaluationService()

        live_report = await evaluation_service.evaluate_run(
            run_id=runtime_result.run_id,
            capture_live_state=True,
        )

        persisted_report = await evaluation_service.evaluate_run(
            run_id=runtime_result.run_id,
        )

        verification = await read_verification(run_id=runtime_result.run_id)

        if live_report.state_source != "live":
            raise RuntimeError("First evaluation did not use live state.")

        if persisted_report.state_source != "persisted":
            raise RuntimeError("Second evaluation did not use persisted state.")

        if not live_report.passed:
            raise RuntimeError("Live evaluation did not pass.")

        if not persisted_report.passed:
            raise RuntimeError("Persisted evaluation did not pass.")

        if verification["ticket_count"] != 1:
            raise RuntimeError("Expected exactly one matching ticket.")

        if verification["ticket_status"] != "open":
            raise RuntimeError("Created ticket is not open.")

        if verification["ticket_title"] != "数据平台访问问题":
            raise RuntimeError("Created ticket title is wrong.")

        if verification["tool_operation_count"] != 1:
            raise RuntimeError("Expected exactly one create_ticket ToolOperation.")

        if verification["tool_operation_statuses"] != ["succeeded"]:
            raise RuntimeError("create_ticket ToolOperation did not succeed.")

        if not verification["source_operation_matches"]:
            raise RuntimeError("Ticket source_operation_id does not match ToolOperation.")

        if verification["evaluation_count"] != 1:
            raise RuntimeError("Expected exactly one EvaluationResult.")

        if not verification["evaluation_passed"]:
            raise RuntimeError("Persisted EvaluationResult did not pass.")

        if verification["database_state_source"] != "persisted":
            raise RuntimeError(
                "Persisted EvaluationResult does not contain the expected state source."
            )

        output = {
            "runtime": {
                "run_id": (runtime_result.run_id),
                "status": (runtime_result.status),
                "total_steps": (runtime_result.total_steps),
                "total_tool_calls": (runtime_result.total_tool_calls),
                "input_tokens": (runtime_result.input_tokens),
                "output_tokens": (runtime_result.output_tokens),
                "latency_ms": (runtime_result.latency_ms),
            },
            "first_evaluation": (compact_evaluation(live_report)),
            "second_evaluation": (compact_evaluation(persisted_report)),
            "verification": verification,
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        await engine.dispose()


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(
            async_main(),
            loop_factory=(create_windows_selector_event_loop),
        )
    else:
        asyncio.run(async_main())
