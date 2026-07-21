import asyncio
import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.runtime import AgentRuntime
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry
from app.tools.schemas import ToolDefinition


class FailingModel:
    model_name = "intentional-failure-model"

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        raise RuntimeError("Intentional model failure for runtime testing")


async def async_main() -> None:
    registry = build_default_registry()
    gateway = ToolGateway(registry)

    runtime = AgentRuntime(
        model=FailingModel(),
        model_provider="test",
        model_name="intentional-failure-model",
        registry=registry,
        gateway=gateway,
    )

    result = await runtime.run_benchmark_task(
        task_key="employee_lookup_001",
        task_version=1,
        actor_id="benchmark-agent",
        permissions=["employee.read"],
        prompt_version="stage5c-failure-v1",
        configuration={
            "stage": "5C",
            "test_case": "model_failure",
        },
    )

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    asyncio.run(async_main())
