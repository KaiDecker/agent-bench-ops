import asyncio
import json
import os

from app.agent.deepseek import (
    DeepSeekToolCallingModel,
)
from app.agent.runtime import AgentRuntime
from app.tools.gateway import ToolGateway
from app.tools.registry import build_default_registry

TASK_KEY = "employee_lookup_001"
TASK_VERSION = 1


def require_environment_variable(name: str) -> str:
    value = os.environ.get(name)

    if value is None or not value.strip():
        raise RuntimeError(f"{name} is not set in the current environment.")

    return value.strip()


async def async_main() -> None:
    api_key = require_environment_variable("DEEPSEEK_API_KEY")

    base_url = os.environ.get(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ).strip()

    model_name = os.environ.get(
        "DEEPSEEK_MODEL",
        "deepseek-v4-flash",
    ).strip()

    registry = build_default_registry()
    gateway = ToolGateway(registry)

    model = DeepSeekToolCallingModel(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=60.0,
        max_retries=2,
    )

    runtime = AgentRuntime(
        model=model,
        model_provider="deepseek",
        model_name=model_name,
        registry=registry,
        gateway=gateway,
    )

    result = await runtime.run_benchmark_task(
        task_key=TASK_KEY,
        task_version=TASK_VERSION,
        actor_id="benchmark-agent",
        permissions=[
            "employee.read",
        ],
        prompt_version="stage5c-runtime-v1",
        agent_strategy=("langgraph-model-tool-loop"),
        memory_strategy="messages-state",
        configuration={
            "stage": "5C",
            "provider": "deepseek",
            "thinking_mode": "disabled",
            "parallel_tool_calls": False,
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
