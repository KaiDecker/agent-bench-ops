import asyncio
import json
import os
import sys
from typing import Any

from openai import AsyncOpenAI


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

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        max_retries=1,
    )

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_employee",
                "description": (
                    "Query one employee by employee ID, employee number, or employee name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {
                            "type": "string",
                        },
                        "employee_no": {
                            "type": "string",
                        },
                        "name": {
                            "type": "string",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是企业员工信息查询助手。"
                    "涉及员工事实时必须调用工具，"
                    "不得猜测。每轮最多调用一个工具。"
                ),
            },
            {
                "role": "user",
                "content": ("请查询员工张三的员工编号、部门和当前状态。"),
            },
        ],
        tools=tools,
        tool_choice="auto",
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )

    if not response.choices:
        raise RuntimeError("DeepSeek returned no completion choices.")

    message = response.choices[0].message

    tool_calls = []

    for tool_call in message.tool_calls or []:
        try:
            parsed_arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            parsed_arguments = {
                "_raw": tool_call.function.arguments,
            }

        tool_calls.append(
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": parsed_arguments,
            }
        )

    usage = None

    if response.usage is not None:
        usage = {
            "prompt_tokens": (response.usage.prompt_tokens),
            "completion_tokens": (response.usage.completion_tokens),
            "total_tokens": (response.usage.total_tokens),
        }

    output = {
        "model": response.model,
        "finish_reason": (response.choices[0].finish_reason),
        "content": message.content,
        "tool_calls": tool_calls,
        "usage": usage,
    }

    print(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not tool_calls:
        print(
            "\nERROR: The model did not request a tool call.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    first_call = tool_calls[0]

    if first_call["name"] != "get_employee":
        print(
            f"\nERROR: Unexpected tool name: {first_call['name']}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(async_main())
