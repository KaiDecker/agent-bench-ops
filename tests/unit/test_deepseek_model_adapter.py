from types import SimpleNamespace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.agent.deepseek import (
    DeepSeekToolCallingModel,
)
from app.tools.implementations.employees import (
    GET_EMPLOYEE_TOOL,
)


def build_tool_call_response() -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id="call_deepseek_001",
        function=SimpleNamespace(
            name="get_employee",
            arguments='{"name":"张三"}',
        ),
    )

    message = SimpleNamespace(
        content="好的，我来查询。",
        tool_calls=[tool_call],
    )

    choice = SimpleNamespace(
        message=message,
        finish_reason="tool_calls",
    )

    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )

    return SimpleNamespace(
        id="response_deepseek_001",
        model="deepseek-v4-flash",
        choices=[choice],
        usage=usage,
        system_fingerprint="fake-fingerprint",
    )


def build_final_response() -> SimpleNamespace:
    message = SimpleNamespace(
        content=("张三的员工编号是 E10001，部门是数据平台部，状态为 active。"),
        tool_calls=None,
    )

    choice = SimpleNamespace(
        message=message,
        finish_reason="stop",
    )

    usage = SimpleNamespace(
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180,
    )

    return SimpleNamespace(
        id="response_deepseek_002",
        model="deepseek-v4-flash",
        choices=[choice],
        usage=usage,
        system_fingerprint="fake-fingerprint",
    )


class FakeCompletions:
    def __init__(
        self,
        responses: list[SimpleNamespace],
    ) -> None:
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(
        self,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.requests.append(kwargs)

        if not self._responses:
            raise RuntimeError("No fake DeepSeek response remains")

        return self._responses.pop(0)


class FakeDeepSeekClient:
    def __init__(
        self,
        responses: list[SimpleNamespace],
    ) -> None:
        self.completions = FakeCompletions(responses)

        self.chat = SimpleNamespace(completions=self.completions)


async def test_deepseek_adapter_returns_tool_call() -> None:
    client = FakeDeepSeekClient([build_tool_call_response()])

    model = DeepSeekToolCallingModel(
        model_name="deepseek-v4-flash",
        api_key=None,
        client=client,
    )

    response = await model.ainvoke(
        messages=[HumanMessage(content="查询张三的员工信息")],
        tools=[
            GET_EMPLOYEE_TOOL,
        ],
    )

    assert isinstance(response, AIMessage)

    assert response.tool_calls == [
        {
            "name": "get_employee",
            "args": {
                "name": "张三",
            },
            "id": "call_deepseek_001",
            "type": "tool_call",
        }
    ]

    assert response.usage_metadata == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }

    request = client.completions.requests[0]

    assert request["model"] == ("deepseek-v4-flash")

    assert request["extra_body"] == {
        "thinking": {
            "type": "disabled",
        }
    }

    assert request["tool_choice"] == "auto"
    assert request["tools"][0]["function"]["name"] == ("get_employee")

    assert request["messages"][0]["role"] == ("system")

    assert request["messages"][1] == {
        "role": "user",
        "content": "查询张三的员工信息",
    }


async def test_deepseek_adapter_serializes_tool_history() -> None:
    client = FakeDeepSeekClient([build_final_response()])

    model = DeepSeekToolCallingModel(
        model_name="deepseek-v4-flash",
        api_key=None,
        client=client,
    )

    previous_ai_message = AIMessage(
        content="好的，我来查询。",
        tool_calls=[
            {
                "name": "get_employee",
                "args": {
                    "name": "张三",
                },
                "id": "call_deepseek_001",
                "type": "tool_call",
            }
        ],
    )

    tool_message = ToolMessage(
        content=('{"status":"succeeded","output":{"employee":{"name":"张三"}}}'),
        tool_call_id="call_deepseek_001",
    )

    response = await model.ainvoke(
        messages=[
            HumanMessage(content="查询张三的员工信息"),
            previous_ai_message,
            tool_message,
        ],
        tools=[
            GET_EMPLOYEE_TOOL,
        ],
    )

    assert response.tool_calls == []

    assert response.content == ("张三的员工编号是 E10001，部门是数据平台部，状态为 active。")

    request_messages = client.completions.requests[0]["messages"]

    assert request_messages[2]["role"] == ("assistant")

    assert request_messages[2]["tool_calls"][0]["id"] == "call_deepseek_001"

    assert request_messages[3] == {
        "role": "tool",
        "tool_call_id": "call_deepseek_001",
        "content": ('{"status":"succeeded","output":{"employee":{"name":"张三"}}}'),
    }
