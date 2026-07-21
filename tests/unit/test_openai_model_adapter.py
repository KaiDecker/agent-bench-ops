from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.agent.model import (
    OpenAIToolCallingModel,
    tool_definition_to_openai_schema,
)
from app.tools.implementations.employees import (
    GET_EMPLOYEE_TOOL,
)


class FakeBoundModel:
    """模拟 bind_tools 后的模型。"""

    def __init__(self) -> None:
        self.messages: Sequence[BaseMessage] | None = None

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> AIMessage:
        self.messages = messages

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_employee",
                    "args": {
                        "name": "张三",
                    },
                    "id": "fake_call_001",
                    "type": "tool_call",
                }
            ],
        )


class FakeChatModel:
    """模拟 ChatOpenAI，不访问网络。"""

    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] | None = None
        self.bind_kwargs: dict[str, Any] | None = None
        self.bound_model = FakeBoundModel()

    def bind_tools(
        self,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> FakeBoundModel:
        self.tools = tools
        self.bind_kwargs = kwargs
        return self.bound_model


def test_tool_schema_preserves_internal_metadata() -> None:
    schema = tool_definition_to_openai_schema(GET_EMPLOYEE_TOOL)

    assert schema["type"] == "function"

    function = schema["function"]

    assert function["name"] == "get_employee"
    assert function["description"] == (GET_EMPLOYEE_TOOL.metadata.description)

    parameters = function["parameters"]

    assert parameters["type"] == "object"
    assert "employee_id" in parameters["properties"]
    assert "employee_no" in parameters["properties"]
    assert "name" in parameters["properties"]


async def test_openai_adapter_binds_tools_and_prompt() -> None:
    fake_model = FakeChatModel()

    adapter = OpenAIToolCallingModel(
        model_name="fake-openai-model",
        api_key=None,
        chat_model=fake_model,
    )

    response = await adapter.ainvoke(
        messages=[HumanMessage(content="查询张三的员工信息")],
        tools=[
            GET_EMPLOYEE_TOOL,
        ],
    )

    assert response.tool_calls[0]["name"] == ("get_employee")

    assert fake_model.tools is not None
    assert fake_model.tools[0]["function"]["name"] == ("get_employee")

    assert fake_model.bind_kwargs == {
        "tool_choice": "auto",
        "strict": False,
        "parallel_tool_calls": False,
    }

    messages = fake_model.bound_model.messages

    assert messages is not None
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
