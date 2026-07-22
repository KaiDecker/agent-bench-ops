import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.tools.schemas import ToolDefinition


class AgentDecisionModel(Protocol):
    """Agent Model 节点使用的统一模型接口。"""

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        """根据消息历史和可用工具决定下一步。"""
        ...


def tool_definition_to_openai_schema(
    definition: ToolDefinition,
) -> dict[str, Any]:
    """
    将内部 ToolDefinition 转换为 OpenAI function tool Schema。

    这里只向模型提供名称、描述和参数 Schema。
    真正执行仍由 ToolGateway 负责。
    """

    converted = deepcopy(
        convert_to_openai_tool(
            definition.arguments_model,
            strict=False,
        )
    )

    function_schema = converted.get("function")

    if not isinstance(function_schema, dict):
        raise ValueError("Converted tool schema has no function definition")

    function_schema["name"] = definition.metadata.name
    function_schema["description"] = definition.metadata.description

    return converted


class OpenAIToolCallingModel:
    """使用 ChatOpenAI 的真实 Tool-Calling 模型适配器。"""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | SecretStr | None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        chat_model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if chat_model is None and api_key is None:
            raise ValueError("api_key is required when chat_model is not provided")

        self.model_name = model_name
        self._system_prompt = system_prompt.strip()

        if chat_model is not None:
            self._chat_model = chat_model
        else:
            self._chat_model = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        """调用真实模型，并返回标准 AIMessage。"""

        input_messages: list[BaseMessage] = [
            SystemMessage(content=self._system_prompt),
            *messages,
        ]

        tool_schemas = [tool_definition_to_openai_schema(definition) for definition in tools]

        if tool_schemas:
            runnable = self._chat_model.bind_tools(
                tool_schemas,
                tool_choice="auto",
                strict=False,
                parallel_tool_calls=False,
            )
        else:
            runnable = self._chat_model

        response = await runnable.ainvoke(input_messages)

        if not isinstance(response, AIMessage):
            raise TypeError("OpenAI chat model did not return AIMessage")

        return response


class ScriptedEmployeeLookupModel:
    """
    阶段 5A 使用的确定性离线模型。

    保留它用于无网络测试和回归测试。
    """

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        tool_names = {tool.metadata.name for tool in tools}

        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]

        if not tool_messages:
            if "get_employee" not in tool_names:
                return AIMessage(content=("当前任务没有提供 get_employee 工具，无法查询员工信息。"))

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_employee",
                        "args": {
                            "name": "张三",
                        },
                        "id": "call_get_employee_001",
                        "type": "tool_call",
                    }
                ],
            )

        latest_tool_message = tool_messages[-1]

        if not isinstance(
            latest_tool_message.content,
            str,
        ):
            return AIMessage(content="工具返回了无法解析的结果。")

        try:
            payload = json.loads(latest_tool_message.content)
        except json.JSONDecodeError:
            return AIMessage(content="工具返回的结果不是有效 JSON。")

        if payload.get("status") != "succeeded":
            error = payload.get("error") or {}
            error_message = error.get(
                "message",
                "未知工具错误",
            )

            return AIMessage(content=(f"员工信息查询失败：{error_message}"))

        output = payload.get("output")

        if not isinstance(output, dict):
            return AIMessage(content="工具成功，但没有返回员工数据。")

        employee = output.get("employee")

        if not isinstance(employee, dict):
            return AIMessage(content="工具成功，但员工数据结构不正确。")

        employee_no = employee.get("employee_no")
        department = employee.get("department")
        status = employee.get("status")

        return AIMessage(
            content=(f"张三的员工号是 {employee_no}，部门是 {department}，当前状态是 {status}。")
        )


class ScriptedCreateTicketModel:
    """
    用于 create_ticket_001 的确定性离线模型。

    首次调用直接生成 create_ticket 工具调用；
    收到工具结果后生成最终回答。
    """

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        tool_names = {tool.metadata.name for tool in tools}

        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]

        if not tool_messages:
            if "create_ticket" not in tool_names:
                return AIMessage(content=("当前任务没有提供 create_ticket 工具，无法创建工单。"))

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "requester_employee_id": ("emp_001"),
                            "target_employee_id": ("emp_002"),
                            "ticket_type": "general",
                            "risk_level": "medium",
                            "title": ("数据平台访问问题"),
                            "description": ("李四无法访问数据平台，请协助检查账号权限"),
                        },
                        "id": ("call_create_ticket_001"),
                        "type": "tool_call",
                    }
                ],
            )

        latest_tool_message = tool_messages[-1]

        if not isinstance(
            latest_tool_message.content,
            str,
        ):
            return AIMessage(content="工具返回了无法解析的结果。")

        try:
            payload = json.loads(latest_tool_message.content)
        except json.JSONDecodeError:
            return AIMessage(content=("工具返回的结果不是有效 JSON。"))

        if payload.get("status") != "succeeded":
            error = payload.get("error") or {}
            error_message = error.get(
                "message",
                "未知工具错误",
            )

            return AIMessage(content=(f"工单创建失败：{error_message}"))

        output = payload.get("output")

        if not isinstance(output, dict):
            return AIMessage(content=("工具成功，但没有返回工单数据。"))

        ticket = output.get("ticket")

        if not isinstance(ticket, dict):
            return AIMessage(content=("工具成功，但工单数据结构不正确。"))

        ticket_id = ticket.get("id")
        status = ticket.get("status")
        title = ticket.get("title")

        return AIMessage(
            content=(f"工单 {ticket_id} 已创建，标题为“{title}”，当前状态为 {status}。")
        )
