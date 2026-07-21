import json
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from app.tools.schemas import ToolDefinition


class AgentDecisionModel(Protocol):
    """Agent Model 节点所需的最小接口。"""

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        """根据消息和工具定义产生下一条 AI 消息."""
        ...


class ScriptedEmployeeLookupModel:
    """
    用于阶段 5A 的确定性离线模型。

    第一次调用 get_employee。
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
