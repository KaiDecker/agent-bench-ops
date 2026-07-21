import hashlib
import json
from collections.abc import (
    Callable,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.model import AgentDecisionModel
from app.agent.state import AgentError, AgentState
from app.persistence.database import AsyncSessionFactory
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    ToolExecutionContext,
    ToolExecutionResponse,
)


class ToolExecutor(Protocol):
    """ToolGateway 所实现的最小执行接口。"""

    async def execute(
        self,
        session: AsyncSession,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        idempotency_key: str | None = None,
    ) -> ToolExecutionResponse:
        """执行一个工具调用."""
        ...


type SessionFactory = Callable[
    [],
    AbstractAsyncContextManager[AsyncSession],
]


def build_agent_error(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> AgentError:
    return AgentError(
        code=code,
        message=message,
        details=details or {},
    )


def content_to_text(content: Any) -> str:
    """把 LangChain 消息内容转换成最终文本。"""

    if isinstance(content, str):
        return content

    return json.dumps(
        content,
        ensure_ascii=False,
        default=str,
    )


def resolve_tool_definitions(
    registry: ToolRegistry,
    available_tools: Sequence[str],
):
    """只向模型暴露当前任务允许的工具定义。"""

    definitions = []

    for tool_name in available_tools:
        definition = registry.get(tool_name)

        if definition is not None:
            definitions.append(definition)

    return definitions


def create_model_node(
    *,
    model: AgentDecisionModel,
    registry: ToolRegistry,
):
    """创建 LangGraph Model 节点。"""

    async def model_node(
        state: AgentState,
    ) -> dict[str, Any]:
        if state["step_count"] >= state["max_steps"]:
            error = build_agent_error(
                code="agent_step_budget_exceeded",
                message=("The agent exceeded its maximum number of model steps."),
                details={
                    "step_count": state["step_count"],
                    "max_steps": state["max_steps"],
                },
            )

            return {
                "error": error,
                "final_response": error["message"],
            }

        tool_definitions = resolve_tool_definitions(
            registry,
            state["available_tools"],
        )

        try:
            response = await model.ainvoke(
                messages=state["messages"],
                tools=tool_definitions,
            )
        except Exception:
            error = build_agent_error(
                code="agent_model_error",
                message=("The model failed while deciding the next action."),
            )

            return {
                "step_count": state["step_count"] + 1,
                "error": error,
                "final_response": error["message"],
            }

        if not isinstance(response, AIMessage):
            error = build_agent_error(
                code="invalid_model_response",
                message=("The model did not return an AIMessage."),
            )

            return {
                "step_count": state["step_count"] + 1,
                "error": error,
                "final_response": error["message"],
            }

        final_response = None

        if not response.tool_calls:
            final_response = content_to_text(response.content)

        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
            "final_response": final_response,
        }

    return model_node


def create_tool_node(
    *,
    gateway: ToolExecutor,
    session_factory: SessionFactory = (AsyncSessionFactory),
):
    """创建经过 ToolGateway 执行工具的节点。"""

    async def tool_node(
        state: AgentState,
    ) -> dict[str, Any]:
        last_message: BaseMessage = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            error = build_agent_error(
                code="invalid_tool_node_input",
                message=("The tool node expected an AIMessage."),
            )

            return {
                "error": error,
                "final_response": error["message"],
            }

        tool_calls = last_message.tool_calls

        if not tool_calls:
            return {}

        projected_count = state["tool_call_count"] + len(tool_calls)

        if projected_count > state["max_tool_calls"]:
            error = build_agent_error(
                code="agent_tool_budget_exceeded",
                message=("The agent exceeded its maximum number of tool calls."),
                details={
                    "tool_call_count": (state["tool_call_count"]),
                    "requested_calls": len(tool_calls),
                    "max_tool_calls": (state["max_tool_calls"]),
                },
            )

            budget_messages = [
                ToolMessage(
                    content=json.dumps(
                        {
                            "tool_name": tool_call["name"],
                            "status": "rejected",
                            "output": None,
                            "error": error,
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=(tool_call.get("id") or "missing_tool_call_id"),
                )
                for tool_call in tool_calls
            ]

            return {
                "messages": budget_messages,
                "error": error,
                "final_response": error["message"],
            }

        result_messages: list[ToolMessage] = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["args"]
            tool_call_id = tool_call.get("id") or "missing_tool_call_id"

            call_digest = hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()

            execution_context = ToolExecutionContext(
                run_id=state["run_id"],
                actor_id=state["actor_id"],
                available_tools=set(state["available_tools"]),
                permissions=set(state["permissions"]),
            )

            async with session_factory() as session:
                response = await gateway.execute(
                    session=session,
                    tool_name=tool_name,
                    arguments=arguments,
                    context=execution_context,
                    idempotency_key=(f"agent-call:{call_digest}"),
                )

            result_messages.append(
                ToolMessage(
                    content=json.dumps(
                        response.model_dump(mode="json"),
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call_id,
                )
            )

        return {
            "messages": result_messages,
            "tool_call_count": projected_count,
        }

    return tool_node
