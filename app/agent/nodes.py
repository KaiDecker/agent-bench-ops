import hashlib
import json
from collections.abc import (
    Callable,
    Sequence,
)
from contextlib import AbstractAsyncContextManager
from time import perf_counter
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.model import AgentDecisionModel
from app.agent.recorder import (
    RunStepRecorderProtocol,
    StartedRunStep,
)
from app.agent.serialization import (
    extract_usage,
    serialize_message,
    serialize_messages,
)
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
        """执行一个工具调用。"""
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


async def start_persisted_step(
    *,
    recorder: RunStepRecorderProtocol | None,
    state: AgentState,
    step_number: int,
    step_type: str,
    model_name: str | None,
    tool_name: str | None,
    input_payload: dict[str, Any],
    parent_step_id: str | None,
) -> StartedRunStep | None:
    """存在 run_id 和 recorder 时创建持久化步骤。"""

    if recorder is None or state["run_id"] is None:
        return None

    return await recorder.start_step(
        run_id=state["run_id"],
        parent_step_id=parent_step_id,
        step_number=step_number,
        step_type=step_type,  # type: ignore[arg-type]
        model_name=model_name,
        tool_name=tool_name,
        input_payload=input_payload,
    )


def create_model_node(
    *,
    model: AgentDecisionModel,
    registry: ToolRegistry,
    recorder: RunStepRecorderProtocol | None = None,
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

        run_step_number = state["run_step_count"] + 1

        started_at = perf_counter()
        persisted_step: StartedRunStep | None = None

        try:
            persisted_step = await start_persisted_step(
                recorder=recorder,
                state=state,
                step_number=run_step_number,
                step_type="model",
                model_name=getattr(
                    model,
                    "model_name",
                    type(model).__name__,
                ),
                tool_name=None,
                input_payload={
                    "messages": serialize_messages(state["messages"]),
                    "available_tools": [
                        definition.metadata.name for definition in tool_definitions
                    ],
                },
                parent_step_id=(state["last_run_step_id"]),
            )

            response = await model.ainvoke(
                messages=state["messages"],
                tools=tool_definitions,
            )

            if not isinstance(response, AIMessage):
                raise TypeError("The model did not return an AIMessage")

        except Exception as exc:
            latency_ms = round(
                (perf_counter() - started_at) * 1000,
                2,
            )

            if recorder is not None and persisted_step is not None:
                await recorder.finish_step(
                    step_id=persisted_step.id,
                    status="failed",
                    output_payload=None,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            error = build_agent_error(
                code="agent_model_error",
                message=("The model failed while deciding the next action."),
                details={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

            return {
                "step_count": state["step_count"] + 1,
                "run_step_count": run_step_number,
                "last_run_step_id": (
                    persisted_step.id if persisted_step is not None else state["last_run_step_id"]
                ),
                "error": error,
                "final_response": error["message"],
            }

        latency_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        (
            input_tokens,
            output_tokens,
            _total_tokens,
        ) = extract_usage(response)

        if recorder is not None and persisted_step is not None:
            await recorder.finish_step(
                step_id=persisted_step.id,
                status="succeeded",
                output_payload={
                    "message": serialize_message(response),
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error_type=None,
                error_message=None,
            )

        final_response = None

        if not response.tool_calls:
            final_response = content_to_text(response.content)

        return {
            "messages": [response],
            "step_count": state["step_count"] + 1,
            "run_step_count": run_step_number,
            "last_run_step_id": (
                persisted_step.id if persisted_step is not None else state["last_run_step_id"]
            ),
            "final_response": final_response,
        }

    return model_node


def create_tool_node(
    *,
    gateway: ToolExecutor,
    session_factory: SessionFactory = (AsyncSessionFactory),
    recorder: RunStepRecorderProtocol | None = None,
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
        current_run_step_count = state["run_step_count"]
        current_last_step_id = state["last_run_step_id"]

        # 所有工具调用的直接父步骤都是产生这些调用的 Model 步骤。
        model_parent_step_id = state["last_run_step_id"]

        for tool_call in tool_calls:
            current_run_step_count += 1

            tool_name = tool_call["name"]
            arguments = tool_call["args"]
            tool_call_id = tool_call.get("id") or "missing_tool_call_id"

            call_digest = hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()

            idempotency_key = f"agent-call:{call_digest}"

            persisted_step: StartedRunStep | None = None
            started_at = perf_counter()

            try:
                persisted_step = await start_persisted_step(
                    recorder=recorder,
                    state=state,
                    step_number=current_run_step_count,
                    step_type="tool",
                    model_name=None,
                    tool_name=tool_name,
                    input_payload={
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "idempotency_key": (idempotency_key),
                    },
                    parent_step_id=model_parent_step_id,
                )

                execution_context = ToolExecutionContext(
                    run_id=state["run_id"],
                    step_id=(persisted_step.id if persisted_step is not None else None),
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
                        idempotency_key=(idempotency_key),
                    )

            except Exception as exc:
                latency_ms = round(
                    (perf_counter() - started_at) * 1000,
                    2,
                )

                if recorder is not None and persisted_step is not None:
                    await recorder.finish_step(
                        step_id=persisted_step.id,
                        status="failed",
                        output_payload=None,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

                error = build_agent_error(
                    code="agent_tool_execution_error",
                    message=("The tool node failed while executing a tool."),
                    details={
                        "tool_name": tool_name,
                        "exception_type": (type(exc).__name__),
                        "exception_message": str(exc),
                    },
                )

                return {
                    "run_step_count": (current_run_step_count),
                    "last_run_step_id": (
                        persisted_step.id if persisted_step is not None else current_last_step_id
                    ),
                    "error": error,
                    "final_response": error["message"],
                }

            latency_ms = round(
                (perf_counter() - started_at) * 1000,
                2,
            )

            if recorder is not None and persisted_step is not None:
                await recorder.finish_step(
                    step_id=persisted_step.id,
                    status=("succeeded" if response.status == "succeeded" else "failed"),
                    output_payload={"response": response.model_dump(mode="json")},
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error_type=(response.error.code if response.error is not None else None),
                    error_message=(response.error.message if response.error is not None else None),
                )

            if persisted_step is not None:
                current_last_step_id = persisted_step.id

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
            "run_step_count": current_run_step_count,
            "last_run_step_id": current_last_step_id,
        }

    return tool_node
