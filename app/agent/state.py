from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages


class AgentError(TypedDict):
    """Agent 执行期间的结构化错误。"""

    code: str
    message: str
    details: dict[str, Any]


class AgentState(TypedDict):
    """LangGraph 节点之间共享的运行状态。"""

    messages: Annotated[list[BaseMessage], add_messages]

    run_id: str | None
    actor_id: str

    available_tools: list[str]
    permissions: list[str]

    step_count: int
    tool_call_count: int

    max_steps: int
    max_tool_calls: int

    final_response: str | None
    error: AgentError | None


def build_initial_state(
    *,
    user_request: str,
    run_id: str | None,
    actor_id: str,
    available_tools: list[str],
    permissions: list[str],
    max_steps: int,
    max_tool_calls: int,
) -> AgentState:
    """创建一次 Agent 运行的初始状态。"""

    normalized_request = user_request.strip()

    if not normalized_request:
        raise ValueError("user_request cannot be empty")

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    if max_tool_calls < 0:
        raise ValueError("max_tool_calls cannot be negative")

    return AgentState(
        messages=[
            HumanMessage(content=normalized_request),
        ],
        run_id=run_id,
        actor_id=actor_id,
        available_tools=sorted(set(available_tools)),
        permissions=sorted(set(permissions)),
        step_count=0,
        tool_call_count=0,
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        final_response=None,
        error=None,
    )
