from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
)
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.model import AgentDecisionModel
from app.agent.nodes import (
    ToolExecutor,
    create_model_node,
    create_tool_node,
)
from app.agent.recorder import (
    RunStepRecorderProtocol,
)
from app.agent.state import AgentState
from app.persistence.database import AsyncSessionFactory
from app.tools.registry import ToolRegistry

type SessionFactory = Callable[
    [],
    AbstractAsyncContextManager[AsyncSession],
]


def route_after_model(
    state: AgentState,
) -> Literal["tools", "end"]:
    """Model 请求工具时进入 tools，否则结束。"""

    if state["error"] is not None:
        return "end"

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"


def route_after_tools(
    state: AgentState,
) -> Literal["model", "end"]:
    """工具节点成功后返回模型，错误时结束。"""

    if state["error"] is not None:
        return "end"

    return "model"


def build_agent_graph(
    *,
    model: AgentDecisionModel,
    registry: ToolRegistry,
    gateway: ToolExecutor,
    session_factory: SessionFactory = (AsyncSessionFactory),
    recorder: RunStepRecorderProtocol | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """构建并编译 Tool-Calling Agent 图。"""

    builder = StateGraph(AgentState)

    builder.add_node(
        "model",
        create_model_node(
            model=model,
            registry=registry,
            recorder=recorder,
        ),
    )

    builder.add_node(
        "tools",
        create_tool_node(
            gateway=gateway,
            session_factory=session_factory,
            recorder=recorder,
        ),
    )

    builder.add_edge(START, "model")

    builder.add_conditional_edges(
        "model",
        route_after_model,
        {
            "tools": "tools",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "model": "model",
            "end": END,
        },
    )

    return builder.compile(checkpointer=checkpointer)
