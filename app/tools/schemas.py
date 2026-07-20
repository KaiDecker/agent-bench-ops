from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

type ToolExecutionStatus = Literal[
    "succeeded",
    "failed",
    "rejected",
    "timed_out",
]

type ToolErrorStatus = Literal[
    "failed",
    "rejected",
    "timed_out",
]


class ToolMetadata(BaseModel):
    """工具的静态安全与执行元数据。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)

    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] = "low"

    required_permissions: set[str] = Field(default_factory=set)
    requires_approval: bool = False
    is_idempotent: bool = True
    read_only: bool = False
    timeout_seconds: float = Field(default=5.0, gt=0)


class ToolExecutionContext(BaseModel):
    """单次工具调用的授权上下文。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    step_id: str | None = None
    actor_id: str = "benchmark-agent"

    available_tools: set[str] = Field(default_factory=set)
    permissions: set[str] = Field(default_factory=set)


class ToolError(BaseModel):
    """工具调用的结构化错误。"""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResponse(BaseModel):
    """Tool Gateway 的统一返回结构。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: ToolExecutionStatus

    operation_id: str | None = None
    replayed: bool = False

    output: dict[str, Any] | None = None
    error: ToolError | None = None
    latency_ms: float


class ToolBusinessError(Exception):
    """工具实现主动返回的可预期业务错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.details = details or {}


type ToolHandler = Callable[
    [
        AsyncSession,
        BaseModel,
        ToolExecutionContext,
    ],
    Awaitable[BaseModel],
]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """一个已注册工具的完整定义。"""

    metadata: ToolMetadata
    arguments_model: type[BaseModel]
    result_model: type[BaseModel]
    handler: ToolHandler
