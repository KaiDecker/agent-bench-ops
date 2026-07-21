import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)


def json_safe(value: Any) -> Any:
    """将任意值转换成可写入 JSONB 的结构。"""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    )


def serialize_message(
    message: BaseMessage,
) -> dict[str, Any]:
    """将 LangChain 消息转换成稳定的 JSON 对象。"""

    result: dict[str, Any] = {
        "type": message.type,
        "content": json_safe(message.content),
    }

    if message.id is not None:
        result["id"] = message.id

    if message.name is not None:
        result["name"] = message.name

    if isinstance(message, AIMessage):
        result["tool_calls"] = json_safe(message.tool_calls)

        if message.usage_metadata is not None:
            result["usage_metadata"] = json_safe(message.usage_metadata)

        if message.response_metadata:
            result["response_metadata"] = json_safe(message.response_metadata)

    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id

        if message.status is not None:
            result["status"] = message.status

    return result


def serialize_messages(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """序列化一组消息。"""

    return [serialize_message(message) for message in messages]


def extract_usage(
    message: AIMessage,
) -> tuple[int, int, int]:
    """从 AIMessage 提取标准 Token 统计。"""

    usage = message.usage_metadata or {}

    input_tokens = int(usage.get("input_tokens", 0) or 0)

    output_tokens = int(usage.get("output_tokens", 0) or 0)

    reported_total = int(usage.get("total_tokens", 0) or 0)

    total_tokens = reported_total if reported_total > 0 else input_tokens + output_tokens

    return (
        input_tokens,
        output_tokens,
        total_tokens,
    )
