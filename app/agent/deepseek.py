import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openai import AsyncOpenAI
from pydantic import SecretStr

from app.agent.model import (
    tool_definition_to_openai_schema,
)
from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.tools.schemas import ToolDefinition


def content_to_api_text(content: Any) -> str:
    """将 LangChain 消息内容转换为 API 文本。"""

    if isinstance(content, str):
        return content

    return json.dumps(
        content,
        ensure_ascii=False,
        default=str,
    )


def message_to_deepseek_payload(
    message: BaseMessage,
) -> dict[str, Any]:
    """将 LangChain 消息转换为 DeepSeek Chat 消息。"""

    if isinstance(message, SystemMessage):
        return {
            "role": "system",
            "content": content_to_api_text(message.content),
        }

    if isinstance(message, HumanMessage):
        return {
            "role": "user",
            "content": content_to_api_text(message.content),
        }

    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content_to_api_text(message.content),
        }

    if isinstance(message, AIMessage):
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": content_to_api_text(message.content),
        }

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(
                            tool_call["args"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for tool_call in message.tool_calls
            ]

        return payload

    raise TypeError(f"Unsupported message type for DeepSeek: {type(message).__name__}")


class DeepSeekToolCallingModel:
    """
    DeepSeek OpenAI-compatible Tool Calling 适配器。

    第一版固定关闭 Thinking Mode，避免多轮工具调用时
    额外处理 reasoning_content。
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | SecretStr | None,
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        client: Any | None = None,
    ) -> None:
        normalized_model_name = model_name.strip()
        normalized_base_url = base_url.strip()
        normalized_prompt = system_prompt.strip()

        if not normalized_model_name:
            raise ValueError("model_name cannot be empty")

        if not normalized_base_url:
            raise ValueError("base_url cannot be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if not normalized_prompt:
            raise ValueError("system_prompt cannot be empty")

        self.model_name = normalized_model_name
        self._system_prompt = normalized_prompt

        if client is not None:
            self._client = client
            return

        if api_key is None:
            raise ValueError("api_key is required when client is not provided")

        resolved_api_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key

        if not resolved_api_key.strip():
            raise ValueError("api_key cannot be empty")

        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=normalized_base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    async def ainvoke(
        self,
        *,
        messages: Sequence[BaseMessage],
        tools: Sequence[ToolDefinition],
    ) -> AIMessage:
        """调用 DeepSeek 并转换为标准 AIMessage。"""

        api_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._system_prompt,
            }
        ]

        api_messages.extend(message_to_deepseek_payload(message) for message in messages)

        request_arguments: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "extra_body": {
                "thinking": {
                    "type": "disabled",
                }
            },
        }

        if tools:
            request_arguments["tools"] = [
                tool_definition_to_openai_schema(definition) for definition in tools
            ]

            request_arguments["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**request_arguments)

        if not response.choices:
            raise RuntimeError("DeepSeek returned no completion choices")

        choice = response.choices[0]
        response_message = choice.message

        parsed_tool_calls: list[dict[str, Any]] = []

        for tool_call in response_message.tool_calls or []:
            if not tool_call.id:
                raise ValueError("DeepSeek returned a tool call without an ID")

            function_name = tool_call.function.name

            if not function_name:
                raise ValueError("DeepSeek returned a tool call without a function name")

            try:
                parsed_arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("DeepSeek returned invalid JSON tool arguments") from exc

            if not isinstance(parsed_arguments, dict):
                raise ValueError("DeepSeek tool arguments must be a JSON object")

            parsed_tool_calls.append(
                {
                    "name": function_name,
                    "args": parsed_arguments,
                    "id": tool_call.id,
                    "type": "tool_call",
                }
            )

        response_metadata = {
            "provider": "deepseek",
            "model_name": response.model,
            "response_id": response.id,
            "finish_reason": choice.finish_reason,
            "system_fingerprint": getattr(
                response,
                "system_fingerprint",
                None,
            ),
        }

        message_arguments: dict[str, Any] = {
            "content": response_message.content or "",
            "tool_calls": parsed_tool_calls,
            "response_metadata": response_metadata,
        }

        if response.usage is not None:
            message_arguments["usage_metadata"] = {
                "input_tokens": (response.usage.prompt_tokens),
                "output_tokens": (response.usage.completion_tokens),
                "total_tokens": (response.usage.total_tokens),
            }

        return AIMessage(**message_arguments)


__all__ = [
    "DeepSeekToolCallingModel",
    "message_to_deepseek_payload",
]
