import asyncio
from time import perf_counter
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import AsyncSessionFactory
from app.tools.ledger import (
    OperationClaim,
    OperationLedger,
    OperationRecord,
)
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    ToolBusinessError,
    ToolDefinition,
    ToolError,
    ToolErrorStatus,
    ToolExecutionContext,
    ToolExecutionResponse,
)


class ToolGateway:
    """
    工具调用统一入口。

    负责：
    1. 工具注册检查
    2. Benchmark 工具范围检查
    3. 执行身份权限检查
    4. 参数与结果校验
    5. 工具执行超时
    6. 业务事务提交
    7. 操作账本持久化
    8. 幂等调用回放
    """

    def __init__(
        self,
        registry: ToolRegistry,
        ledger: OperationLedger | None = None,
    ) -> None:
        self._registry = registry
        self._ledger = ledger or OperationLedger(AsyncSessionFactory)

    async def execute(
        self,
        session: AsyncSession,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        idempotency_key: str | None = None,
    ) -> ToolExecutionResponse:
        """执行一次受控工具调用。"""

        started_at = perf_counter()
        definition = self._registry.get(tool_name)

        claim: OperationClaim | None = None

        # 只有存在 run_id 时，才写入工具操作账本。
        if context.run_id is not None:
            try:
                claim = await self._ledger.claim(
                    run_id=context.run_id,
                    step_id=context.step_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    idempotency_key=idempotency_key,
                    risk_level=(
                        definition.metadata.risk_level if definition is not None else "low"
                    ),
                    requires_approval=(
                        definition.metadata.requires_approval if definition is not None else False
                    ),
                    is_idempotent=(
                        definition.metadata.is_idempotent if definition is not None else False
                    ),
                )
            except Exception:
                return self._error_response(
                    tool_name=tool_name,
                    status="failed",
                    started_at=started_at,
                    code="operation_ledger_error",
                    message=("The tool operation could not be registered."),
                )

            # 相同 run_id 和 idempotency_key 已经存在。
            if not claim.created:
                existing = claim.record

                if (
                    existing.tool_name != tool_name
                    or existing.arguments_hash != claim.requested_arguments_hash
                ):
                    return ToolExecutionResponse(
                        tool_name=tool_name,
                        status="rejected",
                        operation_id=existing.operation_id,
                        replayed=True,
                        error=ToolError(
                            code="idempotency_key_conflict",
                            message=(
                                "The idempotency key was already "
                                "used with different arguments "
                                "or a different tool."
                            ),
                        ),
                        latency_ms=self._latency_ms(started_at),
                    )

                return self._response_from_existing(existing)

        operation = claim.record if claim is not None else None

        # 未注册工具。
        if definition is None:
            response = self._error_response(
                tool_name=tool_name,
                status="rejected",
                started_at=started_at,
                code="tool_not_registered",
                message=f"Tool is not registered: {tool_name}",
            )

            return await self._complete(
                response=response,
                operation=operation,
                definition=None,
            )

        # 工具不在当前 Benchmark 允许范围内。
        if tool_name not in context.available_tools:
            response = self._error_response(
                tool_name=tool_name,
                status="rejected",
                started_at=started_at,
                code="tool_not_allowed",
                message=(f"The current benchmark task does not allow tool: {tool_name}"),
            )

            return await self._complete(
                response=response,
                operation=operation,
                definition=definition,
            )

        # 权限检查。
        missing_permissions = definition.metadata.required_permissions - context.permissions

        if missing_permissions:
            response = self._error_response(
                tool_name=tool_name,
                status="rejected",
                started_at=started_at,
                code="permission_denied",
                message=("The execution identity lacks required permissions."),
                details={"missing_permissions": sorted(missing_permissions)},
            )

            return await self._complete(
                response=response,
                operation=operation,
                definition=definition,
            )

        # 工具参数校验。
        try:
            validated_arguments = definition.arguments_model.model_validate(arguments)
        except ValidationError as exc:
            response = self._error_response(
                tool_name=tool_name,
                status="failed",
                started_at=started_at,
                code="invalid_arguments",
                message=("Tool arguments failed schema validation."),
                details={
                    "errors": exc.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    )
                },
            )

            return await self._complete(
                response=response,
                operation=operation,
                definition=definition,
            )

        # 将账本状态更新成 running。
        if operation is not None:
            try:
                await self._ledger.mark_running(operation.database_id)
            except Exception:
                return ToolExecutionResponse(
                    tool_name=tool_name,
                    status="failed",
                    operation_id=operation.operation_id,
                    error=ToolError(
                        code="operation_ledger_update_error",
                        message=("The operation could not be marked as running."),
                    ),
                    latency_ms=self._latency_ms(started_at),
                )

        # operation_id 只能由 Gateway 注入，不能由模型提供。
        execution_context = context

        if operation is not None:
            execution_context = context.model_copy(
                update={
                    "operation_id": operation.operation_id,
                }
            )

        try:
            async with asyncio.timeout(definition.metadata.timeout_seconds):
                # 工具业务副作用位于一个独立事务中。
                async with session.begin():
                    raw_result = await definition.handler(
                        session,
                        validated_arguments,
                        execution_context,
                    )

                    # 结果结构不正确时，事务也会回滚。
                    validated_result = definition.result_model.model_validate(raw_result)

            # 离开 session.begin() 后，业务事务已经提交。
            # 此时注入响应丢失，模拟：
            # 数据库修改成功，但调用方没有收到明确结果。
            if context.fault_injection == "drop_response_after_commit":
                raise TimeoutError("Injected response loss after transaction commit")

        except TimeoutError:
            response = self._error_response(
                tool_name=tool_name,
                status="timed_out",
                started_at=started_at,
                code="tool_timeout",
                message=(
                    "Tool execution completed or timed out, "
                    "but no definitive response was received."
                ),
            )

        except ToolBusinessError as exc:
            response = self._error_response(
                tool_name=tool_name,
                status="failed",
                started_at=started_at,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )

        except ValidationError as exc:
            response = self._error_response(
                tool_name=tool_name,
                status="failed",
                started_at=started_at,
                code="invalid_tool_result",
                message=("Tool implementation returned an invalid result structure."),
                details={
                    "errors": exc.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    )
                },
            )

        except Exception:
            response = self._error_response(
                tool_name=tool_name,
                status="failed",
                started_at=started_at,
                code="tool_execution_error",
                message=("An unexpected tool execution error occurred."),
            )

        else:
            response = ToolExecutionResponse(
                tool_name=tool_name,
                status="succeeded",
                output=validated_result.model_dump(mode="json"),
                latency_ms=self._latency_ms(started_at),
            )

        return await self._complete(
            response=response,
            operation=operation,
            definition=definition,
        )

    async def _complete(
        self,
        *,
        response: ToolExecutionResponse,
        operation: OperationRecord | None,
        definition: ToolDefinition | None,
    ) -> ToolExecutionResponse:
        """将工具最终结果保存到操作账本。"""

        if operation is None:
            return response

        persisted_status = "failed"

        if response.status == "succeeded":
            persisted_status = "succeeded"

        elif response.status == "rejected":
            persisted_status = "rejected"

        elif (
            response.status == "timed_out"
            and definition is not None
            and not definition.metadata.read_only
        ):
            # 写操作发生超时后，副作用可能已经提交。
            persisted_status = "unknown"

        external_reference = self._extract_external_reference(
            response.output,
            (definition.metadata.external_reference_path if definition is not None else None),
        )

        try:
            await self._ledger.finalize(
                database_id=operation.database_id,
                status=persisted_status,
                result=response.output,
                latency_ms=response.latency_ms,
                external_reference=external_reference,
                error_type=(response.error.code if response.error is not None else None),
                error_message=(response.error.message if response.error is not None else None),
                error_details=(response.error.details if response.error is not None else None),
            )
        except Exception:
            return ToolExecutionResponse(
                tool_name=response.tool_name,
                status="failed",
                operation_id=operation.operation_id,
                error=ToolError(
                    code="operation_ledger_finalize_error",
                    message=("The tool finished, but its final ledger state could not be saved."),
                ),
                latency_ms=response.latency_ms,
            )

        return response.model_copy(
            update={
                "operation_id": operation.operation_id,
                "replayed": False,
            }
        )

    @staticmethod
    def _extract_external_reference(
        output: dict[str, Any] | None,
        path: str | None,
    ) -> str | None:
        """
        根据点分路径从工具结果中提取外部业务 ID。

        例如：
            path = "ticket.id"
        """

        if output is None or path is None:
            return None

        current: Any = output

        for path_part in path.split("."):
            if not isinstance(current, dict):
                return None

            if path_part not in current:
                return None

            current = current[path_part]

        if current is None:
            return None

        return str(current)

    @staticmethod
    def _response_from_existing(
        operation: OperationRecord,
    ) -> ToolExecutionResponse:
        """根据已有账本记录构造幂等回放结果。"""

        common = {
            "tool_name": operation.tool_name,
            "operation_id": operation.operation_id,
            "replayed": True,
            "latency_ms": operation.latency_ms or 0.0,
        }

        if operation.status == "succeeded":
            return ToolExecutionResponse(
                status="succeeded",
                output=operation.result,
                **common,
            )

        if operation.status == "rejected":
            return ToolExecutionResponse(
                status="rejected",
                error=ToolError(
                    code=(operation.error_type or "operation_rejected"),
                    message=(operation.error_message or "The operation was rejected."),
                    details=operation.error_details or {},
                ),
                **common,
            )

        if operation.status == "failed":
            return ToolExecutionResponse(
                status="failed",
                error=ToolError(
                    code=(operation.error_type or "operation_failed"),
                    message=(operation.error_message or "The operation failed."),
                    details=operation.error_details or {},
                ),
                **common,
            )

        if operation.status == "unknown":
            return ToolExecutionResponse(
                status="failed",
                error=ToolError(
                    code="operation_state_unknown",
                    message=(
                        "The previous execution may have completed, but its outcome is unknown."
                    ),
                    details={
                        "external_reference": (operation.external_reference),
                    },
                ),
                **common,
            )

        if operation.status == "cancelled":
            return ToolExecutionResponse(
                status="rejected",
                error=ToolError(
                    code="operation_cancelled",
                    message="The operation was cancelled.",
                ),
                **common,
            )

        return ToolExecutionResponse(
            status="failed",
            error=ToolError(
                code="operation_in_progress",
                message=("An operation with this idempotency key is already prepared or running."),
            ),
            **common,
        )

    @staticmethod
    def _latency_ms(started_at: float) -> float:
        return round(
            (perf_counter() - started_at) * 1000,
            2,
        )

    def _error_response(
        self,
        *,
        tool_name: str,
        status: ToolErrorStatus,
        started_at: float,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ToolExecutionResponse:
        """构造标准化工具错误响应。"""

        return ToolExecutionResponse(
            tool_name=tool_name,
            status=status,
            error=ToolError(
                code=code,
                message=message,
                details=details or {},
            ),
            latency_ms=self._latency_ms(started_at),
        )
