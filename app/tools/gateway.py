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
    ToolError,
    ToolErrorStatus,
    ToolExecutionContext,
    ToolExecutionResponse,
)


class ToolGateway:
    """工具调用统一入口。"""

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
        started_at = perf_counter()
        definition = self._registry.get(tool_name)

        claim: OperationClaim | None = None

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

        try:
            async with asyncio.timeout(definition.metadata.timeout_seconds):
                raw_result = await definition.handler(
                    session,
                    validated_arguments,
                    context,
                )

                validated_result = definition.result_model.model_validate(raw_result)

        except TimeoutError:
            response = self._error_response(
                tool_name=tool_name,
                status="timed_out",
                started_at=started_at,
                code="tool_timeout",
                message=(f"Tool execution exceeded {definition.metadata.timeout_seconds} seconds."),
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
        definition: Any,
    ) -> ToolExecutionResponse:
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
            and not (definition.metadata.read_only or definition.metadata.is_idempotent)
        ):
            persisted_status = "unknown"

        try:
            await self._ledger.finalize(
                database_id=operation.database_id,
                status=persisted_status,
                result=response.output,
                latency_ms=response.latency_ms,
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
    def _response_from_existing(
        operation: OperationRecord,
    ) -> ToolExecutionResponse:
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
