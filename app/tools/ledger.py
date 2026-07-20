import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.persistence.platform_models import ToolOperation, generate_id

type PersistedOperationStatus = Literal[
    "prepared",
    "running",
    "succeeded",
    "failed",
    "unknown",
    "rejected",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """与数据库会话无关的工具操作快照。"""

    database_id: str
    operation_id: str
    run_id: str
    step_id: str | None
    tool_name: str
    arguments_hash: str
    idempotency_key: str
    status: str
    retry_count: int
    result: dict[str, Any] | None
    latency_ms: float | None
    error_type: str | None
    error_message: str | None
    error_details: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class OperationClaim:
    """一次幂等操作声明的结果。"""

    record: OperationRecord
    created: bool
    requested_arguments_hash: str


def normalize_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """将工具参数转换为稳定的 JSON 对象。"""

    serialized = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    normalized = json.loads(serialized)

    if not isinstance(normalized, dict):
        raise TypeError("Tool arguments must normalize to a JSON object")

    return normalized


def compute_arguments_hash(
    arguments: dict[str, Any],
) -> str:
    """计算与字典键顺序无关的 SHA-256 参数摘要。"""

    normalized = normalize_arguments(arguments)

    canonical_json = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class OperationLedger:
    """工具操作账本。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def claim(
        self,
        *,
        run_id: str,
        step_id: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        idempotency_key: str | None,
        risk_level: str,
        requires_approval: bool,
        is_idempotent: bool,
    ) -> OperationClaim:
        """
        创建或获取一个幂等工具操作。

        相同 run_id 和 idempotency_key 只允许存在一条记录。
        """

        normalized_arguments = normalize_arguments(arguments)
        arguments_hash = compute_arguments_hash(arguments)

        resolved_key = idempotency_key or f"{tool_name}:{uuid4().hex}"

        if len(resolved_key) > 200:
            raise ValueError("idempotency_key cannot exceed 200 characters")

        table = ToolOperation.__table__

        statement = (
            insert(table)
            .values(
                id=generate_id(),
                operation_id=f"op_{uuid4().hex}",
                run_id=run_id,
                step_id=step_id,
                tool_name=tool_name,
                arguments=normalized_arguments,
                arguments_hash=arguments_hash,
                idempotency_key=resolved_key,
                risk_level=risk_level,
                requires_approval=requires_approval,
                is_idempotent=is_idempotent,
                status="prepared",
                retry_count=0,
            )
            .on_conflict_do_nothing(constraint=("uq_tool_operations_run_id_idempotency_key"))
            .returning(table.c.id)
        )

        async with self._session_factory.begin() as session:
            result = await session.execute(statement)
            inserted_id = result.scalar_one_or_none()

            operation_result = await session.execute(
                select(ToolOperation)
                .where(
                    ToolOperation.run_id == run_id,
                    ToolOperation.idempotency_key == resolved_key,
                )
                .with_for_update()
            )

            operation = operation_result.scalar_one()

            created = inserted_id is not None

            if not created:
                operation.retry_count += 1

            record = self._snapshot(operation)

        return OperationClaim(
            record=record,
            created=created,
            requested_arguments_hash=arguments_hash,
        )

    async def mark_running(
        self,
        database_id: str,
    ) -> None:
        """将操作标记为正在执行。"""

        async with self._session_factory.begin() as session:
            operation = await self._get_for_update(
                session,
                database_id,
            )

            operation.status = "running"
            operation.started_at = datetime.now(UTC)

    async def finalize(
        self,
        *,
        database_id: str,
        status: PersistedOperationStatus,
        result: dict[str, Any] | None,
        latency_ms: float,
        error_type: str | None,
        error_message: str | None,
        error_details: dict[str, Any] | None,
    ) -> None:
        """保存工具操作最终状态。"""

        async with self._session_factory.begin() as session:
            operation = await self._get_for_update(
                session,
                database_id,
            )

            operation.status = status
            operation.result = result
            operation.latency_ms = latency_ms
            operation.error_type = error_type
            operation.error_message = error_message
            operation.error_details = error_details
            operation.finished_at = datetime.now(UTC)

    @staticmethod
    async def _get_for_update(
        session: AsyncSession,
        database_id: str,
    ) -> ToolOperation:
        result = await session.execute(
            select(ToolOperation).where(ToolOperation.id == database_id).with_for_update()
        )

        operation = result.scalar_one_or_none()

        if operation is None:
            raise RuntimeError(f"Tool operation does not exist: {database_id}")

        return operation

    @staticmethod
    def _snapshot(
        operation: ToolOperation,
    ) -> OperationRecord:
        return OperationRecord(
            database_id=operation.id,
            operation_id=operation.operation_id,
            run_id=operation.run_id,
            step_id=operation.step_id,
            tool_name=operation.tool_name,
            arguments_hash=operation.arguments_hash,
            idempotency_key=operation.idempotency_key,
            status=operation.status,
            retry_count=operation.retry_count,
            result=operation.result,
            latency_ms=operation.latency_ms,
            error_type=operation.error_type,
            error_message=operation.error_message,
            error_details=operation.error_details,
        )
