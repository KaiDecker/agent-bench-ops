from collections.abc import (
    Collection,
    Mapping,
)
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import (
    AgentRun,
    ToolOperation,
)

type ReconciliationAction = Literal[
    "close_stale",
    "close_inconclusive",
    "retain_checkpoint",
    "release_resume_claim",
    "manual_review",
]


UNRESOLVED_OPERATION_STATUSES = frozenset(
    {
        "prepared",
        "running",
        "unknown",
    }
)


def parse_resume_started_at(
    value: Any,
) -> datetime | None:
    """解析 configuration.resume_started_at。"""

    if not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return None

    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    run_id: str
    action: ReconciliationAction
    applied: bool
    unresolved_operations: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "action": self.action,
            "applied": self.applied,
            "unresolved_operations": list(self.unresolved_operations),
            "reason": self.reason,
        }


def classify_stale_run(
    *,
    checkpoint_ref: str | None,
    unresolved_operations: Collection[ToolOperation],
    inconclusive_operation_ids: Collection[str],
    configuration: Mapping[
        str,
        Any,
    ]
    | None = None,
    resume_claim_cutoff: datetime | None = None,
) -> ReconciliationAction:
    """对 stale running AgentRun 做保守分类。"""

    if checkpoint_ref:
        runtime_configuration = dict(configuration or {})

        resume_in_progress = (
            runtime_configuration.get(
                "resume_in_progress",
                False,
            )
            is True
        )

        if not resume_in_progress:
            return "retain_checkpoint"

        # 未要求检查恢复租约时保持原有行为。
        if resume_claim_cutoff is None:
            return "retain_checkpoint"

        resume_started_at = parse_resume_started_at(runtime_configuration.get("resume_started_at"))

        # 缺失或非法的租约时间不能自动释放。
        if resume_started_at is None:
            return "manual_review"

        pending_nodes = runtime_configuration.get("next_nodes")

        # 没有待执行节点时，不能安全恢复。
        if (
            not isinstance(
                pending_nodes,
                list,
            )
            or not pending_nodes
        ):
            return "manual_review"

        if resume_started_at <= resume_claim_cutoff:
            return "release_resume_claim"

        return "retain_checkpoint"

    if not unresolved_operations:
        return "close_stale"

    forced_ids = set(inconclusive_operation_ids)

    can_close_inconclusive = all(
        operation.status == "unknown" and operation.operation_id in forced_ids
        for operation in unresolved_operations
    )

    if can_close_inconclusive:
        return "close_inconclusive"

    return "manual_review"


class StaleRunReconciler:
    """处理长时间停留在 running 的 AgentRun。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = (AsyncSessionFactory),
    ) -> None:
        self._session_factory = session_factory

    async def reconcile(
        self,
        *,
        older_than: timedelta,
        apply: bool,
        inconclusive_operation_ids: Collection[str] = (),
        run_ids: Collection[str] = (),
    ) -> list[ReconciliationResult]:
        if older_than.total_seconds() <= 0:
            raise ValueError("older_than must be positive")

        cutoff = datetime.now(UTC) - older_than
        forced_ids = set(inconclusive_operation_ids)

        normalized_run_ids = tuple(sorted({run_id.strip() for run_id in run_ids if run_id.strip()}))

        if run_ids and not normalized_run_ids:
            raise ValueError("run_ids must contain at least one non-empty run ID")

        async with self._session_factory.begin() as session:
            statement = select(AgentRun).where(
                AgentRun.status == "running",
                AgentRun.created_at < cutoff,
            )

            if normalized_run_ids:
                statement = statement.where(AgentRun.id.in_(normalized_run_ids))

            statement = statement.order_by(AgentRun.created_at)

            if apply:
                statement = statement.with_for_update()

            run_result = await session.execute(statement)
            runs = list(run_result.scalars())

            results: list[ReconciliationResult] = []

            for run in runs:
                operation_result = await session.execute(
                    select(ToolOperation)
                    .where(
                        ToolOperation.run_id == run.id,
                        ToolOperation.status.in_(UNRESOLVED_OPERATION_STATUSES),
                    )
                    .order_by(ToolOperation.created_at)
                )

                operations = list(operation_result.scalars())

                action = classify_stale_run(
                    checkpoint_ref=run.checkpoint_ref,
                    unresolved_operations=operations,
                    inconclusive_operation_ids=(forced_ids),
                    configuration=run.configuration,
                    resume_claim_cutoff=cutoff,
                )

                reason = self._reason_for_action(
                    action=action,
                )

                if apply:
                    self._apply_action(
                        run=run,
                        operations=operations,
                        action=action,
                    )

                results.append(
                    ReconciliationResult(
                        run_id=run.id,
                        action=action,
                        applied=(
                            apply
                            and action
                            in {
                                "close_stale",
                                "close_inconclusive",
                                "release_resume_claim",
                            }
                        ),
                        unresolved_operations=tuple(
                            operation.operation_id for operation in operations
                        ),
                        reason=reason,
                    )
                )

            return results

    @staticmethod
    def _reason_for_action(
        *,
        action: ReconciliationAction,
    ) -> str:
        reasons = {
            "close_stale": ("No checkpoint or unresolved tool operation exists."),
            "close_inconclusive": (
                "Historical recovery evidence is no "
                "longer sufficient to determine whether "
                "the tool side effect committed."
            ),
            "retain_checkpoint": ("A checkpoint reference exists and the run may be resumable."),
            "release_resume_claim": (
                "The checkpoint is resumable, but the previous resume ownership claim has expired."
            ),
            "manual_review": (
                "The run has unresolved state that cannot be reconciled automatically."
            ),
        }

        return reasons[action]

    @staticmethod
    def _apply_action(
        *,
        run: AgentRun,
        operations: list[ToolOperation],
        action: ReconciliationAction,
    ) -> None:
        now = datetime.now(UTC)

        if action == "release_resume_claim":
            configuration = dict(run.configuration or {})

            configuration.pop(
                "resume_started_at",
                None,
            )

            configuration.update(
                {
                    "paused": True,
                    "resume_in_progress": False,
                    "resume_status": "paused",
                    "pause_reason": ("resume_claim_expired"),
                }
            )

            run.configuration = configuration
            run.finished_at = None
            return

        if action == "close_stale":
            run.status = "failed"
            run.error_type = "stale_run_reconciled"
            run.error_message = "Stale run had no checkpoint or unresolved tool operations."
            run.finished_at = now
            return

        if action != "close_inconclusive":
            return

        recovery_details = {
            "reason": ("historical_recovery_inconclusive"),
            "business_effect_found": False,
            "benchmark_reset_may_have_removed_evidence": (True),
            "resolution": (
                "The operation cannot be classified as "
                "committed or not committed from the "
                "remaining evidence."
            ),
        }

        for operation in operations:
            operation.status = "failed"
            operation.recovery_count += 1
            operation.recovered_at = now
            operation.recovery_details = recovery_details.copy()
            operation.error_type = "historical_recovery_inconclusive"
            operation.error_message = (
                "Historical benchmark resets may have "
                "removed the business object required "
                "to determine the operation outcome."
            )
            operation.error_details = recovery_details.copy()
            operation.finished_at = now

        run.status = "failed"
        run.error_type = "stale_run_reconciled"
        run.error_message = (
            "The run contained historical unknown tool "
            "operations whose side effects can no longer "
            "be determined."
        )
        run.finished_at = now


__all__ = [
    "ReconciliationResult",
    "StaleRunReconciler",
    "classify_stale_run",
]
