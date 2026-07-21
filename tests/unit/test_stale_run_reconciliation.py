from types import SimpleNamespace

from app.agent.reconciliation import (
    classify_stale_run,
)


def operation(
    operation_id: str,
    status: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        operation_id=operation_id,
        status=status,
    )


def test_closes_run_without_unresolved_operations() -> None:
    action = classify_stale_run(
        checkpoint_ref=None,
        unresolved_operations=[],
        inconclusive_operation_ids=[],
    )

    assert action == "close_stale"


def test_retains_checkpoint_run() -> None:
    action = classify_stale_run(
        checkpoint_ref="checkpoint_001",
        unresolved_operations=[],
        inconclusive_operation_ids=[],
    )

    assert action == "retain_checkpoint"


def test_unknown_operation_requires_explicit_override() -> None:
    unresolved = [
        operation("op_001", "unknown"),
    ]

    assert (
        classify_stale_run(
            checkpoint_ref=None,
            unresolved_operations=unresolved,
            inconclusive_operation_ids=[],
        )
        == "manual_review"
    )

    assert (
        classify_stale_run(
            checkpoint_ref=None,
            unresolved_operations=unresolved,
            inconclusive_operation_ids=[
                "op_001",
            ],
        )
        == "close_inconclusive"
    )
