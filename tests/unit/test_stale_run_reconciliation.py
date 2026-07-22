from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.agent.reconciliation import (
    StaleRunReconciler,
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


def test_releases_expired_resume_claim() -> None:
    cutoff = datetime(
        2026,
        7,
        22,
        4,
        30,
        tzinfo=UTC,
    )

    action = classify_stale_run(
        checkpoint_ref="checkpoint_001",
        unresolved_operations=[],
        inconclusive_operation_ids=[],
        configuration={
            "paused": False,
            "resume_in_progress": True,
            "resume_started_at": (cutoff - timedelta(minutes=10)).isoformat(),
            "next_nodes": [
                "tools",
            ],
        },
        resume_claim_cutoff=cutoff,
    )

    assert action == "release_resume_claim"


def test_retains_active_resume_claim() -> None:
    cutoff = datetime(
        2026,
        7,
        22,
        4,
        30,
        tzinfo=UTC,
    )

    action = classify_stale_run(
        checkpoint_ref="checkpoint_001",
        unresolved_operations=[],
        inconclusive_operation_ids=[],
        configuration={
            "paused": False,
            "resume_in_progress": True,
            "resume_started_at": (cutoff + timedelta(seconds=1)).isoformat(),
            "next_nodes": [
                "tools",
            ],
        },
        resume_claim_cutoff=cutoff,
    )

    assert action == "retain_checkpoint"


def test_malformed_resume_claim_requires_review() -> None:
    action = classify_stale_run(
        checkpoint_ref="checkpoint_001",
        unresolved_operations=[],
        inconclusive_operation_ids=[],
        configuration={
            "resume_in_progress": True,
            "resume_started_at": ("not-a-datetime"),
            "next_nodes": [
                "tools",
            ],
        },
        resume_claim_cutoff=datetime.now(UTC),
    )

    assert action == "manual_review"


def test_apply_releases_resume_claim() -> None:
    run = SimpleNamespace(
        configuration={
            "paused": False,
            "resume_in_progress": True,
            "resume_status": "in_progress",
            "resume_started_at": ("2026-07-22T04:00:00+00:00"),
            "next_nodes": [
                "tools",
            ],
        },
        finished_at=None,
    )

    StaleRunReconciler._apply_action(
        run=run,
        operations=[],
        action="release_resume_claim",
    )

    assert run.configuration["paused"] is True

    assert run.configuration["resume_in_progress"] is False

    assert run.configuration["resume_status"] == "paused"

    assert run.configuration["pause_reason"] == "resume_claim_expired"

    assert "resume_started_at" not in (run.configuration)
