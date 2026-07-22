import pytest

from app.agent.runtime import (
    validate_resume_configuration,
)


def test_resume_in_progress_has_priority() -> None:
    with pytest.raises(
        RuntimeError,
        match="resume is already in progress",
    ):
        validate_resume_configuration(
            run_id="run_001",
            configuration={
                "paused": False,
                "resume_in_progress": True,
                "next_nodes": [
                    "tools",
                ],
            },
        )


def test_resume_requires_paused_run() -> None:
    with pytest.raises(
        RuntimeError,
        match="not marked as paused",
    ):
        validate_resume_configuration(
            run_id="run_001",
            configuration={
                "paused": False,
                "resume_in_progress": False,
                "next_nodes": [
                    "tools",
                ],
            },
        )


def test_resume_requires_pending_nodes() -> None:
    with pytest.raises(
        RuntimeError,
        match="has no pending nodes",
    ):
        validate_resume_configuration(
            run_id="run_001",
            configuration={
                "paused": True,
                "resume_in_progress": False,
                "next_nodes": [],
            },
        )


def test_resume_rejects_invalid_pending_nodes() -> None:
    with pytest.raises(
        RuntimeError,
        match="invalid pending nodes",
    ):
        validate_resume_configuration(
            run_id="run_001",
            configuration={
                "paused": True,
                "resume_in_progress": False,
                "next_nodes": [
                    "",
                ],
            },
        )


def test_resume_returns_pending_nodes() -> None:
    result = validate_resume_configuration(
        run_id="run_001",
        configuration={
            "paused": True,
            "resume_in_progress": False,
            "next_nodes": [
                " tools ",
            ],
        },
    )

    assert result == ("tools",)
