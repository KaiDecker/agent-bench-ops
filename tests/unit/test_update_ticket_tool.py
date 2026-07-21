import pytest
from pydantic import ValidationError

from app.tools.implementations.tickets import (
    UpdateTicketArguments,
)


def test_update_ticket_accepts_title_change() -> None:
    arguments = UpdateTicketArguments(
        ticket_id="ticket_001",
        expected_version=1,
        title="新的工单标题",
    )

    assert arguments.expected_version == 1
    assert arguments.title == "新的工单标题"


def test_update_ticket_requires_a_change() -> None:
    with pytest.raises(
        ValidationError,
        match="At least one",
    ):
        UpdateTicketArguments(
            ticket_id="ticket_001",
            expected_version=1,
        )


def test_update_ticket_rejects_invalid_version() -> None:
    with pytest.raises(ValidationError):
        UpdateTicketArguments(
            ticket_id="ticket_001",
            expected_version=0,
            title="新的标题",
        )


def test_update_ticket_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        UpdateTicketArguments.model_validate(
            {
                "ticket_id": "ticket_001",
                "expected_version": 1,
                "title": "新的标题",
                "unexpected": True,
            }
        )


def test_update_ticket_accepts_resolution() -> None:
    arguments = UpdateTicketArguments(
        ticket_id="ticket_001",
        expected_version=1,
        resolution="已经修复账号权限。",
    )

    assert arguments.resolution == "已经修复账号权限。"
