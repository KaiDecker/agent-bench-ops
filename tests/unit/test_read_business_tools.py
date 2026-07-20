import pytest
from pydantic import ValidationError

from app.tools.implementations.accounts import (
    GetAccountArguments,
)
from app.tools.implementations.permissions import (
    ListEmployeePermissionsArguments,
)
from app.tools.implementations.tickets import (
    GetTicketArguments,
)


def test_get_account_accepts_employee_id() -> None:
    arguments = GetAccountArguments(employee_id="emp_001")

    assert arguments.employee_id == "emp_001"
    assert arguments.username is None


def test_get_account_accepts_username() -> None:
    arguments = GetAccountArguments(username="zhangsan")

    assert arguments.username == "zhangsan"
    assert arguments.employee_id is None


def test_get_account_rejects_no_selector() -> None:
    with pytest.raises(
        ValidationError,
        match="Exactly one",
    ):
        GetAccountArguments()


def test_get_account_rejects_multiple_selectors() -> None:
    with pytest.raises(
        ValidationError,
        match="Exactly one",
    ):
        GetAccountArguments(
            employee_id="emp_001",
            username="zhangsan",
        )


def test_list_permissions_defaults_to_active_only() -> None:
    arguments = ListEmployeePermissionsArguments(employee_id="emp_001")

    assert arguments.include_revoked is False


def test_list_permissions_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ListEmployeePermissionsArguments.model_validate(
            {
                "employee_id": "emp_001",
                "unexpected": True,
            }
        )


def test_get_ticket_arguments_are_valid() -> None:
    arguments = GetTicketArguments(ticket_id="ticket_001")

    assert arguments.ticket_id == "ticket_001"


def test_get_ticket_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        GetTicketArguments(ticket_id="")
