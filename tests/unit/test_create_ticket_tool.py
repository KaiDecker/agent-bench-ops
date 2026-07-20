import pytest
from pydantic import ValidationError

from app.tools.implementations.tickets import (
    CreateTicketArguments,
)


def test_create_ticket_arguments_are_valid() -> None:
    arguments = CreateTicketArguments(
        requester_employee_id="emp_001",
        target_employee_id="emp_002",
        ticket_type="general",
        title="账号访问问题",
        description="员工无法访问内部系统。",
    )

    assert arguments.ticket_type == "general"
    assert arguments.risk_level == "medium"


def test_create_ticket_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CreateTicketArguments.model_validate(
            {
                "requester_employee_id": "emp_001",
                "target_employee_id": "emp_002",
                "ticket_type": "general",
                "title": "测试",
                "description": "测试内容",
                "unexpected": "value",
            }
        )


def test_create_ticket_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        CreateTicketArguments(
            requester_employee_id="emp_001",
            target_employee_id="emp_002",
            ticket_type="delete_everything",
            title="测试",
            description="测试内容",
        )
