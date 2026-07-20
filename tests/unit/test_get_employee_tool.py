import pytest
from pydantic import ValidationError

from app.tools.implementations.employees import (
    GetEmployeeArguments,
)


def test_get_employee_accepts_one_selector() -> None:
    arguments = GetEmployeeArguments(employee_id="emp_001")

    assert arguments.employee_id == "emp_001"


def test_get_employee_rejects_no_selector() -> None:
    with pytest.raises(
        ValidationError,
        match="Exactly one",
    ):
        GetEmployeeArguments()


def test_get_employee_rejects_multiple_selectors() -> None:
    with pytest.raises(
        ValidationError,
        match="Exactly one",
    ):
        GetEmployeeArguments(
            employee_id="emp_001",
            name="张三",
        )


def test_get_employee_rejects_unknown_argument() -> None:
    with pytest.raises(ValidationError):
        GetEmployeeArguments.model_validate(
            {
                "employee_id": "emp_001",
                "unexpected": "value",
            }
        )
