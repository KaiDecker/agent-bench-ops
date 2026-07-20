import pytest
from pydantic import ValidationError

from app.benchmark.schemas import BusinessInitialState


def test_valid_initial_state() -> None:
    state = BusinessInitialState.model_validate(
        {
            "employees": [
                {
                    "id": "emp_001",
                    "employee_no": "E10001",
                    "name": "张三",
                    "status": "active",
                }
            ],
            "accounts": [
                {
                    "id": "account_001",
                    "employee_id": "emp_001",
                    "username": "zhangsan",
                    "status": "active",
                }
            ],
        }
    )

    assert len(state.employees) == 1
    assert state.accounts[0].employee_id == "emp_001"


def test_unknown_employee_reference_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="unknown employee",
    ):
        BusinessInitialState.model_validate(
            {
                "employees": [],
                "accounts": [
                    {
                        "id": "account_001",
                        "employee_id": "emp_missing",
                        "username": "missing",
                    }
                ],
            }
        )


def test_duplicate_employee_id_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="Duplicate values",
    ):
        BusinessInitialState.model_validate(
            {
                "employees": [
                    {
                        "id": "emp_001",
                        "employee_no": "E10001",
                        "name": "张三",
                    },
                    {
                        "id": "emp_001",
                        "employee_no": "E10002",
                        "name": "李四",
                    },
                ]
            }
        )
