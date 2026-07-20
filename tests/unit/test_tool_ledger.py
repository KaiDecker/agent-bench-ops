from app.tools.ledger import compute_arguments_hash


def test_arguments_hash_ignores_dictionary_order() -> None:
    first = {
        "employee_id": "emp_001",
        "options": {
            "include_status": True,
            "language": "zh-CN",
        },
    }

    second = {
        "options": {
            "language": "zh-CN",
            "include_status": True,
        },
        "employee_id": "emp_001",
    }

    assert compute_arguments_hash(first) == compute_arguments_hash(second)


def test_arguments_hash_changes_with_values() -> None:
    first = compute_arguments_hash({"employee_id": "emp_001"})

    second = compute_arguments_hash({"employee_id": "emp_002"})

    assert first != second
