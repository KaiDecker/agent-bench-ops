from app.evaluation.rules import (
    StateExpectation,
)
from app.evaluation.state_oracle import (
    BusinessStateSnapshot,
    FinalStateOracle,
)


def employee_snapshot(
    *,
    status: str = "active",
) -> BusinessStateSnapshot:
    return BusinessStateSnapshot(
        employees=[
            {
                "id": "emp_001",
                "employee_no": "E10001",
                "name": "张三",
                "department": "数据平台部",
                "status": status,
            }
        ]
    )


def test_final_state_oracle_passes_matching_state() -> None:
    oracle = FinalStateOracle()

    result = oracle.evaluate(
        snapshot=employee_snapshot(),
        expectations=[
            StateExpectation.model_validate(
                {
                    "entity": "employees",
                    "where": {
                        "id": "emp_001",
                    },
                    "assertions": [
                        {
                            "field": "employee_no",
                            "operator": "eq",
                            "value": "E10001",
                        },
                        {
                            "field": "status",
                            "operator": "eq",
                            "value": "active",
                        },
                    ],
                }
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.passed_assertions == 2
    assert result.total_assertions == 2
    assert result.violations == []


def test_final_state_oracle_reports_partial_score() -> None:
    oracle = FinalStateOracle()

    result = oracle.evaluate(
        snapshot=employee_snapshot(status="inactive"),
        expectations=[
            StateExpectation.model_validate(
                {
                    "entity": "employees",
                    "where": {
                        "id": "emp_001",
                    },
                    "assertions": [
                        {
                            "field": "employee_no",
                            "operator": "eq",
                            "value": "E10001",
                        },
                        {
                            "field": "status",
                            "operator": "eq",
                            "value": "active",
                        },
                    ],
                }
            )
        ],
    )

    assert result.passed is False
    assert result.score == 0.5
    assert result.passed_assertions == 1
    assert result.violations[0].code == "state_assertion_failed"


def test_final_state_oracle_reports_missing_entity() -> None:
    oracle = FinalStateOracle()

    result = oracle.evaluate(
        snapshot=employee_snapshot(),
        expectations=[
            StateExpectation.model_validate(
                {
                    "entity": "employees",
                    "where": {
                        "id": "emp_missing",
                    },
                    "assertions": [
                        {
                            "field": "status",
                            "operator": "eq",
                            "value": "active",
                        }
                    ],
                }
            )
        ],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.violations[0].code == "state_entity_not_found"


def test_final_state_oracle_rejects_duplicate_matches() -> None:
    snapshot = BusinessStateSnapshot(
        tickets=[
            {
                "id": "ticket_001",
                "source_operation_id": "op_001",
                "requester_employee_id": "emp_001",
                "target_employee_id": "emp_002",
                "ticket_type": "general",
                "status": "open",
                "risk_level": "medium",
                "title": "数据平台访问问题",
                "description": "测试",
                "resolution": None,
                "version": 1,
            },
            {
                "id": "ticket_002",
                "source_operation_id": "op_002",
                "requester_employee_id": "emp_001",
                "target_employee_id": "emp_002",
                "ticket_type": "general",
                "status": "open",
                "risk_level": "medium",
                "title": "数据平台访问问题",
                "description": "测试",
                "resolution": None,
                "version": 1,
            },
        ]
    )

    result = FinalStateOracle().evaluate(
        snapshot=snapshot,
        expectations=[
            StateExpectation.model_validate(
                {
                    "entity": "tickets",
                    "where": {
                        "requester_employee_id": ("emp_001"),
                        "target_employee_id": ("emp_002"),
                        "ticket_type": "general",
                    },
                    "assertions": [
                        {
                            "field": "status",
                            "operator": "eq",
                            "value": "open",
                        }
                    ],
                }
            )
        ],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.violations[0].code == "state_entity_ambiguous"


def test_final_state_oracle_supports_existence() -> None:
    result = FinalStateOracle().evaluate(
        snapshot=employee_snapshot(),
        expectations=[
            StateExpectation.model_validate(
                {
                    "entity": "employees",
                    "where": {
                        "id": "emp_001",
                    },
                    "assertions": [
                        {
                            "field": "unknown_field",
                            "operator": ("not_exists"),
                        },
                        {
                            "field": "department",
                            "operator": "exists",
                        },
                    ],
                }
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0


def test_final_state_oracle_supports_membership() -> None:
    result = FinalStateOracle().evaluate(
        snapshot=employee_snapshot(),
        expectations=[
            StateExpectation.model_validate(
                {
                    "entity": "employees",
                    "where": {
                        "id": "emp_001",
                    },
                    "assertions": [
                        {
                            "field": "status",
                            "operator": "in",
                            "value": [
                                "active",
                                "inactive",
                            ],
                        },
                        {
                            "field": "status",
                            "operator": "not_in",
                            "value": [
                                "terminated",
                            ],
                        },
                    ],
                }
            )
        ],
    )

    assert result.passed is True
    assert result.score == 1.0


def test_final_state_oracle_passes_empty_contract() -> None:
    result = FinalStateOracle().evaluate(
        snapshot=BusinessStateSnapshot(),
        expectations=[],
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.total_assertions == 0
