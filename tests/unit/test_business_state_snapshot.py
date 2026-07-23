from datetime import UTC, datetime
from typing import Any

from app.evaluation.state_oracle import (
    BusinessStateSnapshot,
    BusinessStateSnapshotService,
)


class FakeResult:
    def __init__(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        self._rows = rows

    def mappings(
        self,
    ) -> "FakeResult":
        return self

    def all(
        self,
    ) -> list[dict[str, Any]]:
        return self._rows


class FakeSession:
    def __init__(
        self,
        result_sets: list[list[dict[str, Any]]],
    ) -> None:
        self._result_sets = list(result_sets)
        self.statements: list[Any] = []

    async def execute(
        self,
        statement: Any,
    ) -> FakeResult:
        self.statements.append(statement)

        return FakeResult(self._result_sets.pop(0))


def test_snapshot_has_deterministic_order() -> None:
    snapshot = BusinessStateSnapshot(
        employees=[
            {
                "id": "emp_002",
                "employee_no": "E10002",
                "name": "李四",
                "department": "数据分析部",
                "status": "active",
            },
            {
                "id": "emp_001",
                "employee_no": "E10001",
                "name": "张三",
                "department": "数据平台部",
                "status": "active",
            },
        ],
        tickets=[
            {
                "id": "ticket_002",
                "source_operation_id": "op_002",
                "requester_employee_id": "emp_001",
                "target_employee_id": "emp_002",
                "ticket_type": "general",
                "status": "open",
                "risk_level": "medium",
                "title": "第二张",
                "description": "第二张工单",
                "resolution": None,
                "version": 1,
            },
            {
                "id": "ticket_001",
                "source_operation_id": "op_001",
                "requester_employee_id": "emp_001",
                "target_employee_id": "emp_002",
                "ticket_type": "general",
                "status": "open",
                "risk_level": "medium",
                "title": "第一张",
                "description": "第一张工单",
                "resolution": None,
                "version": 1,
            },
        ],
        ticket_mutations=[
            {
                "id": "mutation_002",
                "ticket_id": "ticket_001",
                "operation_id": "op_update_002",
                "previous_version": 2,
                "new_version": 3,
                "change_payload": {
                    "z": 1,
                    "a": {
                        "y": 2,
                        "x": 1,
                    },
                },
                "result_snapshot": {
                    "version": 3,
                },
            },
            {
                "id": "mutation_001",
                "ticket_id": "ticket_001",
                "operation_id": "op_update_001",
                "previous_version": 1,
                "new_version": 2,
                "change_payload": {
                    "status": "resolved",
                },
                "result_snapshot": {
                    "version": 2,
                },
            },
        ],
    )

    payload = snapshot.to_json_dict()

    assert [item["id"] for item in payload["employees"]] == [
        "emp_001",
        "emp_002",
    ]

    assert [item["id"] for item in payload["tickets"]] == [
        "ticket_001",
        "ticket_002",
    ]

    assert [item["id"] for item in payload["ticket_mutations"]] == [
        "mutation_001",
        "mutation_002",
    ]

    change_payload = payload["ticket_mutations"][1]["change_payload"]

    assert list(change_payload.keys()) == [
        "a",
        "z",
    ]

    assert list(change_payload["a"].keys()) == [
        "x",
        "y",
    ]


async def test_snapshot_service_reads_all_entities() -> None:
    granted_at = datetime(
        2026,
        7,
        23,
        8,
        0,
        tzinfo=UTC,
    )

    session = FakeSession(
        [
            [
                {
                    "id": "emp_001",
                    "employee_no": "E10001",
                    "name": "张三",
                    "department": "数据平台部",
                    "status": "active",
                }
            ],
            [
                {
                    "id": "account_001",
                    "employee_id": "emp_001",
                    "username": "zhangsan",
                    "status": "active",
                    "version": 1,
                }
            ],
            [
                {
                    "id": "permission_001",
                    "code": "ticket.write",
                    "name": "创建工单",
                    "description": None,
                    "risk_level": "medium",
                    "requires_approval": False,
                }
            ],
            [
                {
                    "employee_id": "emp_001",
                    "permission_id": "permission_001",
                    "status": "active",
                    "granted_by": None,
                    "granted_at": granted_at,
                    "revoked_at": None,
                }
            ],
            [
                {
                    "id": "ticket_001",
                    "source_operation_id": "op_001",
                    "requester_employee_id": "emp_001",
                    "target_employee_id": "emp_001",
                    "ticket_type": "general",
                    "status": "open",
                    "risk_level": "medium",
                    "title": "测试",
                    "description": "测试工单",
                    "resolution": None,
                    "version": 1,
                }
            ],
            [
                {
                    "id": "mutation_001",
                    "ticket_id": "ticket_001",
                    "operation_id": "op_update_001",
                    "previous_version": 1,
                    "new_version": 2,
                    "change_payload": {
                        "status": "resolved",
                    },
                    "result_snapshot": {
                        "version": 2,
                    },
                }
            ],
        ]
    )

    service = BusinessStateSnapshotService()

    snapshot = await service.capture_from_session(
        session  # type: ignore[arg-type]
    )

    assert len(session.statements) == 6
    assert snapshot.employees[0].id == ("emp_001")
    assert snapshot.accounts[0].id == ("account_001")
    assert snapshot.permissions[0].code == ("ticket.write")
    assert snapshot.employee_permissions[0].granted_at == granted_at
    assert snapshot.tickets[0].id == ("ticket_001")
    assert snapshot.ticket_mutations[0].operation_id == "op_update_001"


def test_snapshot_excludes_database_timestamps() -> None:
    snapshot = BusinessStateSnapshot(
        employees=[
            {
                "id": "emp_001",
                "employee_no": "E10001",
                "name": "张三",
                "department": None,
                "status": "active",
            }
        ]
    )

    payload = snapshot.to_json_dict()
    employee = payload["employees"][0]

    assert "created_at" not in employee
    assert "updated_at" not in employee
