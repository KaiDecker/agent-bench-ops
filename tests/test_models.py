from sqlalchemy import UniqueConstraint

from app.persistence.base import Base
from app.persistence.models import (
    Account,
    AgentRun,
    Approval,
    BenchmarkTask,
    Employee,
    EmployeePermission,
    EvaluationResult,
    Permission,
    RunStep,
    Ticket,
    ToolOperation,
)


def test_all_models_are_registered() -> None:
    expected_tables = {
        Employee.__tablename__,
        Account.__tablename__,
        Permission.__tablename__,
        EmployeePermission.__tablename__,
        Ticket.__tablename__,
        BenchmarkTask.__tablename__,
        AgentRun.__tablename__,
        RunStep.__tablename__,
        ToolOperation.__tablename__,
        Approval.__tablename__,
        EvaluationResult.__tablename__,
    }

    registered_tables = set(Base.metadata.tables)

    assert expected_tables == registered_tables


def test_tool_operation_contains_recovery_fields() -> None:
    column_names = set(ToolOperation.__table__.columns.keys())

    assert {
        "operation_id",
        "arguments_hash",
        "idempotency_key",
        "status",
        "retry_count",
        "external_reference",
        "error_type",
    }.issubset(column_names)


def test_tool_operation_has_idempotency_constraint() -> None:
    unique_constraint_names = {
        constraint.name
        for constraint in ToolOperation.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert "uq_tool_operations_run_id_idempotency_key" in unique_constraint_names
