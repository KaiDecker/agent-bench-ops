from types import SimpleNamespace
from typing import Any

import pytest

from app.benchmark.reset import (
    BusinessStateResetBlockedError,
    assert_business_state_reset_safe,
)


class FakeResult:
    def __init__(
        self,
        rows: list[SimpleNamespace],
    ) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class FakeSession:
    def __init__(
        self,
        rows: list[SimpleNamespace],
        *,
        dialect_name: str = "postgresql",
    ) -> None:
        self._rows = rows
        self.statements: list[Any] = []
        self._execute_count = 0

        self._bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))

    def get_bind(self) -> SimpleNamespace:
        return self._bind

    async def execute(
        self,
        statement: Any,
    ) -> FakeResult:
        self.statements.append(statement)
        self._execute_count += 1

        if self._bind.dialect.name == "postgresql" and self._execute_count == 1:
            return FakeResult([])

        return FakeResult(self._rows)


async def test_reset_guard_allows_safe_reset() -> None:
    session = FakeSession([])

    await assert_business_state_reset_safe(
        session  # type: ignore[arg-type]
    )

    assert len(session.statements) == 2


async def test_reset_guard_rejects_unknown_operation() -> None:
    session = FakeSession(
        [
            SimpleNamespace(
                operation_id="op_unknown_001",
                run_id="run_001",
                tool_name="create_ticket",
                status="unknown",
            )
        ]
    )

    with pytest.raises(BusinessStateResetBlockedError) as exc_info:
        await assert_business_state_reset_safe(
            session  # type: ignore[arg-type]
        )

    assert exc_info.value.operations[0].operation_id == "op_unknown_001"
