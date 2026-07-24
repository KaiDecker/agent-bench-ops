from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any

import pytest

from app.benchmark.locking import (
    BenchmarkExecutionLockBusyError,
    BenchmarkExecutionLockReleaseError,
    UnsupportedBenchmarkLockBackendError,
    postgres_benchmark_execution_lock,
    validate_advisory_lock_component,
)


class FakeScalarResult:
    def __init__(
        self,
        value: bool,
    ) -> None:
        self._value = value

    def scalar_one(
        self,
    ) -> bool:
        return self._value


class FakeConnection:
    def __init__(
        self,
        responses: list[bool],
        *,
        dialect_name: str = "postgresql",
    ) -> None:
        self._responses = list(responses)

        self.dialect = SimpleNamespace(name=dialect_name)

        self.statements: list[str] = []
        self.parameters: list[dict[str, Any]] = []

        self.entered = False
        self.exited = False
        self.invalidated = False

    async def __aenter__(
        self,
    ) -> "FakeConnection":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.exited = True

    async def execute(
        self,
        statement: Any,
        parameters: dict[
            str,
            Any,
        ],
    ) -> FakeScalarResult:
        self.statements.append(str(statement))

        self.parameters.append(dict(parameters))

        if not self._responses:
            raise AssertionError("No fake result remains")

        return FakeScalarResult(self._responses.pop(0))

    async def invalidate(
        self,
    ) -> None:
        self.invalidated = True


class FakeEngine:
    def __init__(
        self,
        connection: FakeConnection,
    ) -> None:
        self.connection = connection

    def connect(
        self,
    ) -> AbstractAsyncContextManager[FakeConnection]:
        return self.connection


async def test_lock_acquires_and_releases() -> None:
    connection = FakeConnection(
        [
            True,
            True,
        ]
    )

    engine = FakeEngine(connection)

    async with postgres_benchmark_execution_lock(
        database_engine=(
            engine  # type: ignore[arg-type]
        )
    ):
        assert connection.entered is True

    assert connection.exited is True
    assert connection.invalidated is False
    assert len(connection.statements) == 2

    assert "pg_try_advisory_lock" in connection.statements[0]

    assert "pg_advisory_unlock" in connection.statements[1]


async def test_busy_lock_fails_immediately() -> None:
    connection = FakeConnection([False])

    engine = FakeEngine(connection)

    with pytest.raises(
        BenchmarkExecutionLockBusyError,
        match="Another BenchmarkRunner",
    ):
        async with postgres_benchmark_execution_lock(
            database_engine=(
                engine  # type: ignore[arg-type]
            )
        ):
            raise AssertionError("Lock body must not execute")

    assert len(connection.statements) == 1
    assert connection.exited is True


async def test_lock_rejects_non_postgresql_backend() -> None:
    connection = FakeConnection(
        [],
        dialect_name="sqlite",
    )

    engine = FakeEngine(connection)

    with pytest.raises(
        UnsupportedBenchmarkLockBackendError,
        match="requires PostgreSQL",
    ):
        async with postgres_benchmark_execution_lock(
            database_engine=(
                engine  # type: ignore[arg-type]
            )
        ):
            raise AssertionError("Lock body must not execute")

    assert connection.statements == []
    assert connection.exited is True


async def test_failed_unlock_invalidates_connection() -> None:
    connection = FakeConnection(
        [
            True,
            False,
        ]
    )

    engine = FakeEngine(connection)

    with pytest.raises(
        BenchmarkExecutionLockReleaseError,
        match="not owned",
    ):
        async with postgres_benchmark_execution_lock(
            database_engine=(
                engine  # type: ignore[arg-type]
            )
        ):
            pass

    assert connection.invalidated is True
    assert connection.exited is True


async def test_body_error_is_preserved_after_release() -> None:
    connection = FakeConnection(
        [
            True,
            True,
        ]
    )

    engine = FakeEngine(connection)

    with pytest.raises(
        RuntimeError,
        match="experiment exploded",
    ):
        async with postgres_benchmark_execution_lock(
            database_engine=(
                engine  # type: ignore[arg-type]
            )
        ):
            raise RuntimeError("experiment exploded")

    assert len(connection.statements) == 2
    assert connection.exited is True


@pytest.mark.parametrize(
    "value",
    [
        -(2**31) - 1,
        2**31,
    ],
)
def test_rejects_out_of_range_lock_component(
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="signed 32-bit",
    ):
        validate_advisory_lock_component(
            value,
            field_name="value",
        )
