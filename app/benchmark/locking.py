from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.persistence.database import engine

BENCHMARK_LOCK_NAMESPACE = 1_094_864_720
BENCHMARK_LOCK_RESOURCE = 1


class BenchmarkExecutionLockError(RuntimeError):
    """Benchmark 全局执行锁错误。"""


class BenchmarkExecutionLockBusyError(BenchmarkExecutionLockError):
    """另一个 Runner 正在占用共享业务状态。"""


class BenchmarkExecutionLockReleaseError(BenchmarkExecutionLockError):
    """数据库锁释放失败。"""


class UnsupportedBenchmarkLockBackendError(BenchmarkExecutionLockError):
    """当前数据库不支持 PostgreSQL advisory lock。"""


def validate_advisory_lock_component(
    value: int,
    *,
    field_name: str,
) -> int:
    """
    PostgreSQL 双整数 advisory lock 参数必须为
    32 位有符号整数。
    """

    minimum = -(2**31)
    maximum = 2**31 - 1

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")

    if not minimum <= value <= maximum:
        raise ValueError(f"{field_name} must fit in a signed 32-bit integer")

    return value


@asynccontextmanager
async def postgres_benchmark_execution_lock(
    *,
    database_engine: AsyncEngine = engine,
    namespace: int = (BENCHMARK_LOCK_NAMESPACE),
    resource: int = BENCHMARK_LOCK_RESOURCE,
) -> AsyncIterator[None]:
    """
    获取跨连接、跨进程的 Benchmark 全局锁。

    锁为 PostgreSQL session-level advisory lock。
    持有锁的数据库连接在整个实验期间保持打开。

    使用 try-lock 语义：
    若已有 Runner 持锁，立即失败，不进行等待。
    """

    namespace = validate_advisory_lock_component(
        namespace,
        field_name="namespace",
    )

    resource = validate_advisory_lock_component(
        resource,
        field_name="resource",
    )

    parameters = {
        "namespace": namespace,
        "resource": resource,
    }

    async with database_engine.connect() as connection:
        dialect_name = connection.dialect.name

        if dialect_name != "postgresql":
            raise (
                UnsupportedBenchmarkLockBackendError(
                    "Benchmark execution locking "
                    "requires PostgreSQL; "
                    f"current dialect: {dialect_name}"
                )
            )

        try:
            acquisition_result = await connection.execute(
                text("SELECT pg_try_advisory_lock(:namespace, :resource)"),
                parameters,
            )

            acquired = bool(acquisition_result.scalar_one())

        except BaseException:
            # 获取结果不确定时关闭底层连接，
            # PostgreSQL 会随 session 结束释放锁。
            await connection.invalidate()
            raise

        if not acquired:
            raise BenchmarkExecutionLockBusyError(
                "Another BenchmarkRunner is currently using the shared business state."
            )

        body_error: BaseException | None = None

        try:
            yield

        except BaseException as exc:
            body_error = exc
            raise

        finally:
            release_error: BaseException | None = None

            try:
                release_result = await connection.execute(
                    text("SELECT pg_advisory_unlock(:namespace, :resource)"),
                    parameters,
                )

                released = bool(release_result.scalar_one())

                if not released:
                    release_error = BenchmarkExecutionLockReleaseError(
                        "PostgreSQL reported "
                        "that the Benchmark "
                        "execution lock was "
                        "not owned by this "
                        "connection."
                    )

                    await connection.invalidate()

            except BaseException as exc:
                release_error = exc
                await connection.invalidate()

            if release_error is not None:
                message = f"Failed to release the Benchmark execution lock: {release_error}"

                if body_error is not None:
                    # 不用释放错误覆盖实验本身的错误。
                    body_error.add_note(message)

                elif isinstance(
                    release_error,
                    BenchmarkExecutionLockReleaseError,
                ):
                    raise release_error

                else:
                    raise (BenchmarkExecutionLockReleaseError(message)) from release_error


__all__ = [
    "BENCHMARK_LOCK_NAMESPACE",
    "BENCHMARK_LOCK_RESOURCE",
    "BenchmarkExecutionLockBusyError",
    "BenchmarkExecutionLockError",
    "BenchmarkExecutionLockReleaseError",
    "UnsupportedBenchmarkLockBackendError",
    "postgres_benchmark_execution_lock",
    "validate_advisory_lock_component",
]
