import argparse
import asyncio

from sqlalchemy import select

from app.benchmark.reset import (
    capture_business_state,
    normalize_initial_state,
    reset_business_state,
)
from app.benchmark.schemas import BusinessInitialState
from app.persistence.database import AsyncSessionFactory
from app.persistence.platform_models import BenchmarkTask


async def reset_task_state(
    task_key: str,
    version: int,
) -> dict[str, list[dict[str, object]]]:
    """根据数据库中的评测任务恢复业务状态。"""

    async with AsyncSessionFactory() as session:
        async with session.begin():
            result = await session.execute(
                select(BenchmarkTask).where(
                    BenchmarkTask.task_key == task_key,
                    BenchmarkTask.version == version,
                )
            )

            task = result.scalar_one_or_none()

            if task is None:
                raise ValueError(f"Benchmark task not found: {task_key} v{version}")

            initial_state = BusinessInitialState.model_validate(task.initial_state)

            await reset_business_state(session, initial_state)

            actual_state = await capture_business_state(session)
            expected_state = normalize_initial_state(initial_state)

            if actual_state != expected_state:
                raise RuntimeError("Business state verification failed after reset.")

            return actual_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset business data from a benchmark task.")

    parser.add_argument(
        "--task-key",
        required=True,
        help="Benchmark task key.",
    )

    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Benchmark task version.",
    )

    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    try:
        state = await reset_task_state(
            task_key=args.task_key,
            version=args.version,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    counts = {entity: len(records) for entity, records in state.items()}

    print(f"Reset completed for {args.task_key} version {args.version}.")
    print(f"Entity counts: {counts}")


if __name__ == "__main__":
    asyncio.run(async_main())
