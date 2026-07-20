import argparse
import asyncio
from pathlib import Path

from app.benchmark.loader import TaskLoadError, load_task_directory
from app.persistence.database import AsyncSessionFactory
from app.persistence.repositories import BenchmarkTaskRepository


async def import_tasks(directory: Path) -> int:
    """加载目录任务并幂等写入数据库。"""
    tasks = load_task_directory(directory)
    repository = BenchmarkTaskRepository()

    async with AsyncSessionFactory() as session:
        async with session.begin():
            for task_spec in tasks:
                await repository.upsert(session, task_spec)

    return len(tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import AgentBenchOps benchmark tasks.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("benchmark_tasks"),
        help="Directory containing benchmark YAML files.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    try:
        imported_count = await import_tasks(args.directory)
    except TaskLoadError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Imported {imported_count} benchmark task(s) from {args.directory}.")


if __name__ == "__main__":
    asyncio.run(async_main())
