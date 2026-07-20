from pathlib import Path

import yaml
from pydantic import ValidationError

from app.benchmark.schemas import BenchmarkTaskSpec


class TaskLoadError(ValueError):
    """评测任务文件无法读取或校验失败。"""


def load_task_file(path: Path) -> BenchmarkTaskSpec:
    """读取并校验单个 YAML 任务文件。"""
    if not path.is_file():
        raise TaskLoadError(f"Task file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
    except OSError as exc:
        raise TaskLoadError(f"Unable to read task file: {path}") from exc
    except yaml.YAMLError as exc:
        raise TaskLoadError(f"Invalid YAML syntax in {path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise TaskLoadError(f"Task file must contain a YAML object: {path}")

    try:
        return BenchmarkTaskSpec.model_validate(raw_data)
    except ValidationError as exc:
        raise TaskLoadError(f"Task validation failed for {path}:\n{exc}") from exc


def load_task_directory(directory: Path) -> list[BenchmarkTaskSpec]:
    """递归加载目录中的全部 YAML 任务。"""
    if not directory.is_dir():
        raise TaskLoadError(f"Task directory does not exist: {directory}")

    task_paths = sorted(
        [
            *directory.rglob("*.yaml"),
            *directory.rglob("*.yml"),
        ]
    )

    if not task_paths:
        raise TaskLoadError(f"No YAML task files found in: {directory}")

    tasks: list[BenchmarkTaskSpec] = []
    seen_keys: set[tuple[str, int]] = set()

    for path in task_paths:
        task = load_task_file(path)
        identity = (task.task_key, task.version)

        if identity in seen_keys:
            raise TaskLoadError(
                f"Duplicate task identity found: {task.task_key} version {task.version}"
            )

        seen_keys.add(identity)
        tasks.append(task)

    return tasks
