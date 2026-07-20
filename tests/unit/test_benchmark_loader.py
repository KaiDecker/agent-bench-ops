from pathlib import Path

import pytest

from app.benchmark.loader import (
    TaskLoadError,
    load_task_directory,
    load_task_file,
)

VALID_TASK_YAML = """
task_key: test_task_001
version: 1
dataset_version: v1
name: Test task
category: single_tool
user_request: Query an employee.
initial_state: {}
available_tools:
  - get_employee
expected_state: []
required_events: []
forbidden_events: []
temporal_rules: []
budget:
  max_agent_steps: 5
  max_tool_calls: 2
  max_tokens: 1000
  timeout_seconds: 30
metadata: {}
is_active: true
"""


def test_load_task_file(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(VALID_TASK_YAML, encoding="utf-8")

    task = load_task_file(task_path)

    assert task.task_key == "test_task_001"
    assert task.version == 1
    assert task.available_tools == ["get_employee"]
    assert task.budget.max_agent_steps == 5


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        VALID_TASK_YAML + "\nunknown_field: invalid\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskLoadError, match="validation failed"):
        load_task_file(task_path)


def test_duplicate_task_identity_is_rejected(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"

    first_path.write_text(VALID_TASK_YAML, encoding="utf-8")
    second_path.write_text(VALID_TASK_YAML, encoding="utf-8")

    with pytest.raises(TaskLoadError, match="Duplicate task identity"):
        load_task_directory(tmp_path)
