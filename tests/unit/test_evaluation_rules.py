from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.benchmark.schemas import (
    BenchmarkTaskSpec,
)
from app.evaluation.rules import (
    StateAssertion,
    StateExpectation,
    TemporalRule,
    TraceEventRule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASK_FILES = sorted((PROJECT_ROOT / "benchmark_tasks").rglob("*.yaml"))


@pytest.mark.parametrize(
    "task_path",
    TASK_FILES,
)
def test_current_benchmark_tasks_use_strict_rules(
    task_path: Path,
) -> None:
    raw_data = yaml.safe_load(task_path.read_text(encoding="utf-8"))

    task = BenchmarkTaskSpec.model_validate(raw_data)

    assert all(
        isinstance(
            expectation,
            StateExpectation,
        )
        for expectation in task.expected_state
    )

    assert all(
        isinstance(
            event,
            TraceEventRule,
        )
        for event in task.required_events
    )

    assert all(
        isinstance(
            event,
            TraceEventRule,
        )
        for event in task.forbidden_events
    )

    assert all(
        isinstance(
            rule,
            TemporalRule,
        )
        for rule in task.temporal_rules
    )


def test_exists_assertion_rejects_value() -> None:
    with pytest.raises(
        ValidationError,
        match="must not define value",
    ):
        StateAssertion(
            field="resolution",
            operator="exists",
            value=True,
        )


def test_in_assertion_requires_list() -> None:
    with pytest.raises(
        ValidationError,
        match="require value to be a list",
    ):
        StateAssertion(
            field="status",
            operator="in",
            value="open",
        )


def test_eq_assertion_requires_value() -> None:
    with pytest.raises(
        ValidationError,
        match="must define value",
    ):
        StateAssertion(
            field="status",
            operator="eq",
        )


def test_eq_assertion_allows_explicit_null() -> None:
    assertion = StateAssertion(
        field="resolution",
        operator="eq",
        value=None,
    )

    assert assertion.value is None


def test_rule_rejects_unknown_fields() -> None:
    with pytest.raises(
        ValidationError,
    ):
        TraceEventRule.model_validate(
            {
                "event": "tool_succeeded",
                "tool_name": "create_ticket",
                "unexpected": True,
            }
        )


def test_temporal_rule_rejects_replay_event() -> None:
    with pytest.raises(
        ValidationError,
    ):
        TemporalRule.model_validate(
            {
                "first": {
                    "event": "tool_replayed",
                    "tool_name": ("create_ticket"),
                },
                "relation": "before",
                "second": {
                    "event": "tool_succeeded",
                    "tool_name": ("create_ticket"),
                },
            }
        )


def test_temporal_rule_rejects_identical_endpoints() -> None:
    with pytest.raises(
        ValidationError,
        match=("endpoints must not be identical"),
    ):
        TemporalRule.model_validate(
            {
                "first": {
                    "event": "tool_called",
                    "tool_name": ("create_ticket"),
                },
                "relation": "before",
                "second": {
                    "event": "tool_called",
                    "tool_name": ("create_ticket"),
                },
            }
        )
