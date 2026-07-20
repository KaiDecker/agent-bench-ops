import pytest

from app.tools.implementations.employees import (
    GET_EMPLOYEE_TOOL,
)
from app.tools.registry import (
    ToolRegistry,
    build_default_registry,
)


def test_default_registry_contains_expected_tools() -> None:
    registry = build_default_registry()

    assert registry.names() == [
        "create_ticket",
        "get_account",
        "get_employee",
        "get_ticket",
        "list_employee_permissions",
    ]


def test_registry_can_get_each_default_tool() -> None:
    registry = build_default_registry()

    for tool_name in registry.names():
        assert registry.get(tool_name) is not None


def test_registry_rejects_duplicate_tool() -> None:
    registry = ToolRegistry()
    registry.register(GET_EMPLOYEE_TOOL)

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(GET_EMPLOYEE_TOOL)
