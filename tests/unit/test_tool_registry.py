import pytest

from app.tools.implementations.employees import (
    GET_EMPLOYEE_TOOL,
)
from app.tools.registry import (
    ToolRegistry,
    build_default_registry,
)


def test_default_registry_contains_get_employee() -> None:
    registry = build_default_registry()

    assert registry.names() == [
        "create_ticket",
        "get_employee",
    ]
    assert registry.get("get_employee") is not None


def test_registry_rejects_duplicate_tool() -> None:
    registry = ToolRegistry()
    registry.register(GET_EMPLOYEE_TOOL)

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(GET_EMPLOYEE_TOOL)
