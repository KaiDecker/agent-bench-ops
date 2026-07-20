from app.tools.implementations.employees import GET_EMPLOYEE_TOOL
from app.tools.implementations.tickets import CREATE_TICKET_TOOL
from app.tools.schemas import ToolDefinition


class ToolRegistry:
    """工具定义注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        tool_name = definition.metadata.name

        if tool_name in self._tools:
            raise ValueError(f"Tool is already registered: {tool_name}")

        self._tools[tool_name] = definition

    def get(self, tool_name: str) -> ToolDefinition | None:
        return self._tools.get(tool_name)

    def names(self) -> list[str]:
        return sorted(self._tools)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GET_EMPLOYEE_TOOL)
    registry.register(CREATE_TICKET_TOOL)
    return registry
