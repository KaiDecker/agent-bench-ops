from app.tools.implementations.employees import GET_EMPLOYEE_TOOL
from app.tools.schemas import ToolDefinition


class ToolRegistry:
    """工具定义注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """注册工具，禁止同名覆盖。"""
        tool_name = definition.metadata.name

        if tool_name in self._tools:
            raise ValueError(f"Tool is already registered: {tool_name}")

        self._tools[tool_name] = definition

    def get(self, tool_name: str) -> ToolDefinition | None:
        """根据名称获取工具定义。"""
        return self._tools.get(tool_name)

    def names(self) -> list[str]:
        """返回全部已注册工具名称。"""
        return sorted(self._tools)


def build_default_registry() -> ToolRegistry:
    """创建系统默认工具注册表。"""
    registry = ToolRegistry()
    registry.register(GET_EMPLOYEE_TOOL)
    return registry
