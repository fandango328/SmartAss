#!/usr/bin/env python3

from typing import Dict, Any, Callable, Optional

class MCPToolRegistry:
    """Centralized registry for MCP server tool handlers ONLY."""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._managers: Dict[str, Any] = {}
        self.initialized = False

    def register_handler(self, tool_name: str, handler: Callable) -> None:
        self._handlers[tool_name] = handler

    def register_handlers(self, handlers: Dict[str, Callable]) -> None:
        self._handlers.update(handlers)

    def register_manager(self, name: str, manager: Any) -> None:
        self._managers[name] = manager
        
    def get_handler(self, tool_name: str) -> Optional[Callable]:
        return self._handlers.get(tool_name)
        
    def get_manager(self, name: str) -> Optional[Any]:
        return self._managers.get(name)

    def initialize(self) -> None:
        self.initialized = True

# Global instance for MCP tools ONLY
mcp_tool_registry = MCPToolRegistry()
