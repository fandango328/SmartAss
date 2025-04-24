#!/usr/bin/env python3

from typing import Dict, Any, Callable
from functools import wraps

class ToolRegistry:
    """Centralized registry for tool handlers with dependency injection."""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._managers = {}
        self.initialized = False

    def register_handler(self, tool_name: str, handler: Callable) -> None:
        """Register a single tool handler."""
        self._handlers[tool_name] = handler

    def register_handlers(self, handlers: Dict[str, Callable]) -> None:
        """Register multiple tool handlers."""
        self._handlers.update(handlers)

    def register_manager(self, name: str, manager: Any) -> None:
        """Register a manager component."""
        self._managers[name] = manager
        
    def get_handler(self, tool_name: str) -> Callable:
        """Get a tool handler by name."""
        return self._handlers.get(tool_name)
        
    def get_manager(self, name: str) -> Any:
        """Get a manager component by name."""
        return self._managers.get(name)

    def initialize(self) -> None:
        """Mark registry as initialized."""
        self.initialized = True

# Global instance
tool_registry = ToolRegistry()