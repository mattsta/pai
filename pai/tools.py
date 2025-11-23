import asyncio
import enum
import importlib
import inspect
import json
import logging
import pathlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mcp import MCPManager


@dataclass
class ToolDefinition:
    """Encapsulates a registered tool's function and schema."""

    function: Callable
    schema: dict[str, Any]


@dataclass
class ToolAuditEntry:
    """A single audit log entry for a tool execution."""

    timestamp: str
    tool_name: str
    tool_type: str  # "native" or "mcp"
    arguments: dict[str, Any]
    success: bool
    result_preview: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "arguments": self.arguments,
            "success": self.success,
            "result_preview": self.result_preview,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "session_id": self.session_id,
        }


class ToolAuditLogger:
    """
    Logs tool executions for security auditing and debugging.

    Supports logging to:
    - Memory (for session review)
    - File (JSON lines format)
    - Python logging module

    Usage:
        audit_logger = get_audit_logger()
        audit_logger.enable(log_file="/path/to/audit.jsonl")
        # ... tool executions are automatically logged ...
        entries = audit_logger.get_entries()
    """

    def __init__(self) -> None:
        self._entries: list[ToolAuditEntry] = []
        self._enabled: bool = False
        self._log_file: pathlib.Path | None = None
        self._session_id: str | None = None
        self._logger = logging.getLogger("pai.tools.audit")

    def enable(
        self,
        log_file: str | pathlib.Path | None = None,
        session_id: str | None = None,
    ) -> None:
        """Enable audit logging.

        Args:
            log_file: Path to write audit log (JSON lines format)
            session_id: Session identifier to include in log entries
        """
        self._enabled = True
        self._session_id = session_id
        if log_file:
            self._log_file = pathlib.Path(log_file)
            # Create parent directory if needed
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Tool audit logging enabled: {self._log_file}")

    def disable(self) -> None:
        """Disable audit logging."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if audit logging is enabled."""
        return self._enabled

    def log(
        self,
        tool_name: str,
        tool_type: str,
        arguments: dict[str, Any],
        success: bool,
        result: Any = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log a tool execution.

        Args:
            tool_name: Name of the executed tool
            tool_type: "native" or "mcp"
            arguments: Arguments passed to the tool
            success: Whether execution succeeded
            result: Tool result (will be truncated for preview)
            error: Error message if failed
            duration_ms: Execution time in milliseconds
        """
        if not self._enabled:
            return

        # Create preview of result (truncate if too long)
        result_preview = None
        if result is not None:
            result_str = str(result)
            result_preview = result_str[:500] + "..." if len(result_str) > 500 else result_str

        # Sanitize arguments (remove sensitive data patterns)
        sanitized_args = self._sanitize_arguments(arguments)

        entry = ToolAuditEntry(
            timestamp=datetime.now().isoformat(),
            tool_name=tool_name,
            tool_type=tool_type,
            arguments=sanitized_args,
            success=success,
            result_preview=result_preview,
            error=error,
            duration_ms=duration_ms,
            session_id=self._session_id,
        )

        self._entries.append(entry)

        # Log to Python logging
        log_msg = f"Tool: {tool_name} ({tool_type}) - {'OK' if success else 'FAILED'}"
        if success:
            self._logger.info(log_msg)
        else:
            self._logger.warning(f"{log_msg}: {error}")

        # Write to file if configured
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception as e:
                self._logger.error(f"Failed to write audit log: {e}")

    def _sanitize_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """Sanitize arguments to remove potentially sensitive data."""
        sensitive_patterns = ("password", "secret", "token", "key", "credential", "auth")
        sanitized = {}

        for key, value in args.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in sensitive_patterns):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                # Truncate very long strings
                sanitized[key] = value[:1000] + f"... [truncated, {len(value)} chars total]"
            else:
                sanitized[key] = value

        return sanitized

    def get_entries(self, limit: int | None = None) -> list[ToolAuditEntry]:
        """Get audit log entries.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of audit entries
        """
        if limit:
            return self._entries[-limit:]
        return self._entries.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of tool usage."""
        if not self._entries:
            return {"total_executions": 0, "tools": {}}

        tool_stats: dict[str, dict[str, int]] = {}
        for entry in self._entries:
            if entry.tool_name not in tool_stats:
                tool_stats[entry.tool_name] = {"success": 0, "failed": 0}
            if entry.success:
                tool_stats[entry.tool_name]["success"] += 1
            else:
                tool_stats[entry.tool_name]["failed"] += 1

        return {
            "total_executions": len(self._entries),
            "successful": sum(1 for e in self._entries if e.success),
            "failed": sum(1 for e in self._entries if not e.success),
            "tools": tool_stats,
        }

    def clear(self) -> None:
        """Clear all stored entries."""
        self._entries.clear()


# Global audit logger instance
_audit_logger: ToolAuditLogger | None = None


def get_audit_logger() -> ToolAuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = ToolAuditLogger()
    return _audit_logger


TOOL_REGISTRY: dict[str, ToolDefinition] = {}

# Global MCP manager instance (initialized lazily)
_mcp_manager: "MCPManager | None" = None


def get_mcp_manager() -> "MCPManager | None":
    """Get the global MCP manager instance."""
    return _mcp_manager


def set_mcp_manager(manager: "MCPManager") -> None:
    """Set the global MCP manager instance."""
    global _mcp_manager
    _mcp_manager = manager


class ToolError(Exception):
    """Base exception for tool-related errors."""


class ToolNotFound(ToolError):
    """Raised when a tool is not found in the registry."""


class ToolArgumentError(ToolError):
    """Raised on errors related to tool arguments (e.g., validation, conversion)."""


class TemperatureUnit(enum.Enum):
    """Enumeration for temperature units."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


def _generate_schema_for_function(func: Callable) -> dict:
    """Introspects a function to generate an OpenAI-compatible tool schema."""
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    param_docs = {}
    if "Args:" in docstring:
        args_section = docstring.split("Args:")[1].split("Returns:")[0]
        for line in args_section.strip().split("\n"):
            if ":" in line:
                param_part, desc = line.split(":", 1)
                # The name is the first word in the 'name (type)' part.
                param_name = param_part.strip().split(" ")[0]
                param_docs[param_name] = desc.strip()

    type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        param_type = param.annotation
        property_details = None
        if inspect.isclass(param_type) and issubclass(param_type, enum.Enum):
            property_details = {
                "type": "string",
                "description": param_docs.get(name, ""),
                "enum": [e.value for e in param_type],
            }
        elif param_type in type_mapping:
            property_details = {
                "type": type_mapping[param_type],
                "description": param_docs.get(name, ""),
            }

        if property_details:
            properties[name] = property_details
            if param.default is inspect.Parameter.empty:
                required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring.split("\n")[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool(func: Callable) -> Callable:
    """Decorator to register a function as a tool the AI can use."""
    tool_schema = _generate_schema_for_function(func)
    TOOL_REGISTRY[func.__name__] = ToolDefinition(function=func, schema=tool_schema)
    return func


def get_tool_schemas(include_mcp: bool = True) -> list[dict[str, Any]]:
    """Get all tool schemas in OpenAI-compatible format.

    Args:
        include_mcp: Whether to include MCP tools (default True).

    Returns:
        List of tool schemas for all available tools.
    """
    schemas = [t.schema for t in TOOL_REGISTRY.values()] if TOOL_REGISTRY else []

    # Add MCP tools if manager is available
    if include_mcp and _mcp_manager is not None:
        schemas.extend(_mcp_manager.get_all_tools())

    return schemas


def is_mcp_tool(name: str) -> bool:
    """Check if a tool name refers to an MCP tool."""
    return name.startswith("mcp__")


def get_tool_manifest(include_mcp: bool = True) -> str:
    """Generates a text manifest of all registered tools for legacy models.

    Args:
        include_mcp: Whether to include MCP tools (default True).

    Returns:
        Text manifest of all available tools.
    """
    lines = []

    # Native tools
    if TOOL_REGISTRY:
        lines.append("Native Tools:")
        for name, info in TOOL_REGISTRY.items():
            schema = info.schema["function"]
            lines.append(f"- Name: {name}")
            lines.append(f"  Description: {schema['description']}")
            if properties := schema["parameters"]["properties"]:
                lines.append("  Arguments:")
                for arg_name, details in properties.items():
                    arg_type = details.get("type", "any")
                    enum_values = details.get("enum")
                    if enum_values:
                        arg_type = f"string (enum: {', '.join(enum_values)})"
                    lines.append(
                        f"    - {arg_name} ({arg_type}): {details.get('description', '')}"
                    )
            else:
                lines.append("  Arguments: None")
            lines.append("")

    # MCP tools
    if include_mcp and _mcp_manager is not None:
        mcp_tools = _mcp_manager.get_all_tools()
        if mcp_tools:
            lines.append("MCP Tools:")
            for tool_schema in mcp_tools:
                func = tool_schema["function"]
                lines.append(f"- Name: {func['name']}")
                lines.append(f"  Description: {func['description']}")
                if properties := func["parameters"].get("properties", {}):
                    lines.append("  Arguments:")
                    for arg_name, details in properties.items():
                        arg_type = details.get("type", "any")
                        lines.append(
                            f"    - {arg_name} ({arg_type}): {details.get('description', '')}"
                        )
                else:
                    lines.append("  Arguments: None")
                lines.append("")

    if not lines:
        return "No tools available."

    return "You have access to the following tools:\n\n" + "\n".join(lines)


async def execute_tool(name: str, args: dict) -> Any:
    """Execute a tool by name with the given arguments.

    Handles both native tools and MCP tools (prefixed with 'mcp__').
    Automatically logs executions to the audit log when enabled.

    Args:
        name: The tool name (or qualified MCP tool name).
        args: Dictionary of tool arguments.

    Returns:
        The tool execution result.

    Raises:
        ToolNotFound: If the tool is not found.
        ToolError: If execution fails.
    """
    import time

    audit = get_audit_logger()
    start_time = time.perf_counter()

    # Route MCP tools to the MCP manager
    if is_mcp_tool(name):
        if _mcp_manager is None:
            raise ToolNotFound(f"MCP tool '{name}' requested but MCP is not enabled.")
        try:
            result = await _mcp_manager.execute_tool(name, args)
            duration_ms = (time.perf_counter() - start_time) * 1000
            audit.log(name, "mcp", args, success=True, result=result, duration_ms=duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            audit.log(name, "mcp", args, success=False, error=str(e), duration_ms=duration_ms)
            raise ToolError(f"MCP tool '{name}' failed: {e}") from e

    # Handle native tools
    if name not in TOOL_REGISTRY:
        raise ToolNotFound(f"Tool '{name}' not found.")

    tool_def = TOOL_REGISTRY[name]
    func = tool_def.function
    sig = inspect.signature(func)

    # Convert arguments to their correct types, including Enums
    converted_args = {}
    try:
        for param_name, param_obj in sig.parameters.items():
            if param_name in args:
                arg_value = args[param_name]
                param_type = param_obj.annotation
                if inspect.isclass(param_type) and issubclass(param_type, enum.Enum):
                    converted_args[param_name] = param_type(arg_value)
                else:
                    converted_args[param_name] = arg_value
            # Let Python handle missing args with default values
        if inspect.iscoroutinefunction(func):
            result = await func(**converted_args)
        else:
            # Run synchronous functions in a separate thread to avoid blocking the event loop.
            result = await asyncio.to_thread(func, **converted_args)

        duration_ms = (time.perf_counter() - start_time) * 1000
        audit.log(name, "native", args, success=True, result=result, duration_ms=duration_ms)
        return result

    except ValueError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        audit.log(name, "native", args, success=False, error=str(e), duration_ms=duration_ms)
        # Specifically for enum conversion errors
        raise ToolArgumentError(f"Invalid argument value for tool '{name}': {e}") from e
    except TypeError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        audit.log(name, "native", args, success=False, error=str(e), duration_ms=duration_ms)
        # Catches missing required arguments.
        raise ToolArgumentError(
            f"Missing or invalid arguments for tool '{name}': {e}"
        ) from e
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        audit.log(name, "native", args, success=False, error=str(e), duration_ms=duration_ms)
        raise ToolError(f"Error executing tool '{name}' with args {args}: {e}") from e


@tool
def get_current_weather(
    location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS
) -> str:
    """Gets the current weather for a specified location.

    Args:
        location (str): The city and state, e.g., "San Francisco, CA".
        unit (TemperatureUnit): The unit of temperature.
    """
    weather_data = {}
    if "tokyo" in location.lower():
        temp = "15" if unit == TemperatureUnit.CELSIUS else "59"
        weather_data = {"location": "Tokyo", "temperature": temp, "condition": "Cloudy"}
    elif "paris" in location.lower():
        temp = "22" if unit == TemperatureUnit.CELSIUS else "72"
        weather_data = {"location": "Paris", "temperature": temp, "condition": "Sunny"}
    else:
        temp = "30" if unit == TemperatureUnit.CELSIUS else "86"
        weather_data = {"location": location, "temperature": temp, "condition": "Hot"}

    result = {"status": "success", "result": weather_data}
    return json.dumps(result, indent=2)


def load_tools_from_directory(directory: str, printer: Callable = print):
    """Dynamically loads tools from Python files in a given directory."""
    path = pathlib.Path(directory)
    if not path.is_dir():
        # Don't print an error if the default dir doesn't exist.
        return

    # To enable relative imports, we treat the tool directory as a package.
    # Its parent directory must be on the Python path.
    parent_dir = str(path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # The directory must contain an __init__.py to be a package.
    if not (path / "__init__.py").exists():
        printer(
            f"  ⚠️  Warning: Skipping tool directory '{path.name}'. It is not a Python package (missing __init__.py)."
        )
        return

    printer(f"🔎 Loading custom tools from: {path}")
    found = False
    for file_path in path.glob("*.py"):
        # Skip __init__ files and any other private-like files.
        if file_path.stem.startswith("__init__") or file_path.stem.startswith("_"):
            continue
        try:
            # Construct the full module name for a proper import.
            module_name = f"{path.name}.{file_path.stem}"
            importlib.import_module(module_name)
            printer(f"  ✅ Loaded custom tool module: {file_path.name}")
            found = True
        except Exception as e:
            printer(f"  ❌ Failed to load {file_path.name}: {e}")
    if not found:
        printer("  (No custom tools found)")
