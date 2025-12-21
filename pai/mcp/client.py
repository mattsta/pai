"""
MCP client implementation using stdio transport.

This module implements the MCP (Model Context Protocol) client that can:
- Connect to MCP servers via subprocess (stdio transport)
- Discover available tools from servers
- Execute tools and return results
- Manage server lifecycle
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import MCPConfig, MCPServerConfig

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors."""


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""


class MCPToolError(MCPError):
    """Raised when tool execution fails."""


class ServerStatus(Enum):
    """Status of an MCP server connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPTool:
    """Represents a tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    @property
    def qualified_name(self) -> str:
        """Returns the fully qualified tool name (server__tool)."""
        return f"mcp__{self.server_name}__{self.name}"

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": f"[MCP:{self.server_name}] {self.description}",
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPServer:
    """
    Manages a connection to a single MCP server.

    Uses JSON-RPC 2.0 over stdio to communicate with the server process.
    """

    config: MCPServerConfig
    status: ServerStatus = ServerStatus.DISCONNECTED
    tools: list[MCPTool] = field(default_factory=list)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _request_id: int = field(default=0, repr=False)
    _pending_requests: dict[int, asyncio.Future[Any]] = field(
        default_factory=dict, repr=False
    )
    _read_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _stderr_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _server_capabilities: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        """Server name from config."""
        return self.config.name

    @property
    def is_connected(self) -> bool:
        """Check if server is connected and running."""
        return (
            self.status == ServerStatus.CONNECTED
            and self._process is not None
            and self._process.returncode is None
        )

    async def connect(self) -> None:
        """Start the MCP server process and establish connection."""
        if self.is_connected:
            logger.info(f"MCP server '{self.name}' already connected")
            return

        self.status = ServerStatus.CONNECTING
        logger.info(
            f"Starting MCP server '{self.name}': {' '.join(self.config.command)}"
        )

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.env)

            # Start the server process
            self._process = await asyncio.create_subprocess_exec(
                *self.config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Start reading responses and stderr
            self._read_task = asyncio.create_task(self._read_responses())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            # Initialize the connection
            await self._initialize()

            # Discover tools
            await self._discover_tools()

            self.status = ServerStatus.CONNECTED
            logger.info(
                f"MCP server '{self.name}' connected with {len(self.tools)} tools"
            )

            # Invalidate tool schema cache since new tools are available
            try:
                from pai.tools import invalidate_tool_cache

                invalidate_tool_cache()
            except ImportError:
                pass  # Tools module not available

        except Exception as e:
            self.status = ServerStatus.ERROR
            logger.error(f"Failed to connect to MCP server '{self.name}': {e}")
            await self.disconnect()
            raise MCPConnectionError(f"Failed to connect to '{self.name}': {e}") from e

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        # Cancel read tasks
        for task in [self._read_task, self._stderr_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._read_task = None
        self._stderr_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                self._process.kill()
            except Exception:
                pass
            self._process = None

        self.tools.clear()
        self._pending_requests.clear()
        self.status = ServerStatus.DISCONNECTED
        logger.info(f"MCP server '{self.name}' disconnected")

        # Invalidate tool schema cache since tools are no longer available
        try:
            from pai.tools import invalidate_tool_cache

            invalidate_tool_cache()
        except ImportError:
            pass  # Tools module not available

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request and wait for response."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError(f"Server '{self.name}' not connected")

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Create a future for this request
        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[request_id] = future

        try:
            # Send the request
            request_str = json.dumps(request) + "\n"
            self._process.stdin.write(request_str.encode())
            await self._process.stdin.drain()

            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=self.config.timeout)
            return result

        except TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Request '{method}' timed out") from None
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Request '{method}' failed: {e}") from e

    async def _read_responses(self) -> None:
        """Read responses from the server process."""
        if not self._process or not self._process.stdout:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    response = json.loads(line.decode().strip())
                    request_id = response.get("id")

                    if request_id and request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if "error" in response:
                            future.set_exception(
                                MCPError(
                                    response["error"].get("message", "Unknown error")
                                )
                            )
                        else:
                            future.set_result(response.get("result"))
                    elif "method" in response:
                        # Handle notifications from server
                        logger.debug(f"MCP notification: {response['method']}")

                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON from MCP server: {line.decode().strip()}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error reading from MCP server '{self.name}': {e}")
            self.status = ServerStatus.ERROR
        finally:
            # Clean up any pending requests when the reader exits
            for request_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(
                        MCPConnectionError(f"Server '{self.name}' connection closed")
                    )
            self._pending_requests.clear()

    async def _read_stderr(self) -> None:
        """Read and log stderr from the server process."""
        if not self._process or not self._process.stderr:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                stderr_msg = line.decode().strip()
                if stderr_msg:
                    logger.warning(f"MCP server '{self.name}' stderr: {stderr_msg}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Error reading stderr from '{self.name}': {e}")

    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "pai", "version": "0.3.0"},
            },
        )
        # Store server capabilities for future reference
        self._server_capabilities = result.get("capabilities", {}) if result else {}
        logger.debug(f"MCP server '{self.name}' initialized: {result}")

        # Send initialized notification
        if self._process and self._process.stdin:
            notification = (
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
                + "\n"
            )
            self._process.stdin.write(notification.encode())
            await self._process.stdin.drain()

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        result = await self._send_request("tools/list")
        self.tools.clear()

        for tool_data in result.get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get(
                    "inputSchema", {"type": "object", "properties": {}}
                ),
                server_name=self.name,
            )
            self.tools.append(tool)

        logger.info(f"Discovered {len(self.tools)} tools from '{self.name}'")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool on this server.

        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments as a dictionary

        Returns:
            Tool result as a string

        Raises:
            MCPConnectionError: If server is not connected
            MCPToolError: If tool execution fails or returns an error
        """
        if not self.is_connected:
            raise MCPConnectionError(f"Server '{self.name}' not connected")

        result = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        # Check for isError flag in response
        if result.get("isError"):
            error_msg = "Tool execution failed"
            content = result.get("content", [])
            if content and content[0].get("type") == "text":
                error_msg = content[0].get("text", error_msg)
            raise MCPToolError(f"Tool '{tool_name}' error: {error_msg}")

        # MCP returns content array
        content = result.get("content", [])
        if content and len(content) > 0:
            first_content = content[0]
            if first_content.get("type") == "text":
                text_result: str = first_content.get("text", "")
                return text_result
            # Handle other content types (image, resource, etc.)
            return json.dumps(first_content)
        return json.dumps(result)


class MCPManager:
    """
    Manages multiple MCP server connections.

    This is the main entry point for MCP functionality in PAI.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._config: MCPConfig | None = None

    @property
    def servers(self) -> dict[str, MCPServer]:
        """Dictionary of managed servers."""
        return self._servers

    @property
    def connected_servers(self) -> list[MCPServer]:
        """List of currently connected servers."""
        return [s for s in self._servers.values() if s.is_connected]

    def load_config(self, config: MCPConfig) -> None:
        """Load MCP configuration."""
        self._config = config
        for server_config in config.servers:
            if server_config.name not in self._servers:
                self._servers[server_config.name] = MCPServer(config=server_config)

    async def connect_all(self, printer: Any = print) -> None:
        """Connect to all configured servers that have auto_connect enabled."""
        if not self._config or not self._config.enabled:
            return

        for server in self._servers.values():
            if server.config.enabled and server.config.auto_connect:
                try:
                    await server.connect()
                    printer(
                        f"  ✅ MCP server '{server.name}' connected ({len(server.tools)} tools)"
                    )
                except MCPConnectionError as e:
                    printer(f"  ❌ MCP server '{server.name}' failed: {e}")

    async def connect_server(
        self,
        name: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> MCPServer:
        """
        Connect to a specific server, optionally creating it on-the-fly.

        Args:
            name: Server name/identifier
            command: Command to start server (if not already configured)
            env: Environment variables for the server

        Returns:
            The connected MCPServer instance
        """
        if name in self._servers:
            server = self._servers[name]
            if not server.is_connected:
                await server.connect()
            return server

        if not command:
            raise MCPError(f"Server '{name}' not found and no command provided")

        # Create new server config
        config = MCPServerConfig(
            name=name,
            command=command,
            env=env or {},
        )
        server = MCPServer(config=config)
        self._servers[name] = server
        await server.connect()
        return server

    async def disconnect_server(self, name: str) -> None:
        """Disconnect a specific server."""
        if name in self._servers:
            await self._servers[name].disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for server in self._servers.values():
            await server.disconnect()

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools from all connected servers in OpenAI schema format."""
        tools = []
        for server in self.connected_servers:
            for tool in server.tools:
                tools.append(tool.to_openai_schema())
        return tools

    def get_tool_by_name(self, qualified_name: str) -> tuple[MCPServer, MCPTool] | None:
        """
        Find a tool by its qualified name.

        Args:
            qualified_name: Full tool name (mcp__server__tool)

        Returns:
            Tuple of (server, tool) or None if not found
        """
        if not qualified_name.startswith("mcp__"):
            return None

        parts = qualified_name.split("__", 2)
        if len(parts) != 3:
            return None

        _, server_name, tool_name = parts
        server = self._servers.get(server_name)
        if not server or not server.is_connected:
            return None

        for tool in server.tools:
            if tool.name == tool_name:
                return (server, tool)
        return None

    async def execute_tool(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute an MCP tool by its qualified name.

        Args:
            qualified_name: Full tool name (mcp__server__tool)
            arguments: Tool arguments

        Returns:
            Tool execution result as string
        """
        result = self.get_tool_by_name(qualified_name)
        if not result:
            raise MCPToolError(f"MCP tool '{qualified_name}' not found")

        server, tool = result
        return await server.call_tool(tool.name, arguments)

    def get_status_summary(self) -> str:
        """Get a summary of all server statuses."""
        if not self._servers:
            return "No MCP servers configured"

        lines = []
        for name, server in self._servers.items():
            status = server.status.value
            tool_count = len(server.tools) if server.is_connected else 0
            lines.append(f"  {name}: {status} ({tool_count} tools)")
        return "\n".join(lines)
