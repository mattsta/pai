"""
MCP (Model Context Protocol) client implementation for PAI.

This module provides integration with MCP servers, allowing PAI to discover
and use tools from external MCP-compliant servers.

Architecture:
    MCPManager
        └── MCPServer (one per configured server)
                └── MCPTool (tools discovered from server)

Usage:
    manager = MCPManager()
    await manager.connect_server("filesystem", command=["npx", "-y", "@anthropic/mcp-server-filesystem"])
    tools = manager.get_all_tools()  # Returns tool schemas for AI
    result = await manager.execute_tool("filesystem__read_file", {"path": "/tmp/test.txt"})
"""

from .client import MCPManager, MCPServer, MCPTool
from .config import MCPConfig, MCPServerConfig

__all__ = ["MCPManager", "MCPServer", "MCPTool", "MCPConfig", "MCPServerConfig"]
