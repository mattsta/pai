"""Tests for MCP (Model Context Protocol) functionality."""

import pytest

from pai.mcp import MCPConfig, MCPManager, MCPServerConfig, MCPTool
from pai.mcp.client import MCPServer, ServerStatus


class TestMCPConfig:
    """Tests for MCP configuration models."""

    def test_server_config_defaults(self):
        """Test MCPServerConfig default values."""
        config = MCPServerConfig(
            name="test-server",
            command=["python", "-m", "mcp_server"],
        )
        assert config.name == "test-server"
        assert config.command == ["python", "-m", "mcp_server"]
        assert config.env == {}
        assert config.enabled is True
        assert config.auto_connect is True
        assert config.timeout == 30.0

    def test_server_config_custom_values(self):
        """Test MCPServerConfig with custom values."""
        config = MCPServerConfig(
            name="custom-server",
            command=["node", "server.js"],
            env={"API_KEY": "secret"},
            enabled=False,
            auto_connect=False,
            timeout=60.0,
        )
        assert config.name == "custom-server"
        assert config.env == {"API_KEY": "secret"}
        assert config.enabled is False
        assert config.auto_connect is False
        assert config.timeout == 60.0

    def test_mcp_config_defaults(self):
        """Test MCPConfig default values."""
        config = MCPConfig()
        assert config.enabled is True
        assert config.servers == []

    def test_mcp_config_with_servers(self):
        """Test MCPConfig with multiple servers."""
        server1 = MCPServerConfig(name="server1", command=["cmd1"])
        server2 = MCPServerConfig(name="server2", command=["cmd2"])
        config = MCPConfig(servers=[server1, server2])
        assert len(config.servers) == 2
        assert config.servers[0].name == "server1"
        assert config.servers[1].name == "server2"


class TestMCPTool:
    """Tests for MCPTool."""

    def test_tool_creation(self):
        """Test creating an MCPTool."""
        tool = MCPTool(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            server_name="filesystem",
        )
        assert tool.name == "read_file"
        assert tool.description == "Read contents of a file"
        assert tool.server_name == "filesystem"

    def test_qualified_name(self):
        """Test qualified name generation."""
        tool = MCPTool(
            name="read_file",
            description="Read file",
            input_schema={},
            server_name="filesystem",
        )
        assert tool.qualified_name == "mcp__filesystem__read_file"

    def test_to_openai_schema(self):
        """Test conversion to OpenAI schema format."""
        tool = MCPTool(
            name="search",
            description="Search for text",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            server_name="web",
        )
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mcp__web__search"
        assert "[MCP:web]" in schema["function"]["description"]
        assert schema["function"]["parameters"]["type"] == "object"


class TestMCPServer:
    """Tests for MCPServer state management."""

    def test_server_initial_state(self):
        """Test MCPServer initial state."""
        config = MCPServerConfig(name="test", command=["echo"])
        server = MCPServer(config=config)

        assert server.name == "test"
        assert server.status == ServerStatus.DISCONNECTED
        assert server.tools == []
        assert server.is_connected is False

    def test_server_connected_state(self):
        """Test MCPServer connected state check."""
        config = MCPServerConfig(name="test", command=["echo"])
        server = MCPServer(config=config)

        # Not connected without process
        assert server.is_connected is False

        # Manually set status (simulating connection)
        server.status = ServerStatus.CONNECTED
        # Still not connected because _process is None
        assert server.is_connected is False


class TestMCPManager:
    """Tests for MCPManager."""

    def test_manager_initial_state(self):
        """Test MCPManager initial state."""
        manager = MCPManager()
        assert manager.servers == {}
        assert manager.connected_servers == []

    def test_load_config(self):
        """Test loading configuration."""
        manager = MCPManager()
        config = MCPConfig(
            servers=[
                MCPServerConfig(name="server1", command=["cmd1"]),
                MCPServerConfig(name="server2", command=["cmd2"]),
            ]
        )
        manager.load_config(config)

        assert "server1" in manager.servers
        assert "server2" in manager.servers
        assert len(manager.servers) == 2

    def test_get_tool_by_name_not_found(self):
        """Test get_tool_by_name returns None for unknown tools."""
        manager = MCPManager()
        assert manager.get_tool_by_name("mcp__unknown__tool") is None
        assert manager.get_tool_by_name("not_mcp_tool") is None
        assert manager.get_tool_by_name("mcp__invalid") is None

    def test_get_all_tools_empty(self):
        """Test get_all_tools with no connected servers."""
        manager = MCPManager()
        assert manager.get_all_tools() == []

    def test_status_summary_no_servers(self):
        """Test status summary with no servers."""
        manager = MCPManager()
        assert manager.get_status_summary() == "No MCP servers configured"

    def test_status_summary_with_servers(self):
        """Test status summary with servers."""
        manager = MCPManager()
        config = MCPConfig(servers=[MCPServerConfig(name="test", command=["cmd"])])
        manager.load_config(config)
        summary = manager.get_status_summary()

        assert "test" in summary
        assert "disconnected" in summary


class TestMCPToolIntegration:
    """Integration tests for MCP tools with the tool system."""

    def test_is_mcp_tool(self):
        """Test MCP tool name detection."""
        from pai.tools import is_mcp_tool

        assert is_mcp_tool("mcp__server__tool") is True
        assert is_mcp_tool("mcp__nested__server__tool") is True
        assert is_mcp_tool("regular_tool") is False
        assert is_mcp_tool("mcp_without_double_underscore") is False

    def test_mcp_manager_global_instance(self):
        """Test getting/setting global MCP manager."""
        from pai.tools import get_mcp_manager, set_mcp_manager

        # Initially None
        original = get_mcp_manager()

        # Set a new manager
        new_manager = MCPManager()
        set_mcp_manager(new_manager)
        assert get_mcp_manager() is new_manager

        # Restore original state
        if original is not None:
            set_mcp_manager(original)

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_no_manager(self):
        """Test executing MCP tool when manager is not set."""
        from pai.tools import ToolNotFound, execute_tool, set_mcp_manager

        # Ensure no manager is set
        set_mcp_manager(None)  # type: ignore

        with pytest.raises(ToolNotFound) as exc_info:
            await execute_tool("mcp__server__tool", {})
        assert "MCP is not enabled" in str(exc_info.value)
