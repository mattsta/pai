"""Configuration models for MCP servers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    model_config = ConfigDict(extra="allow")  # Allow additional fields for extensibility

    name: str = Field(..., description="Unique identifier for this server")
    command: list[str] = Field(
        ..., description="Command and arguments to start the server"
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the server"
    )
    enabled: bool = Field(default=True, description="Whether this server is enabled")
    auto_connect: bool = Field(
        default=True, description="Connect automatically on startup"
    )
    timeout: float = Field(
        default=30.0, description="Connection timeout in seconds"
    )


class MCPConfig(BaseModel):
    """Root configuration for MCP in pai.toml."""

    enabled: bool = Field(default=True, description="Enable MCP support globally")
    servers: list[MCPServerConfig] = Field(
        default_factory=list, description="List of MCP servers to connect to"
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPConfig:
        """Create config from a dictionary (e.g., from TOML)."""
        servers = []
        for server_data in data.get("servers", []):
            servers.append(MCPServerConfig(**server_data))
        return cls(enabled=data.get("enabled", True), servers=servers)
