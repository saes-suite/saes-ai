"""Model Context Protocol (MCP) package for SAES."""

from saes_ai.mcp.server import SAESMCPServer, main
from saes_ai.mcp.tools import SAES_MCP_TOOLS, MCPToolSpec

__all__ = [
    "SAESMCPServer",
    "SAES_MCP_TOOLS",
    "MCPToolSpec",
    "main",
]
