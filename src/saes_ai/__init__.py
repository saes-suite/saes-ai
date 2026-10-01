"""SAES-AI: AI Agent, Statistical Chat Assistant, and Model Context Protocol (MCP) server for SAES."""

from saes_ai.chat.agent import StatisticalChatAgent
from saes_ai.chat.context import build_analysis_context
from saes_ai.chat.prompts import STATISTICAL_SYSTEM_PROMPT
from saes_ai.mcp.server import SAESMCPServer
from saes_ai.mcp.tools import SAES_MCP_TOOLS, MCPToolSpec
from saes_ai.server.routes import router as chat_router

__version__ = "0.1.0"

__all__ = [
    "StatisticalChatAgent",
    "build_analysis_context",
    "STATISTICAL_SYSTEM_PROMPT",
    "SAESMCPServer",
    "SAES_MCP_TOOLS",
    "MCPToolSpec",
    "chat_router",
]
