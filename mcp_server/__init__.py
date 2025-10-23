"""
Chatterbox MCP Server - Lightweight TTS for LLM agents.
"""

from .server import ChatterboxMCPServer
from .config import ChatterboxConfig

__version__ = "0.1.0"
__all__ = ["ChatterboxMCPServer", "ChatterboxConfig"]
