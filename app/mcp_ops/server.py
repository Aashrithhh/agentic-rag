"""FastMCP server entrypoint for RAG Ops diagnostics."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.mcp_ops.resources import register_resources
from app.mcp_ops.tools import register_tools

mcp = FastMCP("rag-ops-mcp")
register_tools(mcp)
register_resources(mcp)


if __name__ == "__main__":
    mcp.run()
