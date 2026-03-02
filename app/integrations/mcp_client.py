"""Typed MCP client adapter for calling the local ops server over stdio."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPClientError(RuntimeError):
    """Raised when MCP calls fail."""


@dataclass
class MCPOpsClient:
    command: str
    args: list[str]

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        server = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments or {})

                is_error = getattr(result, "isError", False) or getattr(result, "is_error", False)
                structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
                content = getattr(result, "content", []) or []

                texts: list[str] = []
                for item in content:
                    text = getattr(item, "text", None)
                    if text is not None:
                        texts.append(str(text))

                payload: dict[str, Any] = {
                    "is_error": bool(is_error),
                    "texts": texts,
                }
                if structured is not None:
                    payload["structured"] = structured

                if payload["is_error"]:
                    raise MCPClientError("; ".join(texts) or "MCP tool returned an error")

                return payload

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return asyncio.run(self._call_tool_async(tool_name, arguments))
