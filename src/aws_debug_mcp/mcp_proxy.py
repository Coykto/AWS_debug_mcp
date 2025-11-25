"""MCP proxy for wrapping AWS MCP servers."""
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPProxy:
    """Proxy to AWS MCP servers, exposing only selected tools."""

    def __init__(self):
        self.cloudwatch_session: ClientSession | None = None
        self.aws_profile = os.getenv("AWS_PROFILE", "")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")

    @asynccontextmanager
    async def _connect_to_cloudwatch(self):
        """Connect to AWS CloudWatch MCP server."""
        server_params = StdioServerParameters(
            command="uvx",
            args=["awslabs.cloudwatch-mcp-server@latest"],
            env={
                "AWS_PROFILE": self.aws_profile,
                "AWS_REGION": self.aws_region,
                "FASTMCP_LOG_LEVEL": "ERROR",
            },
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def call_cloudwatch_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the CloudWatch MCP server."""
        async with self._connect_to_cloudwatch() as session:
            result = await session.call_tool(tool_name, arguments)
            return result.content


# Global proxy instance
proxy = MCPProxy()
