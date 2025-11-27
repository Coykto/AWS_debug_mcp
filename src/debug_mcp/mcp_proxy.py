"""MCP proxy for wrapping AWS MCP servers."""

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPProxy:
    """Proxy to AWS MCP servers, exposing only selected tools."""

    def __init__(self, aws_profile: str = "", aws_region: str = "us-east-1"):
        self.cloudwatch_session: ClientSession | None = None
        self.aws_profile = aws_profile
        self.aws_region = aws_region

    def _extract_content(self, result_content: list) -> dict | str:
        """
        Extract actual data from MCP result content.

        MCP results are a list of content blocks. This extracts the text
        from the first block and attempts to parse it as JSON.

        Args:
            result_content: List of content blocks from MCP call_tool result

        Returns:
            Parsed JSON dict if the content is JSON, otherwise the raw text string
        """
        if not result_content:
            return {}

        # Get the first content block
        first_block = result_content[0]

        # Extract text based on content block type
        if hasattr(first_block, "text"):
            text = first_block.text
        elif isinstance(first_block, dict) and "text" in first_block:
            text = first_block["text"]
        else:
            # If we can't extract text, return the block as-is
            return first_block

        # Try to parse as JSON
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, return as string
            return text

    def _get_env(self, **overrides: str) -> dict[str, str]:
        """Get environment dict with AWS credentials, merging with current env."""
        env = os.environ.copy()
        env.update(
            {
                "AWS_PROFILE": self.aws_profile,
                "AWS_REGION": self.aws_region,
                **overrides,
            }
        )
        return env

    @asynccontextmanager
    async def _connect_to_cloudwatch(self):
        """Connect to AWS CloudWatch MCP server."""
        server_params = StdioServerParameters(
            command="uvx",
            args=["awslabs.cloudwatch-mcp-server@latest"],
            env=self._get_env(FASTMCP_LOG_LEVEL="ERROR"),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def call_cloudwatch_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the CloudWatch MCP server."""
        async with self._connect_to_cloudwatch() as session:
            result = await session.call_tool(tool_name, arguments)
            return self._extract_content(result.content)

    @asynccontextmanager
    async def _connect_to_ecs(self):
        """Connect to AWS ECS MCP server."""
        server_params = StdioServerParameters(
            command="uvx",
            args=["awslabs.ecs-mcp-server@latest"],
            env=self._get_env(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def call_ecs_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the ECS MCP server."""
        async with self._connect_to_ecs() as session:
            result = await session.call_tool(tool_name, arguments)
            return self._extract_content(result.content)

    @asynccontextmanager
    async def _connect_to_stepfunctions(self):
        """Connect to AWS Step Functions MCP server."""
        server_params = StdioServerParameters(
            command="uvx",
            args=["awslabs.stepfunctions-tool-mcp-server@latest"],
            env=self._get_env(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def call_stepfunctions_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the Step Functions MCP server."""
        async with self._connect_to_stepfunctions() as session:
            result = await session.call_tool(tool_name, arguments)
            return self._extract_content(result.content)


# Global proxy instance - will be initialized by __main__.py with CLI args
proxy: MCPProxy | None = None


def init_proxy(aws_profile: str = "", aws_region: str = "us-east-1"):
    """Initialize the global proxy with AWS credentials from CLI args."""
    global proxy
    proxy = MCPProxy(aws_profile=aws_profile, aws_region=aws_region)
