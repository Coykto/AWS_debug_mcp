"""Main MCP server setup - proxies AWS MCP servers with selective tool exposure."""
import os
from fastmcp import FastMCP
from .mcp_proxy import proxy


# Initialize MCP server
mcp = FastMCP("aws-debug-mcp")


# Get configured tools from environment variable
# Format: comma-separated list like "describe_log_groups,analyze_log_group"
# If not set, expose all tools by default
configured_tools_str = os.getenv("AWS_DEBUG_MCP_TOOLS", "all")
if configured_tools_str.lower() == "all":
    configured_tools = None  # None means expose all
else:
    configured_tools = set(tool.strip() for tool in configured_tools_str.split(",") if tool.strip())


def should_expose_tool(tool_name: str) -> bool:
    """Check if a tool should be exposed based on configuration."""
    if configured_tools is None:
        return True  # Expose all
    return tool_name in configured_tools


# CloudWatch Logs Tools - proxied from awslabs.cloudwatch-mcp-server
if should_expose_tool("describe_log_groups"):
    @mcp.tool()
    async def describe_log_groups(log_group_name_prefix: str = "") -> dict:
        """
        List CloudWatch log groups.

        Args:
            log_group_name_prefix: Filter log groups by prefix (e.g., /aws/lambda/, /ecs/)
        """
        return await proxy.call_cloudwatch_tool(
            "describe_log_groups",
            {"log_group_name_prefix": log_group_name_prefix}
        )


if should_expose_tool("analyze_log_group"):
    @mcp.tool()
    async def analyze_log_group(
        log_group_name: str,
        start_time: str,
        end_time: str,
        filter_pattern: str = ""
    ) -> dict:
        """
        Analyze CloudWatch logs for anomalies, message patterns, and error patterns.

        Args:
            log_group_name: Log group name
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            filter_pattern: Optional filter pattern
        """
        args = {
            "log_group_name": log_group_name,
            "start_time": start_time,
            "end_time": end_time,
        }
        if filter_pattern:
            args["filter_pattern"] = filter_pattern

        return await proxy.call_cloudwatch_tool("analyze_log_group", args)


if should_expose_tool("execute_log_insights_query"):
    @mcp.tool()
    async def execute_log_insights_query(
        log_group_names: list[str],
        query_string: str,
        start_time: str,
        end_time: str,
        limit: int = 100
    ) -> dict:
        """
        Execute CloudWatch Logs Insights query.

        Args:
            log_group_names: List of log group names to query
            query_string: CloudWatch Insights query
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            limit: Maximum results
        """
        return await proxy.call_cloudwatch_tool(
            "execute_log_insights_query",
            {
                "log_group_names": log_group_names,
                "query_string": query_string,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
            }
        )


if should_expose_tool("get_logs_insight_query_results"):
    @mcp.tool()
    async def get_logs_insight_query_results(query_id: str) -> dict:
        """
        Get results from a CloudWatch Logs Insights query.

        Args:
            query_id: Query ID from execute_log_insights_query
        """
        return await proxy.call_cloudwatch_tool(
            "get_logs_insight_query_results",
            {"query_id": query_id}
        )
