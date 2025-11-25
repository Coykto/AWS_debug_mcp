# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Debug MCP is an open-source MCP (Model Context Protocol) server for debugging AWS distributed systems (Lambda, Step Functions, ECS) directly from Claude Code or any MCP client. The project aims to eliminate context switching between AWS console interfaces by bringing AWS debugging capabilities into AI coding assistants.

**Current Status**: Phase 1 MVP Complete - CloudWatch Logs tools are implemented and working. The server is installable locally via `uv run aws-debug-mcp` and can be used in other projects via the local configuration or eventually via `uvx --from git+https://github.com/...`.

## Development Commands

### Environment Setup
```bash
# Install dependencies (when implemented)
uv sync

# Install with dev dependencies
uv sync --dev

# Run the MCP server locally
uv run aws-debug-mcp

# Run in development mode from project root
uv run python -m aws_debug_mcp
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/aws_debug_mcp

# Run specific test file
uv run pytest tests/test_cloudwatch_logs.py
```

### Code Quality
```bash
# Lint code with ruff
uv run ruff check src/

# Format code with ruff
uv run ruff format src/

# Type checking with mypy
uv run mypy src/
```

### Manual Testing
```bash
# Test AWS client factory in Python REPL
uv run python
>>> from aws_debug_mcp.aws.client_factory import AWSClientFactory
>>> from aws_debug_mcp.tools.cloudwatch_logs import CloudWatchLogsTools
>>> factory = AWSClientFactory()
>>> tools = CloudWatchLogsTools(factory)
>>> tools.describe_log_groups(prefix="/aws/lambda/")
```

## Architecture

### Project Structure
The project follows src-layout with a simple proxy design:

```
src/aws_debug_mcp/
├── server.py              # Main MCP server with proxied tool registrations
├── mcp_proxy.py           # MCP proxy client for connecting to AWS MCPs
├── __main__.py            # Entry point for CLI execution
├── tools/                 # Future: additional tool modules if needed
└── utils/                 # Future: shared utilities if needed
```

**Key Files**:
- `server.py` - Defines which tools to expose using FastMCP decorators. Each tool forwards to the upstream AWS MCP.
- `mcp_proxy.py` - Manages subprocess connections to AWS MCP servers (CloudWatch, Step Functions, etc.) and forwards tool calls.
- `__main__.py` - CLI entry point that starts the FastMCP server.

### Key Architectural Decisions

**MCP Proxy Pattern**: This server acts as a **proxy/gateway** to AWS MCP servers, exposing only the tools you need while hiding the rest. It does NOT reimplement AWS functionality - it wraps existing AWS MCP servers.

**How It Works**:
1. Spawns AWS MCP servers (e.g., awslabs.cloudwatch-mcp-server) as subprocesses
2. Uses MCP client SDK to communicate with upstream servers
3. Exposes only selected tools through proxied calls
4. Hides all other tools from the client

**Why This Approach**:
- No need to reimplement AWS APIs with boto3
- AWS maintains the underlying functionality
- Easy to add/remove tools by simply adding/removing proxy functions
- Keeps tool list minimal and focused on debugging workflows

**AWS Authentication**: Environment-based using `AWS_PROFILE` and `AWS_REGION` variables passed through to upstream AWS MCP servers.

**Tool Filtering**: Only expose debugging tools needed for daily work. Currently exposing from CloudWatch MCP:
- `describe_log_groups` - List log groups
- `analyze_log_group` - Analyze logs for patterns/anomalies
- `execute_log_insights_query` - Run Insights queries
- `get_logs_insight_query_results` - Get query results

**Read-Only Design**: All tools are read-only debugging operations. No CRUD or write operations to maintain safety and simplicity.

### Installation Model

Designed to be installable via uvx directly from GitHub:
```bash
uvx --from git+https://github.com/username/aws-debug-mcp aws-debug-mcp
```

This pattern mimics the "serena" MCP server, making it easy for teams to install without package registry publishing.

### Configuration in Claude Code

Users add the server to their MCP configuration:
```json
{
  "mcpServers": {
    "aws-debug-mcp": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/username/aws-debug-mcp", "aws-debug-mcp"],
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

## Implementation Phases

### Phase 1: MVP (CloudWatch Logs) - ✅ COMPLETE
The MVP proxies selected CloudWatch Logs tools from awslabs.cloudwatch-mcp-server:
- `describe_log_groups` - List available log groups with optional prefix filtering
- `analyze_log_group` - Analyze logs for anomalies and patterns
- `execute_log_insights_query` - Execute CloudWatch Insights queries
- `get_logs_insight_query_results` - Get query results

Implementation pattern (to add more tools):
1. Identify the tool you want from the upstream AWS MCP server
2. Add a proxy function in `server.py` with `@mcp.tool()` decorator
3. Call `await proxy.call_cloudwatch_tool(tool_name, arguments)`
4. The proxy handles subprocess spawning and communication

### Phase 2: Expand Services
- Step Functions execution details and history
- ECS task logs and failures
- Lambda invocation logs
- Time-based correlation across services

### Phase 3: Polish
- Comprehensive test coverage
- Enhanced error handling and validation
- Community documentation and examples

## Adding New Tools

To expose additional tools from AWS MCP servers:

### 1. Find the tool in the upstream AWS MCP
Check the AWS MCP documentation for available tools:
- [CloudWatch MCP Server](https://awslabs.github.io/mcp/servers/cloudwatch-mcp-server)
- [Step Functions MCP Server](https://awslabs.github.io/mcp/servers/step-functions-mcp-server)
- [Other AWS MCPs](https://awslabs.github.io/mcp/)

### 2. Add proxy function in server.py
```python
@mcp.tool()
async def my_new_tool(arg1: str, arg2: int) -> dict:
    """Tool description for Claude."""
    return await proxy.call_cloudwatch_tool(
        "upstream_tool_name",
        {"arg1": arg1, "arg2": arg2}
    )
```

### 3. For new AWS services (Step Functions, ECS, etc.)
Add a new proxy method in `mcp_proxy.py`:
```python
async def call_stepfunctions_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call a tool on the Step Functions MCP server."""
    async with self._connect_to_stepfunctions() as session:
        result = await session.call_tool(tool_name, arguments)
        return result.content
```

## Dependencies

**Core**:
- `fastmcp>=0.2.0` - MCP server framework (provides tool decoration and server hosting)
- `mcp>=1.0.0` - MCP SDK (provides client for connecting to upstream AWS MCPs)
- `pydantic>=2.0.0` - Data validation

**Upstream AWS MCPs** (spawned as subprocesses):
- `awslabs.cloudwatch-mcp-server` - CloudWatch Logs, Metrics, Alarms
- `awslabs.step-functions-mcp-server` - Step Functions (Phase 2)
- `awslabs.ecs-mcp-server` - ECS tasks and services (Phase 2)

**Development**:
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `ruff>=0.1.0` - Linting and formatting
- `mypy>=1.0.0` - Static type checking

**Package Management**: uv (modern Python package manager)

## Testing Strategy

### Unit Tests
Mock boto3 clients to test tool logic without AWS API calls. Focus on:
- Time range parsing edge cases
- Response formatting
- Error handling

### Integration Tests (Manual)
Run against real AWS account (dev/test) to verify:
- Actual API responses match expectations
- Authentication works with profiles
- Time ranges produce correct results

Test locally before pushing by configuring in Claude Code with local path:
```json
{
  "mcpServers": {
    "aws-debug-mcp-local": {
      "command": "uv",
      "args": ["run", "aws-debug-mcp"],
      "cwd": "/full/path/to/aws-debug-mcp"
    }
  }
}
```

## Important Context

### Related Files
- **BOOTSTRAP.md** - Complete technical implementation guide with full code examples for all modules
- **QUICKSTART.md** - Step-by-step checklist for implementing the project from scratch
- **README.md** - User-facing documentation and project goals

### Design Philosophy
- **Tight scope**: Start minimal, expand based on real usage
- **Team-friendly**: Easy installation, environment-based auth
- **Debugging-focused**: Read-only tools, no infrastructure changes
- **Learning project**: Also serves to deepen MCP development expertise

### AWS Credentials
Never commit credentials. Always use:
- AWS profiles (`~/.aws/credentials`)
- Environment variables (`AWS_PROFILE`, `AWS_REGION`)
- IAM roles when running in AWS environments

### Common Pitfalls
- CloudWatch Insights queries are asynchronous - must poll `get_query_results()` until status is "Complete"
- AWS timestamps are in milliseconds, not seconds
- Log group names must be exact matches (case-sensitive)
- Time parsing requires careful handling of timezones (default to UTC)