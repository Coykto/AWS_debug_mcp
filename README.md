# AWS Debug MCP

MCP server for debugging AWS distributed systems (Lambda, Step Functions, ECS) directly from Claude Code or any MCP client.

**Status**: ✅ Complete with 26 tools from CloudWatch, ECS, and Step Functions
**Repository**: https://github.com/Coykto/AWS_debug_mcp

## Quick Start

### Installation

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "aws-debug-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Coykto/AWS_debug_mcp",
        "aws-debug-mcp"
      ],
      "env": {
        "AWS_PROFILE": "your-aws-profile-name",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

**Prerequisites:**
- Python 3.11+
- `uvx` installed ([installation guide](https://docs.astral.sh/uv/))
- AWS credentials configured (`aws configure`)

### What You Can Ask Claude

**CloudWatch Logs:**
- "List all Lambda log groups"
- "Search for ERROR in /aws/lambda/my-function from the last hour"
- "Analyze /aws/lambda/my-function logs for patterns"

**CloudWatch Metrics & Alarms:**
- "Show me Lambda invocation metrics for my-function from yesterday"
- "What alarms are currently active?"
- "Get recommended alarms for my Lambda function"

**ECS Debugging:**
- "Troubleshoot ECS service my-service in cluster my-cluster"
- "Show me ECS task failures for my-service"
- "List all running ECS tasks in my-cluster"

**ECS Deployment:**
- "Containerize my app in /path/to/app"
- "Build and push Docker image to ECR"

**AWS Documentation:**
- "Search AWS docs for Lambda best practices"

**Step Functions Debugging:**
- "List all my Step Functions state machines"
- "Show me the workflow definition and Lambda functions for state machine X"
- "Show me failed executions for state machine X from the last 3 days"
- "Get execution details including the workflow definition"
- "Find executions where the Match state output contains 'company' and show me the Lambda ARNs"

## How It Works

This MCP server acts as a **proxy/gateway** to AWS MCP servers:
1. Spawns AWS MCP servers (awslabs.cloudwatch-mcp-server, awslabs.ecs-mcp-server, etc.) as subprocesses
2. Forwards only the tools you select, hiding the rest
3. Keeps your tool list minimal and focused on debugging

**No reimplementation** - AWS maintains the underlying functionality, we just filter which tools are exposed.

## Available Tools

### CloudWatch (11 tools)

**Logs:**
- `describe_log_groups` - List CloudWatch log groups
- `analyze_log_group` - Analyze logs for anomalies and patterns
- `execute_log_insights_query` - Run CloudWatch Insights queries
- `get_logs_insight_query_results` - Get query results
- `cancel_logs_insight_query` - Cancel running query

**Metrics:**
- `get_metric_data` - Retrieve detailed metric data
- `get_metric_metadata` - Get metric descriptions
- `get_recommended_metric_alarms` - Get alarm recommendations
- `analyze_metric` - Analyze metrics for trends

**Alarms:**
- `get_active_alarms` - List currently active alarms
- `get_alarm_history` - Get alarm state change history

### ECS (10 tools)

**Deployment:**
- `containerize_app` - Generate Dockerfile and configs
- `build_and_push_image_to_ecr` - Build and push to ECR
- `validate_ecs_express_mode_prerequisites` - Verify IAM roles
- `wait_for_service_ready` - Poll service status
- `delete_app` - Remove deployment

**Troubleshooting:**
- `ecs_troubleshooting_tool` - Diagnostics (events, task failures, logs, network)
- `ecs_resource_management` - Manage clusters, services, tasks

**Documentation:**
- `aws_knowledge_aws___search_documentation` - Search AWS docs
- `aws_knowledge_aws___read_documentation` - Convert docs to markdown
- `aws_knowledge_aws___recommend` - Get doc recommendations

### Step Functions (5 tools)

**Debugging:**
- `list_state_machines` - List all state machines (get ARNs)
- `get_state_machine_definition` - Get ASL definition with extracted Lambda ARNs and resources
- `list_step_function_executions` - List executions with filtering
- `get_step_function_execution_details` - Get full execution details with state inputs/outputs
- `search_step_function_executions` - Advanced search with state/input/output pattern matching

**Definition & Resources:**
The `get_state_machine_definition` tool extracts:
- Full Amazon States Language (ASL) workflow definition
- Lambda function ARNs used in the workflow
- Other resources (SNS topics, SQS queues, DynamoDB tables, nested state machines)
- IAM role, logging, and tracing configuration

You can also include the definition with execution details using `include_definition=True` in:
- `get_step_function_execution_details` - See the workflow definition alongside execution data
- `search_step_function_executions` - See definitions with filtered execution results

## Configuration

### AWS Authentication
```json
"env": {
  "AWS_PROFILE": "your-profile-name",
  "AWS_REGION": "us-east-1"
}
```

### Tool Selection

Filter which tools to expose using `AWS_DEBUG_MCP_TOOLS`:

```json
// Minimal - only logs
"AWS_DEBUG_MCP_TOOLS": "describe_log_groups,execute_log_insights_query,get_logs_insight_query_results"

// Debugging focus - logs, metrics, alarms, ECS troubleshooting
"AWS_DEBUG_MCP_TOOLS": "describe_log_groups,analyze_log_group,execute_log_insights_query,get_active_alarms,ecs_troubleshooting_tool,ecs_resource_management"

// Expose all 26 tools (default)
"AWS_DEBUG_MCP_TOOLS": "all"
```

**Available tool names:**

CloudWatch: `describe_log_groups`, `analyze_log_group`, `execute_log_insights_query`, `get_logs_insight_query_results`, `cancel_logs_insight_query`, `get_metric_data`, `get_metric_metadata`, `get_recommended_metric_alarms`, `analyze_metric`, `get_active_alarms`, `get_alarm_history`

ECS: `containerize_app`, `build_and_push_image_to_ecr`, `validate_ecs_express_mode_prerequisites`, `wait_for_service_ready`, `delete_app`, `ecs_troubleshooting_tool`, `ecs_resource_management`, `aws_knowledge_aws___search_documentation`, `aws_knowledge_aws___read_documentation`, `aws_knowledge_aws___recommend`

Step Functions: `list_state_machines`, `get_state_machine_definition`, `list_step_function_executions`, `get_step_function_execution_details`, `search_step_function_executions`

## Troubleshooting

### Server won't start
- Check AWS credentials: `aws sts get-caller-identity --profile YOUR_PROFILE`
- Verify uvx is installed: `uvx --version`
- Check Claude Code MCP logs in settings

### Wrong AWS account
- Update `AWS_PROFILE` in the env section
- Make sure the profile exists in `~/.aws/credentials`

### Too many tools in the list
- Set `AWS_DEBUG_MCP_TOOLS` to only the tools you need
- See available tool names above

## Development

### Local Development

```bash
# Install dependencies
uv sync

# Run the server
uv run aws-debug-mcp

# Test
uv run pytest
```

### Adding New AWS MCPs

To expose tools from other AWS MCP servers:

**1. Add connection to `src/aws_debug_mcp/mcp_proxy.py`:**

```python
@asynccontextmanager
async def _connect_to_SERVICE(self):
    server_params = StdioServerParameters(
        command="uvx",
        args=["awslabs.SERVICE-mcp-server@latest"],
        env={"AWS_PROFILE": self.aws_profile, "AWS_REGION": self.aws_region}
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def call_SERVICE_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
    async with self._connect_to_SERVICE() as session:
        result = await session.call_tool(tool_name, arguments)
        return result.content
```

**2. Add proxy functions to `src/aws_debug_mcp/server.py`:**

```python
if should_expose_tool("your_tool_name"):
    @mcp.tool()
    async def your_tool_name(arg1: str) -> dict:
        """Tool description for Claude."""
        return await proxy.call_SERVICE_tool("upstream_tool_name", {"arg1": arg1})
```

**3. Test and use:**

```bash
# Add to your AWS_DEBUG_MCP_TOOLS list
"AWS_DEBUG_MCP_TOOLS": "your_tool_name,other_tools"
```

**Resources:**
- [AWS MCP Servers](https://awslabs.github.io/mcp/) - Find available MCPs and tools
- [CloudWatch MCP](https://awslabs.github.io/mcp/servers/cloudwatch-mcp-server)
- [ECS MCP](https://awslabs.github.io/mcp/servers/ecs-mcp-server)
- [Step Functions MCP](https://awslabs.github.io/mcp/servers/stepfunctions-tool-mcp-server)

## Team Sharing

Share with your team:
1. They update `AWS_PROFILE` with their own profile name
2. Optionally adjust `AWS_REGION` if different
3. Optionally customize `AWS_DEBUG_MCP_TOOLS` to their preference

## License

MIT License - See LICENSE file
