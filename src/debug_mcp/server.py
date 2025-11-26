"""Main MCP server setup - proxies AWS MCP servers with selective tool exposure."""

import os
from datetime import UTC

from fastmcp import FastMCP

from .mcp_proxy import proxy
from .tools.langsmith import get_langsmith_debugger
from .tools.stepfunctions import StepFunctionsDebugger
from .utils.run_memory import get_memory_store

# Initialize MCP server
mcp = FastMCP("debug-mcp")


# Get configured tools from environment variable
# Format: comma-separated list like "describe_log_groups,analyze_log_group"
# If not set, expose core debugging tools by default
# Set to "all" to expose all 26 tools
DEFAULT_TOOLS = (
    # CloudWatch Logs (5 tools)
    "describe_log_groups,"
    "analyze_log_group,"
    "execute_log_insights_query,"
    "get_logs_insight_query_results,"
    "cancel_logs_insight_query,"
    # Step Functions (5 tools)
    "list_state_machines,"
    "get_state_machine_definition,"
    "list_step_function_executions,"
    "get_step_function_execution_details,"
    "search_step_function_executions,"
    # LangSmith (6 tools)
    "list_langsmith_projects,"
    "list_langsmith_runs,"
    "get_langsmith_run_details,"
    "search_langsmith_runs,"
    "search_run_content,"
    "get_run_field"
)

configured_tools_str = os.getenv("DEBUG_MCP_TOOLS", DEFAULT_TOOLS)
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
        return await proxy.call_cloudwatch_tool("describe_log_groups", {"log_group_name_prefix": log_group_name_prefix})


if should_expose_tool("analyze_log_group"):

    @mcp.tool()
    async def analyze_log_group(log_group_name: str, start_time: str, end_time: str, filter_pattern: str = "") -> dict:
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
        limit: int = 100,
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
            },
        )


if should_expose_tool("get_logs_insight_query_results"):

    @mcp.tool()
    async def get_logs_insight_query_results(query_id: str) -> dict:
        """
        Get results from a CloudWatch Logs Insights query.

        Args:
            query_id: Query ID from execute_log_insights_query
        """
        return await proxy.call_cloudwatch_tool("get_logs_insight_query_results", {"query_id": query_id})


if should_expose_tool("cancel_logs_insight_query"):

    @mcp.tool()
    async def cancel_logs_insight_query(query_id: str) -> dict:
        """
        Cancel an in-progress CloudWatch Logs Insights query.

        Args:
            query_id: Query ID to cancel
        """
        return await proxy.call_cloudwatch_tool("cancel_logs_insight_query", {"query_id": query_id})


# CloudWatch Metrics Tools
if should_expose_tool("get_metric_data"):

    @mcp.tool()
    async def get_metric_data(
        metric_namespace: str,
        metric_name: str,
        dimensions: dict,
        statistic: str,
        start_time: str,
        end_time: str,
        period: int = 300,
    ) -> dict:
        """
        Retrieve detailed CloudWatch metric data for any CloudWatch metric.

        Args:
            metric_namespace: CloudWatch metric namespace (e.g., AWS/Lambda)
            metric_name: Metric name (e.g., Invocations)
            dimensions: Metric dimensions (e.g., {"FunctionName": "my-function"})
            statistic: Statistic type (Average, Sum, Maximum, Minimum, SampleCount)
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            period: Period in seconds (default: 300)
        """
        return await proxy.call_cloudwatch_tool(
            "get_metric_data",
            {
                "metric_namespace": metric_namespace,
                "metric_name": metric_name,
                "dimensions": dimensions,
                "statistic": statistic,
                "start_time": start_time,
                "end_time": end_time,
                "period": period,
            },
        )


if should_expose_tool("get_metric_metadata"):

    @mcp.tool()
    async def get_metric_metadata(metric_namespace: str, metric_name: str) -> dict:
        """
        Get comprehensive metadata about a specific CloudWatch metric.

        Args:
            metric_namespace: CloudWatch metric namespace
            metric_name: Metric name
        """
        return await proxy.call_cloudwatch_tool(
            "get_metric_metadata",
            {"metric_namespace": metric_namespace, "metric_name": metric_name},
        )


if should_expose_tool("get_recommended_metric_alarms"):

    @mcp.tool()
    async def get_recommended_metric_alarms(metric_namespace: str, metric_name: str, dimensions: dict) -> dict:
        """
        Get recommended alarms for a CloudWatch metric based on best practices and statistical analysis.

        Args:
            metric_namespace: CloudWatch metric namespace
            metric_name: Metric name
            dimensions: Metric dimensions
        """
        return await proxy.call_cloudwatch_tool(
            "get_recommended_metric_alarms",
            {
                "metric_namespace": metric_namespace,
                "metric_name": metric_name,
                "dimensions": dimensions,
            },
        )


if should_expose_tool("analyze_metric"):

    @mcp.tool()
    async def analyze_metric(
        metric_namespace: str, metric_name: str, dimensions: dict, start_time: str, end_time: str
    ) -> dict:
        """
        Analyze CloudWatch metric data to determine trend, seasonality, and statistical properties.

        Args:
            metric_namespace: CloudWatch metric namespace
            metric_name: Metric name
            dimensions: Metric dimensions
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
        """
        return await proxy.call_cloudwatch_tool(
            "analyze_metric",
            {
                "metric_namespace": metric_namespace,
                "metric_name": metric_name,
                "dimensions": dimensions,
                "start_time": start_time,
                "end_time": end_time,
            },
        )


# CloudWatch Alarms Tools
if should_expose_tool("get_active_alarms"):

    @mcp.tool()
    async def get_active_alarms() -> dict:
        """
        Identify currently active CloudWatch alarms across the account.
        """
        return await proxy.call_cloudwatch_tool("get_active_alarms", {})


if should_expose_tool("get_alarm_history"):

    @mcp.tool()
    async def get_alarm_history(alarm_name: str) -> dict:
        """
        Retrieve historical state changes and patterns for a given CloudWatch alarm.

        Args:
            alarm_name: CloudWatch alarm name
        """
        return await proxy.call_cloudwatch_tool("get_alarm_history", {"alarm_name": alarm_name})


# ECS Tools - proxied from awslabs.ecs-mcp-server
if should_expose_tool("containerize_app"):

    @mcp.tool()
    async def containerize_app(app_directory: str) -> dict:
        """
        Generate Dockerfile and container configurations for web applications following best practices.

        Args:
            app_directory: Path to application source code directory
        """
        return await proxy.call_ecs_tool("containerize_app", {"app_directory": app_directory})


if should_expose_tool("build_and_push_image_to_ecr"):

    @mcp.tool()
    async def build_and_push_image_to_ecr(app_directory: str, image_tag: str, repository_name: str) -> dict:
        """
        Create ECR infrastructure, build Docker images, and push them to the repository.

        Args:
            app_directory: Path to application directory
            image_tag: Docker image tag
            repository_name: ECR repository name
        """
        return await proxy.call_ecs_tool(
            "build_and_push_image_to_ecr",
            {
                "app_directory": app_directory,
                "image_tag": image_tag,
                "repository_name": repository_name,
            },
        )


if should_expose_tool("validate_ecs_express_mode_prerequisites"):

    @mcp.tool()
    async def validate_ecs_express_mode_prerequisites(
        task_execution_role_name: str, infrastructure_role_name: str, ecr_image_uri: str
    ) -> dict:
        """
        Verify that required IAM roles and Docker images exist before ECS deployment.

        Args:
            task_execution_role_name: Name of task execution IAM role
            infrastructure_role_name: Name of infrastructure IAM role
            ecr_image_uri: ECR image URI
        """
        return await proxy.call_ecs_tool(
            "validate_ecs_express_mode_prerequisites",
            {
                "task_execution_role_name": task_execution_role_name,
                "infrastructure_role_name": infrastructure_role_name,
                "ecr_image_uri": ecr_image_uri,
            },
        )


if should_expose_tool("wait_for_service_ready"):

    @mcp.tool()
    async def wait_for_service_ready(cluster_name: str, service_name: str) -> dict:
        """
        Poll ECS service status until tasks reach a running state (checks every 10 seconds).

        Args:
            cluster_name: ECS cluster name
            service_name: ECS service name
        """
        return await proxy.call_ecs_tool(
            "wait_for_service_ready", {"cluster_name": cluster_name, "service_name": service_name}
        )


if should_expose_tool("delete_app"):

    @mcp.tool()
    async def delete_app(service_name: str, cluster_name: str) -> dict:
        """
        Remove Express Mode deployment and associated infrastructure including ECR stacks.

        Args:
            service_name: ECS service name
            cluster_name: ECS cluster name
        """
        return await proxy.call_ecs_tool("delete_app", {"service_name": service_name, "cluster_name": cluster_name})


if should_expose_tool("ecs_troubleshooting_tool"):

    @mcp.tool()
    async def ecs_troubleshooting_tool(
        action: str,
        cluster_name: str = "",
        service_name: str = "",
        task_id: str = "",
        stack_name: str = "",
    ) -> dict:
        """
        Consolidated diagnostics tool for ECS issue resolution.

        Actions: initial_assessment, cloudformation_diagnostics, service_events,
        task_failure_analysis, cloudwatch_logs, image_pull_failure, network_diagnostics

        Args:
            action: Troubleshooting action to perform
            cluster_name: ECS cluster name (varies by action)
            service_name: ECS service name (varies by action)
            task_id: ECS task ID (varies by action)
            stack_name: CloudFormation stack name (varies by action)
        """
        args = {"action": action}
        if cluster_name:
            args["cluster_name"] = cluster_name
        if service_name:
            args["service_name"] = service_name
        if task_id:
            args["task_id"] = task_id
        if stack_name:
            args["stack_name"] = stack_name

        return await proxy.call_ecs_tool("ecs_troubleshooting_tool", args)


if should_expose_tool("ecs_resource_management"):

    @mcp.tool()
    async def ecs_resource_management(
        resource_type: str, operation: str, resource_identifier: str = "", configuration: dict = {}
    ) -> dict:
        """
        Comprehensive access to ECS resources for monitoring and management.

        Supports read/list/describe operations on clusters, services, tasks, task definitions,
        container instances, capacity providers, and ECR repositories.

        Args:
            resource_type: Type of resource (cluster, service, task, etc.)
            operation: Operation type (list, describe, create, update, delete)
            resource_identifier: Resource identifier (varies by operation)
            configuration: Configuration details (for create/update operations)
        """
        args = {"resource_type": resource_type, "operation": operation}
        if resource_identifier:
            args["resource_identifier"] = resource_identifier
        if configuration:
            args["configuration"] = configuration

        return await proxy.call_ecs_tool("ecs_resource_management", args)


if should_expose_tool("aws_knowledge_aws___search_documentation"):

    @mcp.tool()
    async def aws_knowledge_aws___search_documentation(query: str, topic_area: str = "") -> dict:
        """
        Search AWS documentation including latest docs, API references, blogs, and best practices.

        Args:
            query: Search query
            topic_area: Optional topic area to filter results
        """
        args = {"query": query}
        if topic_area:
            args["topic_area"] = topic_area

        return await proxy.call_ecs_tool("aws_knowledge_aws___search_documentation", args)


if should_expose_tool("aws_knowledge_aws___read_documentation"):

    @mcp.tool()
    async def aws_knowledge_aws___read_documentation(documentation_url: str) -> dict:
        """
        Convert AWS documentation pages to markdown format.

        Args:
            documentation_url: AWS documentation page URL
        """
        return await proxy.call_ecs_tool(
            "aws_knowledge_aws___read_documentation", {"documentation_url": documentation_url}
        )


if should_expose_tool("aws_knowledge_aws___recommend"):

    @mcp.tool()
    async def aws_knowledge_aws___recommend(topic: str) -> dict:
        """
        Get content recommendations for AWS documentation pages.

        Args:
            topic: Topic or content area
        """
        return await proxy.call_ecs_tool("aws_knowledge_aws___recommend", {"topic": topic})


# Step Functions Debugging Tools - using boto3 directly
# These tools provide comprehensive debugging capabilities for Step Functions executions
# Initialize debugger (uses AWS_REGION from environment)
sf_debugger = StepFunctionsDebugger()


if should_expose_tool("list_state_machines"):

    @mcp.tool()
    async def list_state_machines(max_results: int = 100) -> dict:
        """
        List all Step Functions state machines in the account.

        Args:
            max_results: Maximum number of state machines to return (default: 100)
        """
        state_machines = sf_debugger.list_state_machines(max_results=max_results)
        return {"state_machines": state_machines, "count": len(state_machines)}


if should_expose_tool("list_step_function_executions"):

    @mcp.tool()
    async def list_step_function_executions(
        state_machine_arn: str,
        status_filter: str = "",
        max_results: int = 100,
        hours_back: int = 168,
    ) -> dict:
        """
        List executions for a Step Functions state machine.

        Args:
            state_machine_arn: ARN of the state machine
            status_filter: Optional status filter (RUNNING, SUCCEEDED, FAILED, TIMED_OUT, ABORTED)
            max_results: Maximum number of executions to return (default: 100)
            hours_back: Number of hours to look back (default: 168 = 7 days)
        """
        executions = sf_debugger.list_executions(
            state_machine_arn=state_machine_arn,
            status_filter=status_filter if status_filter else None,
            max_results=max_results,
            hours_back=hours_back,
        )
        return {
            "executions": executions,
            "count": len(executions),
            "state_machine_arn": state_machine_arn,
        }


if should_expose_tool("get_state_machine_definition"):

    @mcp.tool()
    async def get_state_machine_definition(state_machine_arn: str) -> dict:
        """
        Get the state machine definition including ASL and extracted resources.

        Returns the full Amazon States Language (ASL) definition along with
        extracted Lambda ARNs and other resource ARNs used in the workflow.

        Args:
            state_machine_arn: ARN of the state machine
        """
        return sf_debugger.get_state_machine_definition(state_machine_arn)


if should_expose_tool("get_step_function_execution_details"):

    @mcp.tool()
    async def get_step_function_execution_details(execution_arn: str, include_definition: bool = False) -> dict:
        """
        Get detailed information about a specific Step Functions execution.

        Includes full execution history with state-level inputs and outputs.
        Only Step Functions states are included (Lambda task events are filtered out).

        Args:
            execution_arn: ARN of the execution
            include_definition: If True, includes the state machine definition with Lambda ARNs (default: False)
        """
        if include_definition:
            return sf_debugger.get_execution_details_with_definition(execution_arn)
        return sf_debugger.get_execution_details(execution_arn)


if should_expose_tool("search_step_function_executions"):

    @mcp.tool()
    async def search_step_function_executions(
        state_machine_arn: str,
        state_name: str = "",
        input_pattern: str = "",
        output_pattern: str = "",
        status_filter: str = "",
        max_results: int = 50,
        hours_back: int = 168,
        include_definition: bool = False,
    ) -> dict:
        """
        Search Step Functions executions with advanced filtering.

        Supports regex patterns for state names and input/output content matching.
        This is powerful for finding specific execution scenarios.

        Args:
            state_machine_arn: ARN of the state machine
            state_name: Filter by state name (supports regex, e.g., "Match.*Entity")
            input_pattern: Regex pattern to match in state inputs (e.g., "customer_id.*12345")
            output_pattern: Regex pattern to match in state outputs (e.g., "entity_type.*company")
            status_filter: Optional status filter (RUNNING, SUCCEEDED, FAILED, etc.)
            max_results: Maximum number of executions to process (default: 50)
            hours_back: Number of hours to look back (default: 168 = 7 days)
            include_definition: If True, includes the state machine definition with Lambda ARNs (default: False)
        """
        executions = sf_debugger.search_executions(
            state_machine_arn=state_machine_arn,
            state_name=state_name if state_name else None,
            input_pattern=input_pattern if input_pattern else None,
            output_pattern=output_pattern if output_pattern else None,
            status_filter=status_filter if status_filter else None,
            max_results=max_results,
            hours_back=hours_back,
            include_definition=include_definition,
        )
        return {
            "executions": executions,
            "count": len(executions),
            "filters": {
                "state_name": state_name or None,
                "input_pattern": input_pattern or None,
                "output_pattern": output_pattern or None,
                "status": status_filter or None,
            },
        }


# LangSmith Debugging Tools - using langsmith SDK
# Supports multiple environments via AWS Secrets Manager or .env file
if should_expose_tool("list_langsmith_projects"):

    @mcp.tool()
    async def list_langsmith_projects(environment: str, limit: int = 100) -> dict:
        """
        List available LangSmith projects.

        Args:
            environment: Environment to query ('prod', 'dev', 'local')
                        - prod: Uses PRODUCTION/env/vars from AWS Secrets Manager
                        - dev: Uses DEV/env/vars from AWS Secrets Manager
                        - local: Loads from .env file
            limit: Maximum number of projects to return (default: 100)
        """
        debugger = get_langsmith_debugger(environment)
        projects = debugger.list_projects(limit=limit)
        return {
            "environment": environment,
            "projects": projects,
            "count": len(projects),
            "default_project": debugger.default_project,
        }


if should_expose_tool("list_langsmith_runs"):

    @mcp.tool()
    async def list_langsmith_runs(
        environment: str,
        project_name: str = "",
        run_type: str = "",
        is_root: bool = True,
        error_only: bool = False,
        hours_back: int = 24,
        limit: int = 100,
    ) -> dict:
        """
        List runs/traces from a LangSmith project.

        Args:
            environment: Environment to query ('prod', 'dev', 'local')
            project_name: Project name (uses default from credentials if empty)
            run_type: Filter by type: chain, llm, tool, retriever, embedding, prompt, parser
            is_root: If True, return only root runs/top-level traces (default: True)
            error_only: If True, return only errored runs (default: False)
            hours_back: Number of hours to look back (default: 24)
            limit: Maximum number of runs to return (default: 100)
        """
        from datetime import datetime, timedelta

        debugger = get_langsmith_debugger(environment)

        start_time = datetime.now(UTC) - timedelta(hours=hours_back)

        runs = debugger.list_runs(
            project_name=project_name if project_name else None,
            run_type=run_type if run_type else None,
            is_root=is_root,
            error=True if error_only else None,
            start_time=start_time,
            limit=limit,
        )

        return {
            "environment": environment,
            "project": project_name or debugger.default_project,
            "runs": runs,
            "count": len(runs),
            "filters": {
                "run_type": run_type or None,
                "is_root": is_root,
                "error_only": error_only,
                "hours_back": hours_back,
            },
        }


if should_expose_tool("get_langsmith_run_details"):

    @mcp.tool()
    async def get_langsmith_run_details(
        environment: str, run_id: str, include_children: bool = True, full_content: bool = False
    ) -> dict:
        """
        Get detailed information about a specific LangSmith run/trace.

        By default, returns a SUMMARY with key metadata. Full content is stored in memory
        with semantic embeddings (sentence-transformers) and can be searched using
        search_run_content() or accessed via get_run_field().

        Args:
            environment: Environment to query ('prod', 'dev', 'local')
            run_id: The run ID (UUID) to retrieve
            include_children: If True, also fetch child runs (default: True)
            full_content: If True, return full content instead of summary (default: False).
                         WARNING: Full content can be ~25k+ tokens. Use only when necessary.

        Returns:
            - reference_id: Use this with search_run_content() or get_run_field()
            - summary: Key metadata (status, latency, tokens, tools used, etc.)
            - If full_content=True: Also includes the complete run data

        TIP: Use search_run_content(reference_id, "your query") to semantically search
        within the run without loading everything into context.
        """
        debugger = get_langsmith_debugger(environment)
        details = debugger.get_run_details(run_id, include_children=include_children)

        # Create reference ID and store in memory
        reference_id = f"{environment}:{run_id}"
        memory = get_memory_store()

        # Extract summary information
        summary = _extract_run_summary(details)

        # Store full content in memory for later retrieval
        memory.store(reference_id, details, summary=summary)

        result = {
            "environment": environment,
            "reference_id": reference_id,
            "summary": summary,
            "hint": (
                "Full content stored in memory. Use search_run_content(reference_id, query) "
                "to search within this run, or get_run_field(reference_id, field_path) for specific fields."
            ),
        }

        # Optionally include full content (for backward compatibility or when explicitly needed)
        if full_content:
            result["run"] = details
            result["warning"] = "Full content included. This may use significant context."

        return result


def _extract_run_summary(details: dict) -> dict:
    """Extract key summary information from run details."""
    summary = {
        "id": details.get("id"),
        "name": details.get("name"),
        "status": details.get("status"),
        "run_type": details.get("run_type"),
        "latency_seconds": details.get("latency_seconds"),
        "total_tokens": details.get("total_tokens"),
        "prompt_tokens": details.get("prompt_tokens"),
        "completion_tokens": details.get("completion_tokens"),
        "error": details.get("error"),
        "link": details.get("link"),
    }

    # Extract tools called from chat history
    tools_called = []
    outputs = details.get("outputs", {})
    chat_history = outputs.get("chat_history", [])
    if isinstance(chat_history, list):
        for msg in chat_history:
            if isinstance(msg, dict):
                # Look for tool calls in AI messages
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        tools_called.append(tc["name"])
                # Look for tool messages
                if msg.get("type") == "tool":
                    tool_name = msg.get("name", "unknown_tool")
                    if tool_name not in tools_called:
                        tools_called.append(tool_name)

    summary["tools_called"] = list(set(tools_called)) if tools_called else []
    summary["message_count"] = len(chat_history) if isinstance(chat_history, list) else 0

    # Include user query if available
    inputs = details.get("inputs", {})
    if isinstance(inputs, dict):
        input_data = inputs.get("input", {})
        if isinstance(input_data, dict):
            summary["user_query"] = input_data.get("user_query")

    # Include final response text if available
    if isinstance(outputs, dict):
        response = outputs.get("response", {})
        if isinstance(response, dict):
            final_text = response.get("final_text", "")
            if final_text:
                # Truncate for summary
                summary["response_preview"] = final_text[:500] + ("..." if len(final_text) > 500 else "")

    # Child run count
    children = details.get("children", [])
    summary["child_count"] = len(children) if isinstance(children, list) else 0

    return summary


if should_expose_tool("search_langsmith_runs"):

    @mcp.tool()
    async def search_langsmith_runs(
        environment: str,
        search_text: str,
        project_name: str = "",
        hours_back: int = 24,
        limit: int = 50,
        include_children: bool = True,
    ) -> dict:
        """
        Search for LangSmith conversations containing specific text content.

        PURPOSE: This is a lightweight search tool designed to find conversations
        that contain a specific string. It returns only minimal match information
        to avoid large responses. Use this to locate a conversation, then use
        get_langsmith_run_details to retrieve full details.

        IMPORTANT - Use Specific Search Strings:
        - GOOD: "Error code XYZ-12345", "user@specific-email.com", "order_id: 98765"
        - BAD: "error", "failed", "user" (too generic, will match many runs)

        The more unique your search text, the faster and more accurate results.

        Args:
            environment: Environment to query ('prod', 'dev', 'local')
            search_text: The text to search for (case-insensitive). Use unique
                        identifiers, error codes, or specific phrases for best results.
            project_name: Project name (uses default from credentials if empty)
            hours_back: Number of hours to look back (default: 24)
            limit: Maximum runs to search through (default: 50)
            include_children: Search in child runs too (default: True)

        Returns:
            List of matches with run_id, context snippet, and link to full details

        If this tool fails, please report the issue at:
        https://github.com/Coykto/debug_mcp/issues
        """
        debugger = get_langsmith_debugger(environment)

        try:
            matches = debugger.find_conversation_by_content(
                search_text=search_text,
                project_name=project_name if project_name else None,
                hours_back=hours_back,
                limit=limit,
                include_children=include_children,
            )
        except Exception as e:
            # Provide helpful error message with guidance
            error_msg = str(e)
            return {
                "error": True,
                "error_message": error_msg,
                "environment": environment,
                "search_text": search_text,
                "guidance": (
                    "This search tool encountered an error. "
                    "DO NOT fall back to using get_langsmith_run_details to search - "
                    "that approach will overflow context with ~25k tokens per run. "
                    "Instead, please report this issue so it can be fixed."
                ),
                "report_issue": "https://github.com/Coykto/debug_mcp/issues",
                "tip": "Include the error message and search parameters when reporting.",
            }

        return {
            "environment": environment,
            "project": project_name or debugger.default_project,
            "search_text": search_text,
            "matches": matches,
            "match_count": len(matches),
            "runs_searched": limit,
            "hours_back": hours_back,
            "tip": "Use get_langsmith_run_details with a run_id to get full conversation details",
        }


if should_expose_tool("search_run_content"):

    @mcp.tool()
    async def search_run_content(
        reference_id: str,
        query: str,
        search_type: str = "auto",
        max_results: int = 5,
    ) -> dict:
        """
        Search within a previously fetched LangSmith run's content using semantic similarity.

        This tool searches through the full content of a run that was previously
        retrieved with get_langsmith_run_details(). Use this to find specific
        information without loading the entire run into context.

        The run content is automatically chunked and embedded using sentence-transformers
        (all-MiniLM-L6-v2 model) for semantic search. This means you can search using
        natural language queries and find semantically related content.

        Args:
            reference_id: The reference_id returned by get_langsmith_run_details()
                         (format: "environment:run_id", e.g., "dev:abc-123-def")
            query: What to search for. Can be:
                   - Specific text: "July 5th, 2024"
                   - Keywords: "birthday start date"
                   - Semantic queries: "when did the project start" (finds related content)
            search_type: Search method to use:
                        - "auto": Use semantic similarity search (default, recommended)
                        - "keyword": Exact keyword/text matching only
                        - "similar": Explicit semantic similarity search
            max_results: Maximum number of matching chunks to return (default: 5)

        Returns:
            List of matching content chunks with their location (path) in the data structure
            and similarity scores (for semantic search)

        Example:
            1. First: get_langsmith_run_details(environment="dev", run_id="abc-123")
               -> Returns reference_id: "dev:abc-123"
            2. Then: search_run_content(reference_id="dev:abc-123", query="RAG tool results")
               -> Returns relevant chunks containing RAG tool information
        """
        memory = get_memory_store()
        stored_run = memory.get(reference_id)

        if not stored_run:
            return {
                "error": True,
                "error_message": f"No stored run found for reference_id: {reference_id}",
                "hint": "Make sure to call get_langsmith_run_details() first to store the run content.",
                "available_runs": [r["reference_id"] for r in memory.list_stored_runs()],
            }

        # Determine search method
        if search_type == "similar":
            results = memory.search_similar(reference_id, query, max_results=max_results)
            method_used = "semantic_similarity"
        elif search_type == "keyword":
            results = memory.search_keyword(reference_id, query, max_results=max_results)
            method_used = "keyword"
        else:  # auto
            # Try similarity first if embeddings are available
            results = memory.search_similar(reference_id, query, max_results=max_results)
            if results and any("similarity" in r for r in results):
                method_used = "semantic_similarity"
            else:
                results = memory.search_keyword(reference_id, query, max_results=max_results)
                method_used = "keyword"

        return {
            "reference_id": reference_id,
            "query": query,
            "search_method": method_used,
            "results": results,
            "result_count": len(results),
            "tip": "Use get_run_field(reference_id, path) to get the full content at a specific path",
        }


if should_expose_tool("get_run_field"):

    @mcp.tool()
    async def get_run_field(reference_id: str, field_path: str) -> dict:
        """
        Get a specific field from a previously fetched LangSmith run.

        Use this to retrieve the full content of a specific field after
        finding it with search_run_content().

        Args:
            reference_id: The reference_id from get_langsmith_run_details()
            field_path: Dot-notation path to the field. Examples:
                       - "outputs.chat_history.2.content" - Get 3rd message content
                       - "outputs.response.final_text" - Get the final response
                       - "inputs.input.user_query" - Get the user's question
                       - "children.0.outputs" - Get first child run's outputs

        Returns:
            The value at the specified path, or error if not found

        Example paths for common data:
            - User query: "inputs.input.user_query"
            - Chat history: "outputs.chat_history"
            - Specific message: "outputs.chat_history.{index}.content"
            - Final response: "outputs.response.final_text"
            - Tool calls: "outputs.chat_history.{index}.tool_calls"
        """
        memory = get_memory_store()
        stored_run = memory.get(reference_id)

        if not stored_run:
            return {
                "error": True,
                "error_message": f"No stored run found for reference_id: {reference_id}",
                "hint": "Make sure to call get_langsmith_run_details() first to store the run content.",
                "available_runs": [r["reference_id"] for r in memory.list_stored_runs()],
            }

        value = memory.get_field(reference_id, field_path)

        if value is None:
            # Try to provide helpful suggestions
            available_keys = []
            parts = field_path.split(".")
            parent_path = ".".join(parts[:-1]) if len(parts) > 1 else ""
            parent = memory.get_field(reference_id, parent_path) if parent_path else stored_run.full_data

            if isinstance(parent, dict):
                available_keys = list(parent.keys())[:20]  # Limit suggestions
            elif isinstance(parent, list):
                available_keys = [f"{i}" for i in range(min(len(parent), 20))]

            return {
                "error": True,
                "error_message": f"Field not found at path: {field_path}",
                "parent_path": parent_path or "(root)",
                "available_keys": available_keys,
                "hint": "Check the path and try again. Use search_run_content() to find content locations.",
            }

        # Calculate size info for large values
        size_info = {}
        if isinstance(value, str):
            size_info = {"type": "string", "length": len(value), "word_count": len(value.split())}
        elif isinstance(value, list):
            size_info = {"type": "list", "length": len(value)}
        elif isinstance(value, dict):
            size_info = {"type": "dict", "keys": list(value.keys())[:20]}

        return {
            "reference_id": reference_id,
            "field_path": field_path,
            "value": value,
            "size_info": size_info,
        }
