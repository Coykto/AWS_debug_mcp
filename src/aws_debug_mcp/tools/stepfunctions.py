"""
AWS Step Functions debugging tools.

Provides comprehensive debugging capabilities for Step Functions executions including:
- Listing state machines and executions
- Retrieving execution details with state-level inputs/outputs
- Searching executions by state name and input/output patterns
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3


class StepFunctionsDebugger:
    """Core debugging functionality for Step Functions."""

    def __init__(self, region: str | None = None):
        """
        Initialize the debugger.

        Args:
            region: AWS region (defaults to AWS_REGION env var or us-east-1)
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.sfn_client = boto3.client("stepfunctions", region_name=self.region)

    def list_state_machines(self, max_results: int = 100) -> list[dict[str, Any]]:
        """
        List all Step Functions state machines in the account.

        Args:
            max_results: Maximum number of state machines to return

        Returns:
            List of state machine metadata including ARN, name, type, creation date
        """
        state_machines = []
        paginator = self.sfn_client.get_paginator("list_state_machines")

        for page in paginator.paginate(PaginationConfig={"MaxItems": max_results}):
            for sm in page["stateMachines"]:
                state_machines.append(
                    {
                        "name": sm["name"],
                        "arn": sm["stateMachineArn"],
                        "type": sm["type"],
                        "creationDate": sm["creationDate"].isoformat(),
                    }
                )

        return state_machines

    def generate_execution_link(
        self, execution_arn: str, region: str | None = None
    ) -> str:
        """
        Generate AWS console link for an execution.

        Args:
            execution_arn: Full ARN of the execution
            region: AWS region (defaults to self.region)

        Returns:
            AWS console URL for the execution
        """
        region = region or self.region
        base_url = f"https://{region}.console.aws.amazon.com/states/home"
        return f"{base_url}?region={region}#/v2/executions/details/{execution_arn}"

    def list_executions(
        self,
        state_machine_arn: str,
        status_filter: str | None = None,
        max_results: int = 100,
        hours_back: int = 168,  # 7 days default
    ) -> list[dict[str, Any]]:
        """
        List executions for a state machine.

        Args:
            state_machine_arn: ARN of the state machine
            status_filter: Optional status filter (RUNNING, SUCCEEDED, FAILED, TIMED_OUT, ABORTED)
            max_results: Maximum number of executions to return
            hours_back: Number of hours to look back (default: 168 = 7 days)

        Returns:
            List of execution metadata including name, ARN, status, dates, console link
        """
        executions = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        paginator = self.sfn_client.get_paginator("list_executions")
        pagination_params = {
            "stateMachineArn": state_machine_arn,
            "PaginationConfig": {"MaxItems": max_results},
        }

        if status_filter:
            pagination_params["statusFilter"] = status_filter

        for page in paginator.paginate(**pagination_params):
            for execution in page["executions"]:
                # Skip executions older than cutoff
                if execution["startDate"] < cutoff_date:
                    continue

                exec_data = {
                    "name": execution["name"],
                    "arn": execution["executionArn"],
                    "status": execution["status"],
                    "startDate": execution["startDate"].isoformat(),
                    "link": self.generate_execution_link(execution["executionArn"]),
                }

                # Add stop date if execution is completed
                if "stopDate" in execution:
                    exec_data["stopDate"] = execution["stopDate"].isoformat()

                executions.append(exec_data)

        return executions

    def parse_state_history(self, history: list[dict]) -> dict[str, dict]:
        """
        Parse execution history to extract state inputs and outputs.

        Only processes StateEntered and StateExited events, ignoring Lambda events.

        Args:
            history: List of history events from get_execution_history

        Returns:
            Dictionary mapping state names to their inputs and outputs
        """
        result = defaultdict(lambda: {"inputs": [], "outputs": []})

        for event in history:
            event_type = event.get("type", "")

            # Only process Step Functions state events
            if "StateEntered" in event_type:
                details = event.get("stateEnteredEventDetails", {})
                state_name = details.get("name")
                state_input = details.get("input")

                if state_name and state_input:
                    result[state_name]["inputs"].append(state_input)

            elif "StateExited" in event_type:
                details = event.get("stateExitedEventDetails", {})
                state_name = details.get("name")
                state_output = details.get("output")

                if state_name and state_output:
                    result[state_name]["outputs"].append(state_output)

        return dict(result)

    def get_execution_details(self, execution_arn: str) -> dict[str, Any]:
        """
        Get detailed information about a specific execution including state history.

        Args:
            execution_arn: ARN of the execution

        Returns:
            Execution details with parsed state inputs/outputs
        """
        # Get execution description
        execution = self.sfn_client.describe_execution(executionArn=execution_arn)

        # Get execution history
        history = []
        history_paginator = self.sfn_client.get_paginator("get_execution_history")

        for page in history_paginator.paginate(executionArn=execution_arn):
            history.extend(page["events"])

        # Parse state history
        states = self.parse_state_history(history)

        # Build result
        result = {
            "name": execution["name"],
            "arn": execution["executionArn"],
            "stateMachineArn": execution["stateMachineArn"],
            "status": execution["status"],
            "startDate": execution["startDate"].isoformat(),
            "input": execution.get("input", ""),
            "link": self.generate_execution_link(execution_arn),
            "states": states,
        }

        # Add optional fields
        if "stopDate" in execution:
            result["stopDate"] = execution["stopDate"].isoformat()
        if "output" in execution:
            result["output"] = execution["output"]
        if "error" in execution:
            result["error"] = execution["error"]
        if "cause" in execution:
            result["cause"] = execution["cause"]

        return result

    def search_executions(
        self,
        state_machine_arn: str,
        state_name: str | None = None,
        input_pattern: str | None = None,
        output_pattern: str | None = None,
        status_filter: str | None = None,
        max_results: int = 50,
        hours_back: int = 168,
        include_definition: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search executions with advanced filtering by state name and input/output patterns.

        Args:
            state_machine_arn: ARN of the state machine
            state_name: Filter by state name (supports regex)
            input_pattern: Regex pattern to match in state inputs
            output_pattern: Regex pattern to match in state outputs
            status_filter: Optional status filter
            max_results: Maximum number of executions to process
            hours_back: Number of hours to look back
            include_definition: If True, includes the state machine definition with each result

        Returns:
            List of matching executions with full state history
        """
        # First, list executions
        executions = self.list_executions(
            state_machine_arn=state_machine_arn,
            status_filter=status_filter,
            max_results=max_results,
            hours_back=hours_back,
        )

        # Get definition once if needed (avoid re-fetching for each execution)
        definition = None
        if include_definition:
            try:
                definition = self.get_state_machine_definition(state_machine_arn)
            except Exception:
                pass  # Silently fail if definition can't be retrieved

        # If no state/pattern filters, just enrich with history and return
        if not state_name and not input_pattern and not output_pattern:
            # Enrich with state history
            for execution in executions:
                try:
                    details = self.get_execution_details(execution["arn"])
                    execution["states"] = details["states"]
                    # Add input/output if available
                    if "input" in details:
                        execution["input"] = details["input"]
                    if "output" in details:
                        execution["output"] = details["output"]
                    if "error" in details:
                        execution["error"] = details["error"]
                    if "cause" in details:
                        execution["cause"] = details["cause"]
                    # Add definition if requested
                    if definition:
                        execution["stateMachineDefinition"] = definition
                except Exception as e:
                    execution["states"] = {}
                    execution["error"] = f"Failed to retrieve history: {str(e)}"
            return executions

        # Apply filters
        filtered = []
        for execution in executions:
            try:
                # Get full details with state history
                details = self.get_execution_details(execution["arn"])
                states = details["states"]

                # Apply filters
                if self._matches_filters(
                    states, state_name, input_pattern, output_pattern
                ):
                    # Merge details into execution
                    execution.update(details)
                    # Add definition if requested
                    if definition:
                        execution["stateMachineDefinition"] = definition
                    filtered.append(execution)

            except Exception:
                # Skip executions that fail to process
                continue

        return filtered

    def _matches_filters(
        self,
        states: dict[str, dict],
        state_name: str | None,
        input_pattern: str | None,
        output_pattern: str | None,
    ) -> bool:
        """
        Check if execution states match the given filters.

        Args:
            states: Parsed state history
            state_name: State name filter (regex)
            input_pattern: Input pattern filter (regex)
            output_pattern: Output pattern filter (regex)

        Returns:
            True if execution matches all filters
        """
        # Check state name filter
        if state_name:
            state_pattern = re.compile(state_name, re.IGNORECASE)
            matching_states = [s for s in states.keys() if state_pattern.search(s)]

            if not matching_states:
                return False

            states_to_check = matching_states
        else:
            states_to_check = list(states.keys())

        # Check input pattern
        if input_pattern:
            input_regex = re.compile(input_pattern, re.IGNORECASE)
            found_input = False

            for state in states_to_check:
                for inp in states[state].get("inputs", []):
                    if input_regex.search(inp):
                        found_input = True
                        break
                if found_input:
                    break

            if not found_input:
                return False

        # Check output pattern
        if output_pattern:
            output_regex = re.compile(output_pattern, re.IGNORECASE)
            found_output = False

            for state in states_to_check:
                for out in states[state].get("outputs", []):
                    if output_regex.search(out):
                        found_output = True
                        break
                if found_output:
                    break

            if not found_output:
                return False

        return True

    def get_state_machine_definition(self, state_machine_arn: str) -> dict[str, Any]:
        """
        Get the state machine definition including ASL and extracted resources.

        Args:
            state_machine_arn: ARN of the state machine

        Returns:
            Dictionary containing:
            - definition: Parsed ASL definition (dict)
            - resources: Extracted Lambda ARNs and other resources
            - metadata: State machine metadata (name, type, creation date, etc.)
        """
        # Get state machine description
        response = self.sfn_client.describe_state_machine(stateMachineArn=state_machine_arn)

        # Parse the definition (it's returned as a JSON string)
        definition = json.loads(response["definition"])

        # Extract resources from the definition
        resources = self._extract_resources_from_definition(definition)

        # Build result
        result = {
            "name": response["name"],
            "arn": response["stateMachineArn"],
            "type": response["type"],
            "status": response["status"],
            "creationDate": response["creationDate"].isoformat(),
            "roleArn": response["roleArn"],
            "definition": definition,
            "resources": resources,
        }

        # Add optional fields
        if "loggingConfiguration" in response:
            result["loggingConfiguration"] = response["loggingConfiguration"]
        if "tracingConfiguration" in response:
            result["tracingConfiguration"] = response["tracingConfiguration"]

        return result

    def _extract_resources_from_definition(
        self, definition: dict[str, Any]
    ) -> dict[str, list[str]]:
        """
        Extract Lambda ARNs and other resources from the state machine definition.

        Args:
            definition: Parsed ASL definition

        Returns:
            Dictionary with categorized resources:
            - lambdas: List of Lambda function ARNs
            - sns_topics: List of SNS topic ARNs
            - sqs_queues: List of SQS queue ARNs
            - dynamodb_tables: List of DynamoDB table ARNs
            - step_functions: List of nested Step Functions ARNs
            - other: List of other resource ARNs
        """
        resources: dict[str, list[str]] = {
            "lambdas": [],
            "sns_topics": [],
            "sqs_queues": [],
            "dynamodb_tables": [],
            "step_functions": [],
            "other": [],
        }

        def extract_from_state(state: dict[str, Any]) -> None:
            """Recursively extract resources from a state."""
            # Check for Resource field
            if "Resource" in state:
                resource = state["Resource"]

                # Categorize by ARN pattern
                if "lambda" in resource.lower():
                    if resource not in resources["lambdas"]:
                        resources["lambdas"].append(resource)
                elif "sns" in resource.lower():
                    if resource not in resources["sns_topics"]:
                        resources["sns_topics"].append(resource)
                elif "sqs" in resource.lower():
                    if resource not in resources["sqs_queues"]:
                        resources["sqs_queues"].append(resource)
                elif "dynamodb" in resource.lower():
                    if resource not in resources["dynamodb_tables"]:
                        resources["dynamodb_tables"].append(resource)
                elif "states" in resource.lower() and "stateMachine" in resource:
                    if resource not in resources["step_functions"]:
                        resources["step_functions"].append(resource)
                else:
                    if resource not in resources["other"]:
                        resources["other"].append(resource)

            # Handle Parallel and Map states with branches
            if "Branches" in state:
                for branch in state["Branches"]:
                    if "States" in branch:
                        for branch_state in branch["States"].values():
                            extract_from_state(branch_state)

            # Handle Catch blocks
            if "Catch" in state:
                for catch in state["Catch"]:
                    # Catch blocks don't have resources, but might have nested states
                    pass

        # Extract from all states
        if "States" in definition:
            for state in definition["States"].values():
                extract_from_state(state)

        return resources

    def get_execution_details_with_definition(
        self, execution_arn: str
    ) -> dict[str, Any]:
        """
        Get execution details along with the state machine definition.

        Args:
            execution_arn: ARN of the execution

        Returns:
            Execution details with state machine definition included
        """
        # Get execution details
        details = self.get_execution_details(execution_arn)

        # Get state machine definition
        definition = self.get_state_machine_definition(details["stateMachineArn"])

        # Add definition to result
        details["stateMachineDefinition"] = definition

        return details