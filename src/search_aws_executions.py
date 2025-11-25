"""
AWS Step Functions Execution Search Tool

A powerful tool for searching and filtering AWS Step Functions executions with smart caching
and auto-generated filenames based on your search criteria.

WORKFLOW:
    1. Download executions once with full history (creates executions.json)
    2. Apply various filters without re-downloading (auto-uses executions.json)
    3. Each filter creates a descriptively named output file

EXAMPLES:

    # Initial download - get all executions with full history
    python search_aws_executions.py \\
        --state-machine-arn "arn:aws:states:us-west-2:381492197841:stateMachine:production_fireflies_processing"
    # Creates: executions.json

    # Find manually matched executions (auto-uses executions.json)
    python search_aws_executions.py --state-name "Send_No_Project_Slack_Message"
    # Creates: filtered_state_send_no_project_slack_message.json

    # Find executions that assigned call to a company
    python search_aws_executions.py \\
        --state-name "Match_Transcript_To_Entity" \\
        --output-pattern "entity_type\\":\\s*\\"company"
    # Creates: filtered_state_match_transcript_to_entity_output_entity_type_s_company.json

    # Find executions that assigned call to a deal
    python search_aws_executions.py \\
        --state-name "Match_Transcript_To_Entity" \\
        --output-pattern "entity_type\\":\\s*\\"deal"
    # Creates: filtered_state_match_transcript_to_entity_output_entity_type_s_deal.json

    # Find failed executions with timeout errors
    python search_aws_executions.py --output-pattern "timeout|TimeoutError"
    # Creates: filtered_output_timeout_timeouterror.json

    # Search in a different cached file
    python search_aws_executions.py \\
        --input-file previous_executions.json \\
        --state-name "ProcessPayment"
    # Creates: filtered_state_processpayment.json

FEATURES:
    - Smart caching: Download once, filter many times
    - Auto-detects executions.json when no input specified
    - Auto-generates descriptive filenames from filters
    - Regex support for flexible pattern matching
    - Preserves AWS console links for each execution
    - Filters only Step Functions states (ignores Lambda events)
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Callable

import boto3


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to a filesystem-safe slug.

    Args:
        text: Text to slugify
        max_length: Maximum length of the slug

    Returns:
        Filesystem-safe slug
    """
    if not text:
        return ""

    # Replace non-alphanumeric characters with underscores
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text)

    # Remove leading/trailing underscores
    slug = slug.strip("_")

    # Truncate to max length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")

    return slug.lower()


def generate_output_filename(
    status_filter: str | None = None,
    state_name: str | None = None,
    input_pattern: str | None = None,
    output_pattern: str | None = None,
    input_file: str | None = None,
) -> str:
    """
    Generate output filename based on filters applied.

    Args:
        status_filter: Execution status filter
        state_name: State name filter
        input_pattern: Input pattern filter
        output_pattern: Output pattern filter
        input_file: Input file used (if any)

    Returns:
        Generated filename
    """
    parts = []

    # Base name
    if input_file:
        parts.append("filtered")
    else:
        parts.append("executions")

    # Add filter components
    if status_filter:
        parts.append(f"status_{status_filter.lower()}")

    if state_name:
        # Simplify state name for filename
        state_slug = slugify(state_name, 30)
        parts.append(f"state_{state_slug}")

    if input_pattern:
        # Extract key parts from the pattern for the filename
        # Try to find meaningful parts like entity names, IDs, etc.
        pattern_slug = slugify(input_pattern, 30)
        parts.append(f"input_{pattern_slug}")

    if output_pattern:
        # Extract key parts from the pattern
        pattern_slug = slugify(output_pattern, 30)
        parts.append(f"output_{pattern_slug}")

    # If no filters were applied, keep it simple
    if len(parts) == 1:
        return f"{parts[0]}.json"

    return "_".join(parts) + ".json"


class StepFunctionExecutionSearcher:
    """Search and filter Step Functions executions with flexible criteria."""

    def __init__(
        self,
        state_machine_arn: str,
        region: str = "us-west-2",
        hours_back: int = 360,  # 15 days * 24 hours
    ):
        """
        Initialize the searcher with a specific state machine.

        Args:
            state_machine_arn: Full ARN of the state machine
            region: AWS region (default: us-west-2)
            hours_back: Number of hours to look back for executions (default: 360 = 15 days)
        """
        self.state_machine_arn = state_machine_arn
        self.region = region
        self.hours_back = hours_back
        self.sfn_client = boto3.client("stepfunctions", region_name=region)

        # Extract state machine name and account ID from ARN for link generation
        arn_parts = state_machine_arn.split(":")
        self.account_id = arn_parts[4]
        self.state_machine_name = arn_parts[-1]

        # Calculate cutoff date
        self.cutoff_date = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    def generate_execution_link(self, execution_name: str) -> str:
        """Generate AWS console link for an execution."""
        base_url = f"https://{self.region}.console.aws.amazon.com/states/home"
        execution_arn = f"arn:aws:states:{self.region}:{self.account_id}:execution:{self.state_machine_name}:{execution_name}"
        return f"{base_url}?region={self.region}#/v2/executions/details/{execution_arn}"

    def download_executions(self, status_filter: str | None = None) -> list[dict]:
        """
        Download all executions for the state machine within the date range.

        Args:
            status_filter: Optional status filter (RUNNING, SUCCEEDED, FAILED, etc.)

        Returns:
            List of execution dictionaries
        """
        executions = []
        paginator = self.sfn_client.get_paginator("list_executions")

        pagination_params = {"stateMachineArn": self.state_machine_arn}
        if status_filter:
            pagination_params["statusFilter"] = status_filter

        for page in paginator.paginate(**pagination_params):
            for execution in page["executions"]:
                if execution["startDate"] >= self.cutoff_date:
                    executions.append(
                        {
                            "name": execution["name"],
                            "arn": execution["executionArn"],
                            "status": execution["status"],
                            "startDate": execution["startDate"].isoformat(),
                            "link": self.generate_execution_link(execution["name"]),
                        }
                    )
                else:
                    # Executions are returned in reverse chronological order
                    break

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

    def enrich_with_history(self, executions: list[dict]) -> list[dict]:
        """
        Enrich executions with their state history.

        Args:
            executions: List of execution dictionaries

        Returns:
            Enriched execution list with history
        """
        for idx, execution in enumerate(executions):
            print(f"Processing execution {idx + 1} of {len(executions)}: {execution['name']}")

            history = []
            history_paginator = self.sfn_client.get_paginator("get_execution_history")

            try:
                for page in history_paginator.paginate(executionArn=execution["arn"]):
                    history.extend(page["events"])

                execution["states"] = self.parse_state_history(history)
            except Exception as e:
                print(f"  Error processing execution {execution['name']}: {e}")
                execution["states"] = {}

        return executions

    def create_filter(
        self,
        state_name: str | None = None,
        input_pattern: str | None = None,
        output_pattern: str | None = None,
        custom_filter: Callable[[dict], bool] | None = None,
    ) -> Callable[[dict], bool]:
        """
        Create a filter function for executions.

        Args:
            state_name: Filter by state name (can use regex)
            input_pattern: Regex pattern to match in state inputs
            output_pattern: Regex pattern to match in state outputs
            custom_filter: Custom filter function that takes an execution dict

        Returns:
            Filter function
        """

        def filter_execution(execution: dict) -> bool:
            # Apply custom filter first if provided
            if custom_filter and not custom_filter(execution):
                return False

            states = execution.get("states", {})

            # Check state name filter
            if state_name:
                state_pattern = re.compile(state_name)
                matching_states = [s for s in states.keys() if state_pattern.search(s)]

                if not matching_states:
                    return False

                # If we have input/output patterns, only check matching states
                states_to_check = matching_states
            else:
                states_to_check = states.keys()

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

        return filter_execution

    def search(
        self,
        status_filter: str | None = None,
        state_name: str | None = None,
        input_pattern: str | None = None,
        output_pattern: str | None = None,
        custom_filter: Callable[[dict], bool] | None = None,
        input_file: str | None = None,
    ) -> list[dict]:
        """
        Search and filter executions.

        Args:
            status_filter: Execution status filter
            state_name: Filter by state name
            input_pattern: Pattern to search in inputs
            output_pattern: Pattern to search in outputs
            custom_filter: Custom filter function
            input_file: Optional JSON file to load executions from instead of downloading

        Returns:
            List of filtered executions
        """
        if input_file:
            print(f"Loading executions from {input_file}...")
            with open(input_file, "r") as f:
                executions = json.load(f)
            print(f"Loaded {len(executions)} executions from file")

            # Check if we need to apply filters
            if state_name or input_pattern or output_pattern or custom_filter:
                # Check if executions have states data
                if executions and "states" not in executions[0]:
                    print(
                        "Warning: Loaded executions don't have state history. Filters may not work properly."
                    )
        else:
            print(f"Downloading executions for {self.state_machine_name}...")
            executions = self.download_executions(status_filter)
            print(
                f"Found {len(executions)} executions in the last {self.hours_back} hours ({self.hours_back / 24:.1f} days)"
            )

            print("Enriching with execution history...")
            executions = self.enrich_with_history(executions)

        # Apply filters if any are specified
        if state_name or input_pattern or output_pattern or custom_filter:
            print("Applying filters...")

            filter_func = self.create_filter(
                state_name=state_name,
                input_pattern=input_pattern,
                output_pattern=output_pattern,
                custom_filter=custom_filter,
            )

            executions = [e for e in executions if filter_func(e)]

            print(f"Found {len(executions)} matching executions")

        return executions

    def save_results(
        self,
        executions: list[dict],
        output_file: str,
    ) -> str:
        """
        Save search results to JSON file.

        Args:
            executions: List of executions to save
            output_file: Output file path

        Returns:
            The output file path
        """
        with open(output_file, "w") as f:
            json.dump(executions, f, indent=2, default=str)

        print(f"\n✅ Saved {len(executions)} executions")
        print(f"📄 Output file: {output_file}")
        return output_file


def main():
    """Command-line interface for the Step Functions searcher."""
    parser = argparse.ArgumentParser(
        description="Search and filter AWS Step Functions executions with smart caching",
        epilog="""
EXAMPLES:
  # Download all executions with history:
  %(prog)s --state-machine-arn "arn:aws:states:us-west-2:123456:stateMachine:my-state-machine"

  # Filter cached executions (auto-uses executions.json):
  %(prog)s --state-name "Send_No_Project_Slack_Message"

  # Search for specific entity type in outputs:
  %(prog)s --output-pattern '"entity_type":\\s*"company"'

  # Combine multiple filters:
  %(prog)s --state-name "Match.*Entity" --output-pattern "deal"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--state-machine-arn",
        help="ARN of the Step Functions state machine (required for initial download)",
    )

    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )

    parser.add_argument(
        "--hours-back",
        type=int,
        default=360,
        help="Time window for downloading executions in hours (default: 360 = 15 days)",
    )

    parser.add_argument(
        "--input-file",
        help="Load from cached JSON file (defaults to executions.json if exists)",
    )

    parser.add_argument(
        "--status",
        choices=["RUNNING", "SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"],
        help="Filter by execution status (only when downloading from AWS)",
    )

    parser.add_argument(
        "--state-name",
        help='Filter by state name (regex supported, e.g. "Match.*Entity")',
    )

    parser.add_argument(
        "--input-pattern",
        help='Search pattern in state inputs (regex, e.g. "customer_id.*12345")',
    )

    parser.add_argument(
        "--output-pattern",
        help='Search pattern in state outputs (regex, e.g. "entity_type.*company")',
    )

    parser.add_argument(
        "--output-file",
        help="Custom output filename (default: auto-generated from filters)",
    )

    args = parser.parse_args()

    # Default to executions.json if neither input file nor state machine ARN provided
    if not args.input_file and not args.state_machine_arn:
        import os

        if os.path.exists("executions.json"):
            print("No input file or state machine ARN specified. Using default: executions.json")
            args.input_file = "executions.json"
        else:
            parser.error(
                "Either --state-machine-arn or --input-file must be provided (executions.json not found)"
            )

    if args.input_file:
        # When using input file, we don't need AWS connection
        # Create a minimal searcher just for filtering
        # Use dummy ARN if loading from file
        searcher = StepFunctionExecutionSearcher(
            state_machine_arn="arn:aws:states:us-west-2:000000000000:stateMachine:dummy",
            region=args.region,
            hours_back=args.hours_back,
        )
    else:
        # Initialize searcher with actual AWS connection
        searcher = StepFunctionExecutionSearcher(
            state_machine_arn=args.state_machine_arn,
            region=args.region,
            hours_back=args.hours_back,
        )

    # Perform search
    results = searcher.search(
        status_filter=args.status if not args.input_file else None,
        state_name=args.state_name,
        input_pattern=args.input_pattern,
        output_pattern=args.output_pattern,
        input_file=args.input_file,
    )

    # Generate output filename if not provided
    if not args.output_file:
        output_file = generate_output_filename(
            status_filter=args.status if not args.input_file else None,
            state_name=args.state_name,
            input_pattern=args.input_pattern,
            output_pattern=args.output_pattern,
            input_file=args.input_file,
        )
    else:
        output_file = args.output_file

    # Save results
    searcher.save_results(
        results,
        output_file,
    )


if __name__ == "__main__":
    main()
