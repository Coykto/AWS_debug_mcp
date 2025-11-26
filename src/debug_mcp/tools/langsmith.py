"""
LangSmith debugging tools.

Provides comprehensive debugging capabilities for LangSmith traces including:
- Listing projects
- Listing and filtering runs/traces
- Getting detailed run information with inputs/outputs
- Searching runs by various criteria (errors, latency, tags, metadata)

Supports multiple environments:
- 'prod'/'production': AWS Secrets Manager (PRODUCTION/env/vars)
- 'dev'/'development': AWS Secrets Manager (DEV/env/vars)
- 'local': Loads from .env file using python-dotenv
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from langsmith import Client


# Cache for debugger instances per environment
_debugger_cache: dict[str, "LangSmithDebugger"] = {}


def get_langsmith_debugger(environment: str) -> "LangSmithDebugger":
    """
    Get or create a LangSmith debugger for the specified environment.

    Args:
        environment: Environment name ('prod', 'dev', 'local')
                    - 'prod'/'production': Uses PRODUCTION/env/vars from Secrets Manager
                    - 'dev'/'development': Uses DEV/env/vars from Secrets Manager
                    - 'local': Loads from .env file using python-dotenv

    Returns:
        LangSmithDebugger instance configured for the environment
    """
    env_key = environment.lower()

    if env_key not in _debugger_cache:
        _debugger_cache[env_key] = LangSmithDebugger(environment=env_key)

    return _debugger_cache[env_key]


class LangSmithDebugger:
    """Core debugging functionality for LangSmith traces."""

    def __init__(
        self,
        environment: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
    ):
        """
        Initialize the LangSmith debugger.

        Credentials are loaded in the following order:
        1. Direct parameters (api_key, api_url)
        2. AWS Secrets Manager (if environment is specified)
        3. Environment variables (LANGCHAIN_API_KEY, LANGCHAIN_ENDPOINT)

        Args:
            environment: Environment name for Secrets Manager lookup ('prod', 'dev', 'local')
                        Maps to secret paths like 'PRODUCTION/env/vars' or 'DEV/env/vars'
            api_key: Direct API key (overrides other methods)
            api_url: Direct API URL (optional, defaults to LangSmith cloud)
        """
        self._api_key = api_key
        self._api_url = api_url
        self._environment = environment
        self._client: Client | None = None
        self._default_project: str | None = None

        # Try to initialize credentials
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from various sources."""
        # 1. Direct parameters already set
        if self._api_key:
            return

        # 2. Handle 'local' environment - load from .env file
        if self._environment and self._environment.lower() == "local":
            self._load_from_dotenv()
            return

        # 3. AWS Secrets Manager lookup for prod/dev environments
        if self._environment:
            self._load_from_secrets_manager()
            if self._api_key:
                return

        # 4. Environment variables fallback (no specific environment)
        self._api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        self._api_url = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
        self._default_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")

    def _load_from_dotenv(self) -> None:
        """Load credentials from .env file for local development."""
        try:
            from dotenv import load_dotenv

            # Load .env file (searches current directory and parents)
            load_dotenv()

            # Now read from environment variables
            self._api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
            self._api_url = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
            self._default_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")

        except ImportError:
            # python-dotenv not installed, fall back to env vars
            self._api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
            self._api_url = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
            self._default_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")

    def _load_from_secrets_manager(self) -> None:
        """Load credentials from AWS Secrets Manager."""
        # Map environment names to secret paths
        env_map = {
            "prod": "PRODUCTION/env/vars",
            "production": "PRODUCTION/env/vars",
            "dev": "DEV/env/vars",
            "development": "DEV/env/vars",
        }

        secret_name = env_map.get(self._environment.lower(), self._environment)

        try:
            region = os.getenv("AWS_REGION", "us-east-1")
            secrets_client = boto3.client("secretsmanager", region_name=region)

            response = secrets_client.get_secret_value(SecretId=secret_name)
            secret_data = json.loads(response["SecretString"])

            # Extract LangSmith/LangChain credentials
            self._api_key = secret_data.get("LANGCHAIN_API_KEY") or secret_data.get(
                "LANGSMITH_API_KEY"
            )
            self._api_url = secret_data.get("LANGCHAIN_ENDPOINT") or secret_data.get(
                "LANGSMITH_ENDPOINT"
            )
            self._default_project = secret_data.get("LANGCHAIN_PROJECT") or secret_data.get(
                "LANGSMITH_PROJECT"
            )

        except Exception:
            # Silently fail - will fall back to env vars
            pass

    @property
    def client(self) -> Client:
        """Get or create the LangSmith client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "LangSmith API key not configured. Set LANGCHAIN_API_KEY environment variable, "
                    "use AWS Secrets Manager with environment parameter, or provide api_key directly."
                )

            kwargs = {"api_key": self._api_key}
            if self._api_url:
                kwargs["api_url"] = self._api_url

            self._client = Client(**kwargs)

        return self._client

    @property
    def default_project(self) -> str | None:
        """Get the default project name if configured."""
        return self._default_project

    def list_projects(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        List available LangSmith projects.

        Args:
            limit: Maximum number of projects to return

        Returns:
            List of project metadata including name, id, creation date
        """
        projects = []

        for project in self.client.list_projects(limit=limit):
            projects.append(
                {
                    "name": project.name,
                    "id": str(project.id),
                    "created_at": (
                        project.created_at.isoformat() if project.created_at else None
                    ),
                    "description": project.description,
                    "reference_dataset_id": (
                        str(project.reference_dataset_id)
                        if project.reference_dataset_id
                        else None
                    ),
                }
            )

        return projects

    def list_runs(
        self,
        project_name: str | None = None,
        run_type: str | None = None,
        is_root: bool | None = None,
        error: bool | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        filter_str: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List runs/traces from a LangSmith project with filtering.

        Args:
            project_name: Project name (defaults to configured default project)
            run_type: Filter by run type (chain, llm, tool, retriever, embedding, prompt, parser)
            is_root: If True, return only root runs (top-level traces)
            error: If True, return only errored runs; if False, only successful runs
            start_time: Filter runs started after this time
            end_time: Filter runs started before this time
            limit: Maximum number of runs to return
            filter_str: Advanced filter string using LangSmith filter DSL
                       e.g., 'has(tags, "critical") and eq(error, true)'

        Returns:
            List of run metadata
        """
        project = project_name or self._default_project
        if not project:
            raise ValueError(
                "project_name required. Either pass it as parameter or configure "
                "LANGCHAIN_PROJECT environment variable."
            )

        kwargs: dict[str, Any] = {
            "project_name": project,
            "limit": limit,
        }

        if run_type:
            kwargs["run_type"] = run_type
        if is_root is not None:
            kwargs["is_root"] = is_root
        if error is not None:
            kwargs["error"] = error
        if start_time:
            kwargs["start_time"] = start_time
        if end_time:
            kwargs["end_time"] = end_time
        if filter_str:
            kwargs["filter"] = filter_str

        runs = []
        for run in self.client.list_runs(**kwargs):
            run_data = self._serialize_run(run)
            runs.append(run_data)

        return runs

    def get_run_details(
        self, run_id: str, include_children: bool = False
    ) -> dict[str, Any]:
        """
        Get detailed information about a specific run.

        Args:
            run_id: The run ID (UUID)
            include_children: If True, also fetch child runs

        Returns:
            Full run details including inputs, outputs, metadata
        """
        run = self.client.read_run(run_id)
        result = self._serialize_run(run, include_full_data=True)

        if include_children:
            # Fetch child runs
            children = []
            for child in self.client.list_runs(
                parent_run_id=run_id,
                limit=100,
            ):
                children.append(self._serialize_run(child, include_full_data=True))
            result["children"] = children

        return result

    def search_runs(
        self,
        project_name: str | None = None,
        query: str | None = None,
        run_type: str | None = None,
        error: bool | None = None,
        min_latency: float | None = None,
        max_latency: float | None = None,
        tags: list[str] | None = None,
        metadata_filter: dict[str, str] | None = None,
        hours_back: int = 24,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search runs with advanced filtering criteria.

        Args:
            project_name: Project name (defaults to configured default project)
            query: Text query to search in run names
            run_type: Filter by run type
            error: Filter by error status
            min_latency: Minimum latency in seconds
            max_latency: Maximum latency in seconds
            tags: Filter by tags (runs must have ALL specified tags)
            metadata_filter: Filter by metadata key-value pairs
            hours_back: Number of hours to look back
            limit: Maximum number of runs to return

        Returns:
            List of matching runs
        """
        project = project_name or self._default_project
        if not project:
            raise ValueError(
                "project_name required. Either pass it as parameter or configure "
                "LANGCHAIN_PROJECT environment variable."
            )

        # Build filter string
        filter_parts = []

        if min_latency is not None:
            filter_parts.append(f"gt(latency, {min_latency})")
        if max_latency is not None:
            filter_parts.append(f"lt(latency, {max_latency})")
        if tags:
            for tag in tags:
                filter_parts.append(f'has(tags, "{tag}")')
        if metadata_filter:
            for key, value in metadata_filter.items():
                filter_parts.append(f'eq(metadata_key, "{key}", "{value}")')

        filter_str = " and ".join(filter_parts) if filter_parts else None

        # Calculate time range
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        kwargs: dict[str, Any] = {
            "project_name": project,
            "limit": limit,
            "start_time": start_time,
        }

        if run_type:
            kwargs["run_type"] = run_type
        if error is not None:
            kwargs["error"] = error
        if filter_str:
            kwargs["filter"] = filter_str

        runs = []
        for run in self.client.list_runs(**kwargs):
            # Additional client-side filtering for query
            if query:
                run_name = run.name or ""
                if query.lower() not in run_name.lower():
                    continue

            runs.append(self._serialize_run(run))

        return runs

    def get_run_url(self, run_id: str) -> str:
        """
        Get the LangSmith UI URL for a run.

        Args:
            run_id: The run ID

        Returns:
            URL to view the run in LangSmith UI
        """
        return self.client.get_run_url(run_id=run_id)

    def _serialize_run(self, run: Any, include_full_data: bool = False) -> dict[str, Any]:
        """
        Serialize a run object to a dictionary.

        Args:
            run: LangSmith run object
            include_full_data: If True, include inputs/outputs/error details

        Returns:
            Serialized run data
        """
        data: dict[str, Any] = {
            "id": str(run.id),
            "name": run.name,
            "run_type": run.run_type,
            "status": run.status,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
        }

        # Add latency
        if run.start_time and run.end_time:
            latency = (run.end_time - run.start_time).total_seconds()
            data["latency_seconds"] = round(latency, 3)

        # Add error indicator
        data["error"] = run.error is not None

        # Add parent/child info
        if run.parent_run_id:
            data["parent_run_id"] = str(run.parent_run_id)

        # Add trace ID for cross-correlation with other systems
        if run.trace_id:
            data["trace_id"] = str(run.trace_id)

        # Add tags
        if run.tags:
            data["tags"] = run.tags

        # Add token counts
        if hasattr(run, "total_tokens") and run.total_tokens:
            data["total_tokens"] = run.total_tokens
        if hasattr(run, "prompt_tokens") and run.prompt_tokens:
            data["prompt_tokens"] = run.prompt_tokens
        if hasattr(run, "completion_tokens") and run.completion_tokens:
            data["completion_tokens"] = run.completion_tokens

        # Add link to UI
        try:
            data["link"] = self.get_run_url(str(run.id))
        except Exception:
            pass

        # Include full data if requested
        if include_full_data:
            if run.inputs:
                data["inputs"] = run.inputs
            if run.outputs:
                data["outputs"] = run.outputs
            if run.error:
                data["error_message"] = run.error
            if run.metadata:
                data["metadata"] = run.metadata

            # Include serialized info for debugging
            if hasattr(run, "serialized") and run.serialized:
                data["serialized"] = run.serialized

        return data