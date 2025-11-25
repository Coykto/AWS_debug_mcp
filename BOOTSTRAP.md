# AWS Debug MCP - Project Bootstrap Guide

**Purpose**: A lightweight, team-friendly MCP server for debugging AWS distributed systems (Lambda, Step Functions, ECS) from Claude Code or any MCP client.

**Installation Model**: Like `serena` - installable via `uvx --from git+https://github.com/your-org/aws-debug-mcp`

---

## Project Goals

### Core Objectives
1. **Solve immediate debugging pain**: Reduce time spent clicking through AWS console
2. **Team-shareable**: Environment-based auth, easy to install and configure
3. **Modular design**: Start with CloudWatch Logs, expand to Step Functions/ECS
4. **Open source ready**: Clean architecture, good docs, potential community use

### Non-Goals (for MVP)
- AI-powered log analysis (future enhancement)
- Multi-cloud support (AWS only)
- Complex correlation algorithms (time-based only)

---

## Technical Architecture

### High-Level Design

```
┌─────────────────────────────────────────┐
│         MCP Client (Claude Code)        │
└───────────────┬─────────────────────────┘
                │ stdio
                ↓
┌─────────────────────────────────────────┐
│      AWS Debug MCP Server (Python)      │
│  ┌─────────────────────────────────┐   │
│  │     MCP Server Framework        │   │
│  │  (FastMCP or python-sdk-mcp)    │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│  ┌──────────────┴──────────────────┐   │
│  │      Tool Modules               │   │
│  │  • cloudwatch_logs.py           │   │
│  │  • step_functions.py (future)   │   │
│  │  • ecs_logs.py (future)         │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│  ┌──────────────┴──────────────────┐   │
│  │      AWS Client Factory         │   │
│  │   (boto3, env-based auth)       │   │
│  └─────────────────────────────────┘   │
└───────────────┬─────────────────────────┘
                │ AWS API calls
                ↓
┌─────────────────────────────────────────┐
│   AWS Services (CloudWatch, etc.)      │
└─────────────────────────────────────────┘
```

### Technology Stack

- **Language**: Python 3.10+
- **MCP Framework**: FastMCP (recommended) or official MCP Python SDK
- **AWS SDK**: boto3
- **Package Manager**: uv (for modern Python packaging)
- **Project Structure**: src-layout with pyproject.toml

---

## Project Structure

```
aws-debug-mcp/
├── pyproject.toml              # Project metadata, dependencies, entrypoints
├── README.md                   # User-facing documentation
├── CONTRIBUTING.md             # Development guide
├── LICENSE                     # Open source license (MIT recommended)
├── .env.example                # Example environment variables
├── .gitignore                  # Standard Python gitignore
│
├── src/
│   └── aws_debug_mcp/
│       ├── __init__.py         # Package init, version
│       ├── __main__.py         # Entry point for `python -m aws_debug_mcp`
│       ├── server.py           # Main MCP server setup and registration
│       │
│       ├── tools/              # Tool implementations
│       │   ├── __init__.py
│       │   ├── cloudwatch_logs.py    # CloudWatch Logs tools
│       │   ├── step_functions.py     # Step Functions tools (future)
│       │   └── ecs.py                # ECS tools (future)
│       │
│       ├── aws/                # AWS client management
│       │   ├── __init__.py
│       │   ├── client_factory.py     # Creates boto3 clients with proper auth
│       │   └── config.py             # AWS configuration from env vars
│       │
│       └── utils/              # Shared utilities
│           ├── __init__.py
│           ├── time_utils.py         # Time range parsing
│           └── formatters.py         # Output formatting
│
├── tests/                      # Tests (pytest)
│   ├── __init__.py
│   ├── test_cloudwatch_logs.py
│   └── test_client_factory.py
│
└── examples/                   # Example configurations
    ├── .mcp.json               # Example MCP config for Claude Code
    └── usage_examples.md       # Example queries
```

---

## Setup Instructions

### Prerequisites

1. **Install uv** (modern Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **AWS Credentials**:
   - Configure via `aws configure` OR
   - Set environment variables: `AWS_PROFILE`, `AWS_REGION`

### Initialize Project

```bash
# Create new directory
mkdir aws-debug-mcp
cd aws-debug-mcp

# Initialize with uv
uv init --lib

# Set Python version
uv python pin 3.10

# Install dependencies
uv add boto3 fastmcp pydantic
uv add --dev pytest pytest-cov ruff mypy

# Create src-layout structure
mkdir -p src/aws_debug_mcp/{tools,aws,utils}
mkdir -p tests examples
```

### pyproject.toml Configuration

```toml
[project]
name = "aws-debug-mcp"
version = "0.1.0"
description = "MCP server for debugging AWS distributed systems"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "boto3>=1.35.0",
    "fastmcp>=0.2.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
aws-debug-mcp = "aws_debug_mcp.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
```

---

## Implementation Guide

### Step 1: AWS Client Factory

**File**: `src/aws_debug_mcp/aws/client_factory.py`

```python
"""AWS client factory with environment-based authentication."""
import os
from typing import Optional
import boto3
from botocore.config import Config


class AWSClientFactory:
    """Factory for creating boto3 clients with proper configuration."""

    def __init__(self):
        self.profile = os.getenv("AWS_PROFILE")
        self.region = os.getenv("AWS_REGION", "us-east-1")

        # Create session
        self.session = boto3.Session(
            profile_name=self.profile,
            region_name=self.region
        )

        # Configure boto3 client settings
        self.config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=60,
        )

    def get_client(self, service_name: str):
        """Get boto3 client for specified AWS service."""
        return self.session.client(service_name, config=self.config)
```

**File**: `src/aws_debug_mcp/aws/config.py`

```python
"""AWS configuration from environment."""
from dataclasses import dataclass
import os


@dataclass
class AWSConfig:
    """AWS configuration."""
    profile: str | None
    region: str

    @classmethod
    def from_env(cls) -> "AWSConfig":
        """Load configuration from environment variables."""
        return cls(
            profile=os.getenv("AWS_PROFILE"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
```

### Step 2: CloudWatch Logs Tools

**File**: `src/aws_debug_mcp/tools/cloudwatch_logs.py`

```python
"""CloudWatch Logs tools for MCP."""
from typing import Any
from datetime import datetime, timedelta
from fastmcp import Tool


class CloudWatchLogsTools:
    """CloudWatch Logs tool implementations."""

    def __init__(self, client_factory):
        self.client_factory = client_factory
        self.logs_client = client_factory.get_client("logs")

    def describe_log_groups(self, prefix: str = "") -> dict[str, Any]:
        """
        List CloudWatch log groups.

        Args:
            prefix: Filter log groups by prefix (e.g., /aws/lambda/)

        Returns:
            List of log groups with metadata
        """
        params = {}
        if prefix:
            params["logGroupNamePrefix"] = prefix

        response = self.logs_client.describe_log_groups(**params)

        log_groups = []
        for lg in response.get("logGroups", []):
            log_groups.append({
                "name": lg["logGroupName"],
                "creation_time": lg.get("creationTime"),
                "stored_bytes": lg.get("storedBytes", 0),
                "retention_days": lg.get("retentionInDays", "Never expires"),
            })

        return {
            "log_groups": log_groups,
            "count": len(log_groups)
        }

    def search_logs(
        self,
        log_group: str,
        query: str,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100
    ) -> dict[str, Any]:
        """
        Search logs using CloudWatch Insights query.

        Args:
            log_group: Log group name
            query: CloudWatch Insights query (e.g., "fields @timestamp, @message | filter @message like /ERROR/")
            start_time: Start time (ISO format or relative like "-1h")
            end_time: End time (ISO format or relative like "now")
            limit: Maximum number of results

        Returns:
            Query results with log entries
        """
        # Parse time range
        start_ts, end_ts = self._parse_time_range(start_time, end_time)

        # Start query
        response = self.logs_client.start_query(
            logGroupName=log_group,
            startTime=start_ts,
            endTime=end_ts,
            queryString=query,
            limit=limit
        )

        query_id = response["queryId"]

        # Poll for results (timeout after 30 seconds)
        import time
        for _ in range(30):
            result = self.logs_client.get_query_results(queryId=query_id)
            status = result["status"]

            if status == "Complete":
                return {
                    "status": "success",
                    "results": result["results"],
                    "statistics": result.get("statistics", {})
                }
            elif status == "Failed":
                return {"status": "failed", "error": "Query failed"}

            time.sleep(1)

        return {"status": "timeout", "error": "Query timed out"}

    def get_logs_for_timerange(
        self,
        log_group: str,
        start_time: str,
        end_time: str,
        filter_pattern: str = "",
        limit: int = 100
    ) -> dict[str, Any]:
        """
        Get raw logs for a specific time range.

        Args:
            log_group: Log group name
            start_time: Start time (ISO format or relative)
            end_time: End time (ISO format or relative)
            filter_pattern: CloudWatch filter pattern
            limit: Maximum number of events

        Returns:
            Log events
        """
        start_ts, end_ts = self._parse_time_range(start_time, end_time)

        params = {
            "logGroupName": log_group,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }

        if filter_pattern:
            params["filterPattern"] = filter_pattern

        response = self.logs_client.filter_log_events(**params)

        events = []
        for event in response.get("events", []):
            events.append({
                "timestamp": event["timestamp"],
                "message": event["message"],
                "log_stream": event.get("logStreamName")
            })

        return {
            "events": events,
            "count": len(events),
            "time_range": {
                "start": start_time,
                "end": end_time
            }
        }

    def _parse_time_range(
        self,
        start_time: str | None,
        end_time: str | None
    ) -> tuple[int, int]:
        """Parse time range into Unix timestamps (milliseconds)."""
        now = datetime.now()

        if not end_time or end_time == "now":
            end_dt = now
        else:
            # Try parsing as relative time
            if end_time.startswith("-"):
                delta = self._parse_relative_time(end_time)
                end_dt = now + delta
            else:
                # Parse as ISO format
                end_dt = datetime.fromisoformat(end_time)

        if not start_time:
            # Default to 1 hour ago
            start_dt = end_dt - timedelta(hours=1)
        elif start_time.startswith("-"):
            delta = self._parse_relative_time(start_time)
            start_dt = now + delta
        else:
            start_dt = datetime.fromisoformat(start_time)

        # Convert to milliseconds
        return (
            int(start_dt.timestamp() * 1000),
            int(end_dt.timestamp() * 1000)
        )

    def _parse_relative_time(self, relative: str) -> timedelta:
        """Parse relative time string like '-1h', '-30m', '-2d'."""
        value = int(relative[:-1])
        unit = relative[-1]

        if unit == "h":
            return timedelta(hours=value)
        elif unit == "m":
            return timedelta(minutes=value)
        elif unit == "d":
            return timedelta(days=value)
        else:
            raise ValueError(f"Unknown time unit: {unit}")
```

### Step 3: Main MCP Server

**File**: `src/aws_debug_mcp/server.py`

```python
"""Main MCP server setup."""
from fastmcp import FastMCP
from .aws.client_factory import AWSClientFactory
from .tools.cloudwatch_logs import CloudWatchLogsTools


# Initialize MCP server
mcp = FastMCP("aws-debug-mcp")

# Initialize AWS client factory
client_factory = AWSClientFactory()

# Initialize tool modules
cloudwatch_tools = CloudWatchLogsTools(client_factory)


# Register tools
@mcp.tool()
def describe_log_groups(prefix: str = "") -> dict:
    """
    List CloudWatch log groups.

    Args:
        prefix: Filter log groups by prefix (e.g., /aws/lambda/, /ecs/)
    """
    return cloudwatch_tools.describe_log_groups(prefix)


@mcp.tool()
def search_logs(
    log_group: str,
    query: str,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 100
) -> dict:
    """
    Search logs using CloudWatch Insights query.

    Args:
        log_group: Log group name
        query: CloudWatch Insights query
        start_time: Start time (ISO or relative like "-1h")
        end_time: End time (ISO or relative like "now")
        limit: Maximum results
    """
    return cloudwatch_tools.search_logs(
        log_group=log_group,
        query=query,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )


@mcp.tool()
def get_logs_for_timerange(
    log_group: str,
    start_time: str,
    end_time: str,
    filter_pattern: str = "",
    limit: int = 100
) -> dict:
    """
    Get raw logs for a specific time range.

    Args:
        log_group: Log group name
        start_time: Start time (ISO or relative)
        end_time: End time (ISO or relative)
        filter_pattern: CloudWatch filter pattern (optional)
        limit: Maximum events
    """
    return cloudwatch_tools.get_logs_for_timerange(
        log_group=log_group,
        start_time=start_time,
        end_time=end_time,
        filter_pattern=filter_pattern,
        limit=limit
    )
```

**File**: `src/aws_debug_mcp/__main__.py`

```python
"""Entry point for aws-debug-mcp."""
from .server import mcp


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
```

### Step 4: Make it Installable

**Create** `README.md`:

```markdown
# AWS Debug MCP

MCP server for debugging AWS distributed systems (Lambda, Step Functions, ECS) from Claude Code.

## Installation

Install directly from GitHub using uvx:

```bash
uvx --from git+https://github.com/your-username/aws-debug-mcp aws-debug-mcp
```

Or add to your MCP configuration:

```json
{
  "mcpServers": {
    "aws-debug-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/your-username/aws-debug-mcp",
        "aws-debug-mcp"
      ],
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

## Configuration

Set these environment variables:
- `AWS_PROFILE`: AWS profile name (optional)
- `AWS_REGION`: AWS region (default: us-east-1)

## Available Tools

### describe_log_groups
List CloudWatch log groups with optional prefix filter.

### search_logs
Search logs using CloudWatch Insights queries.

### get_logs_for_timerange
Get raw logs for a specific time window.

## Examples

Ask Claude:
- "Show me Lambda errors from the last hour"
- "List all ECS log groups"
- "Search for ERROR in log group /aws/lambda/my-function between 2pm and 3pm"

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
```

**Create** `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.coverage
htmlcov/
.pytest_cache/

# Environment
.env
.env.local

# OS
.DS_Store
```

---

## Usage Examples

### Claude Code Configuration

**File**: `examples/.mcp.json`

```json
{
  "mcpServers": {
    "aws-debug-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/your-username/aws-debug-mcp",
        "aws-debug-mcp"
      ],
      "env": {
        "AWS_PROFILE": "AdministratorAccess-381492197841",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

### Example Queries

**File**: `examples/usage_examples.md`

```markdown
# Usage Examples

## List Log Groups

"Show me all Lambda log groups"
→ Tool: describe_log_groups(prefix="/aws/lambda/")

## Search for Errors

"Find errors in the fireflies-processing Lambda from the last 2 hours"
→ Tool: search_logs(
    log_group="/aws/lambda/fireflies-processing",
    query="fields @timestamp, @message | filter @message like /ERROR/",
    start_time="-2h",
    end_time="now"
  )

## Get Logs in Time Range

"Show me all logs from my-service between 2pm and 3pm today"
→ Tool: get_logs_for_timerange(
    log_group="/ecs/my-service",
    start_time="2025-01-15T14:00:00",
    end_time="2025-01-15T15:00:00"
  )

## Debugging Step Function

"Find logs related to Step Function execution arn:aws:states:us-east-1:123:execution:MyStateMachine:abc123"
→ Extract execution time window
→ Search relevant Lambda/ECS logs in that window
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_cloudwatch_logs.py`

```python
"""Tests for CloudWatch Logs tools."""
import pytest
from unittest.mock import Mock, patch
from aws_debug_mcp.tools.cloudwatch_logs import CloudWatchLogsTools


@pytest.fixture
def mock_client_factory():
    factory = Mock()
    factory.get_client.return_value = Mock()
    return factory


def test_describe_log_groups(mock_client_factory):
    """Test listing log groups."""
    mock_logs = mock_client_factory.get_client.return_value
    mock_logs.describe_log_groups.return_value = {
        "logGroups": [
            {
                "logGroupName": "/aws/lambda/test-function",
                "creationTime": 1234567890,
                "storedBytes": 1024
            }
        ]
    }

    tools = CloudWatchLogsTools(mock_client_factory)
    result = tools.describe_log_groups()

    assert result["count"] == 1
    assert result["log_groups"][0]["name"] == "/aws/lambda/test-function"
```

### Integration Tests (Manual)

1. Set up AWS credentials
2. Run against real AWS account (test/dev)
3. Verify tools return expected results
4. Test error handling

---

## Development Workflow

### Initial Setup

```bash
# Clone repo
git clone https://github.com/your-username/aws-debug-mcp
cd aws-debug-mcp

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

### Testing Locally

```bash
# Run MCP server locally
uv run aws-debug-mcp

# Test with MCP inspector (if available)
# Or configure in Claude Code and test
```

### Release Process

1. Update version in `pyproject.toml`
2. Tag release: `git tag v0.1.0`
3. Push: `git push origin v0.1.0`
4. Users can install: `uvx --from git+https://github.com/you/aws-debug-mcp@v0.1.0 aws-debug-mcp`

---

## Next Steps

### Phase 1: MVP (CloudWatch Logs only)
- [ ] Set up project structure
- [ ] Implement AWS client factory
- [ ] Implement CloudWatch Logs tools
- [ ] Write README and examples
- [ ] Test locally
- [ ] Push to GitHub
- [ ] Test installation via uvx

### Phase 2: Expand Tools
- [ ] Add Step Functions tools (execution details)
- [ ] Add ECS task log tools
- [ ] Add Lambda invocation tools

### Phase 3: Polish
- [ ] Comprehensive tests
- [ ] Better error handling
- [ ] Performance optimization
- [ ] Documentation improvements

---

## Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Boto3 CloudWatch Logs](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/logs.html)
- [Serena MCP (reference)](https://github.com/oraios/serena)