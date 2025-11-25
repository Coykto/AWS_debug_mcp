# AWS Debug MCP - Quick Start Checklist

Follow these steps to get your MCP server up and running.

## Prerequisites Checklist

- [ ] **uv installed**: Run `uv --version` (if not: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] **AWS credentials configured**: Run `aws sts get-caller-identity`
- [ ] **GitHub repo created**: Create at `github.com/your-username/aws-debug-mcp`
- [ ] **Claude Code installed**: You're using it already!

## Day 1: Bootstrap Project (30 minutes)

### 1. Create Repository

```bash
# Create and navigate to project directory
mkdir aws-debug-mcp
cd aws-debug-mcp

# Initialize git
git init
git branch -M main

# Add remote
git remote add origin git@github.com:your-username/aws-debug-mcp.git
```

### 2. Initialize Python Project

```bash
# Initialize with uv
uv init --lib

# Pin Python version
uv python pin 3.10

# Install core dependencies
uv add boto3 fastmcp pydantic

# Install dev dependencies
uv add --dev pytest pytest-cov ruff mypy
```

### 3. Create Project Structure

```bash
# Create directory structure
mkdir -p src/aws_debug_mcp/{tools,aws,utils}
mkdir -p tests examples

# Create __init__.py files
touch src/aws_debug_mcp/__init__.py
touch src/aws_debug_mcp/tools/__init__.py
touch src/aws_debug_mcp/aws/__init__.py
touch src/aws_debug_mcp/utils/__init__.py
touch tests/__init__.py
```

### 4. Copy Template Files

From `BOOTSTRAP.md`, copy these files:
- [ ] `pyproject.toml` - Project configuration
- [ ] `.gitignore` - Git ignore patterns
- [ ] `README.md` - User-facing documentation
- [ ] `.env.example` - Example environment variables

### 5. Implement Core Files

In order, create these files (code in BOOTSTRAP.md):
- [ ] `src/aws_debug_mcp/aws/config.py` - AWS configuration
- [ ] `src/aws_debug_mcp/aws/client_factory.py` - Boto3 client factory
- [ ] `src/aws_debug_mcp/tools/cloudwatch_logs.py` - CloudWatch tools
- [ ] `src/aws_debug_mcp/server.py` - MCP server setup
- [ ] `src/aws_debug_mcp/__main__.py` - Entry point

### 6. First Commit

```bash
git add .
git commit -m "Initial commit: AWS Debug MCP scaffold"
git push -u origin main
```

**Checkpoint**: Project structure is complete ✅

---

## Day 2: Implement & Test (2-3 hours)

### 7. Implement CloudWatch Logs Tools

Work through `src/aws_debug_mcp/tools/cloudwatch_logs.py`:
- [ ] Implement `describe_log_groups()`
- [ ] Implement `search_logs()` with CloudWatch Insights
- [ ] Implement `get_logs_for_timerange()`
- [ ] Implement `_parse_time_range()` helper
- [ ] Implement `_parse_relative_time()` helper

**Test manually**:
```bash
# Run Python interpreter
uv run python

>>> from aws_debug_mcp.aws.client_factory import AWSClientFactory
>>> from aws_debug_mcp.tools.cloudwatch_logs import CloudWatchLogsTools
>>>
>>> factory = AWSClientFactory()
>>> tools = CloudWatchLogsTools(factory)
>>>
>>> # Test listing log groups
>>> tools.describe_log_groups(prefix="/aws/lambda/")
```

### 8. Test MCP Server Locally

```bash
# Run the MCP server
uv run aws-debug-mcp

# Should start and wait for stdio input
# Press Ctrl+C to exit
```

### 9. Configure in Claude Code

Edit your `.mcp.json` (or Claude Code config):

```json
{
  "mcpServers": {
    "aws-debug-mcp-local": {
      "command": "uv",
      "args": ["run", "aws-debug-mcp"],
      "cwd": "/full/path/to/aws-debug-mcp",
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}
```

### 10. Test with Claude Code

Restart Claude Code and ask:
- "List my Lambda log groups" (should use `describe_log_groups`)
- "Show me errors from [log-group-name] in the last hour" (should use `search_logs`)

**Checkpoint**: MCP server works locally ✅

---

## Day 3: Polish & Release (1-2 hours)

### 11. Write Tests

Create `tests/test_cloudwatch_logs.py`:
- [ ] Test `describe_log_groups()`
- [ ] Test `_parse_time_range()` with various inputs
- [ ] Test `_parse_relative_time()`

Run tests:
```bash
uv run pytest
```

### 12. Update Documentation

- [ ] Update `README.md` with real GitHub URL
- [ ] Create `examples/usage_examples.md` with real use cases
- [ ] Create `CONTRIBUTING.md` for contributors
- [ ] Add `LICENSE` file (MIT recommended)

### 13. Add pyproject.toml Metadata

Ensure `pyproject.toml` has:
```toml
[project]
name = "aws-debug-mcp"
version = "0.1.0"
description = "MCP server for debugging AWS distributed systems"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

[project.urls]
Homepage = "https://github.com/your-username/aws-debug-mcp"
Repository = "https://github.com/your-username/aws-debug-mcp"
Issues = "https://github.com/your-username/aws-debug-mcp/issues"

[project.scripts]
aws-debug-mcp = "aws_debug_mcp.__main__:main"
```

### 14. Tag Release

```bash
git add .
git commit -m "Release v0.1.0: CloudWatch Logs support"
git tag v0.1.0
git push origin main --tags
```

### 15. Test Installation from GitHub

```bash
# Test in a different directory
cd /tmp

# Install via uvx
uvx --from git+https://github.com/your-username/aws-debug-mcp aws-debug-mcp

# Should work!
```

### 16. Update Claude Code Config

Update `.mcp.json` to install from GitHub:

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

**Checkpoint**: Public release complete ✅

---

## Day 4+: Share & Iterate

### 17. Share with Team

- [ ] Demo to Barley team
- [ ] Share GitHub link in Slack
- [ ] Update team's `.mcp.json` examples
- [ ] Collect feedback

### 18. Write About It

- [ ] LinkedIn post: "Building an MCP Server for AWS Debugging"
- [ ] Internal tech blog article
- [ ] Add to your resume/portfolio

### 19. Plan Next Phase

Based on feedback, prioritize:
- [ ] Step Functions tools
- [ ] ECS tools
- [ ] Lambda invocation tools
- [ ] Better error handling
- [ ] Performance improvements

---

## Troubleshooting

### MCP Server Won't Start

Check:
```bash
# Verify uv can run it
uv run python -m aws_debug_mcp

# Check for import errors
uv run python -c "from aws_debug_mcp.server import mcp; print('OK')"
```

### AWS Credentials Not Working

```bash
# Verify AWS access
aws sts get-caller-identity --profile your-profile-name

# Check environment variables
echo $AWS_PROFILE
echo $AWS_REGION
```

### Claude Code Not Seeing Tools

1. Restart Claude Code completely
2. Check MCP config syntax (valid JSON)
3. Check logs: Claude Code → Settings → MCP → View Logs

### uvx Installation Fails

```bash
# Try installing dependencies manually first
cd aws-debug-mcp
uv sync

# Then try uvx again
uvx --from git+https://github.com/your-username/aws-debug-mcp aws-debug-mcp
```

---

## Success Criteria

You're done when:
- ✅ MCP server runs locally
- ✅ Claude Code can list log groups
- ✅ Claude Code can search logs
- ✅ Installable via `uvx --from git+...`
- ✅ README has clear examples
- ✅ At least one team member tested it

---

## Time Estimates

- **Setup & bootstrap**: 30 minutes
- **Core implementation**: 2-3 hours
- **Testing & polish**: 1-2 hours
- **Documentation**: 1 hour
- **Total**: ~5-7 hours (spread across 2-3 days)

---

## Need Help?

Refer to:
- [BOOTSTRAP.md](./BOOTSTRAP.md) - Complete technical guide
- [FastMCP docs](https://github.com/jlowin/fastmcp)
- [Boto3 CloudWatch Logs](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/logs.html)
- Your existing `scripts/search_aws_executions.py` for boto3 patterns

**Good luck! You got this.** 🚀