# AWS Debug MCP

**Status**: Phase 1 MVP Complete ✅
**Version**: 0.1.0

Open-source MCP server for debugging AWS distributed systems (Lambda, Step Functions, ECS) directly from Claude Code or any MCP client.

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
        "git+https://github.com/YOUR_USERNAME/aws-debug-mcp",
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

Replace:
- `YOUR_USERNAME` with the actual GitHub username/org
- `your-aws-profile-name` with your AWS profile name from `~/.aws/credentials`
- `us-east-1` with your preferred AWS region

### Prerequisites

- Python 3.11+
- `uv` or `uvx` installed ([installation guide](https://docs.astral.sh/uv/))
- AWS credentials configured (via `aws configure` or environment variables)

## How It Works

This MCP server acts as a **proxy/gateway** to AWS MCP servers. It:
1. Spawns AWS MCP servers (like awslabs.cloudwatch-mcp-server) as subprocesses
2. Forwards only the tools you select, hiding the rest
3. Keeps your tool list minimal and focused on debugging

**No reimplementation** - AWS maintains the underlying functionality, we just filter which tools are exposed.

## Available Tools

### CloudWatch Logs (Phase 1 - Available Now)

#### `describe_log_groups`
List CloudWatch log groups with optional prefix filtering.

**Example questions:**
- "List all Lambda log groups"
- "Show me ECS log groups"

#### `analyze_log_group`
Analyze CloudWatch logs for anomalies, message patterns, and error patterns.

**Example questions:**
- "Analyze errors in /aws/lambda/my-function from the last hour"
- "Find patterns in my-service logs"

#### `execute_log_insights_query`
Execute CloudWatch Insights queries.

**Example questions:**
- "Search for ERROR in /aws/lambda/my-function from the last hour"
- "Run this Insights query: fields @timestamp, @message | filter @message like /exception/"

#### `get_logs_insight_query_results`
Get results from a previously executed Insights query.

**Example:**
- Used automatically after `execute_log_insights_query` to retrieve results

### Coming Soon (Phase 2)
- Step Functions execution details
- ECS task logs and failures
- Lambda invocation logs

## Configuration

The MCP server uses environment variables for AWS authentication:

- `AWS_PROFILE` - AWS profile name (optional, uses default if not set)
- `AWS_REGION` - AWS region (default: us-east-1)

## Development

See [BOOTSTRAP.md](./BOOTSTRAP.md) for complete development guide.

### Local Development

```bash
# Install dependencies
uv sync

# Run the server
uv run aws-debug-mcp

# Run tests
uv run pytest
```

## Background

### Problem Statement

Debugging distributed AWS applications is painful:
- Manually clicking through CloudWatch console for logs
- Switching between Step Functions, Lambda, and ECS interfaces
- Time-consuming to correlate logs across services
- No way to query AWS debugging info from AI coding assistants

This MCP server brings AWS debugging capabilities directly into Claude Code, eliminating context switching.

### Current State

Manual debugging workflow:
1. Run `search_aws_executions.py` script to find Step Function executions
2. Click AWS console link from script output
3. Manually navigate to CloudWatch for logs
4. Repeat for Lambda/ECS logs
5. Try to correlate events manually

### Desired State

Ask Claude in Claude Code:
- "Show me errors in the fireflies-processing function from last hour"
- "What logs are related to Step Function execution xyz?"
- "Find ECS task failures for my-service"

Claude uses MCP tools to fetch and analyze logs directly.

## Technical Approach

### Architecture
- **Language**: Python 3.10+
- **MCP Framework**: FastMCP
- **AWS SDK**: boto3
- **Package Manager**: uv
- **Installation**: `uvx --from git+https://github.com/you/aws-debug-mcp`

### Modular Design
```
aws-debug-mcp/
├── tools/
│   ├── cloudwatch_logs.py    # Phase 1: MVP
│   ├── step_functions.py     # Phase 2: Future
│   └── ecs.py                 # Phase 2: Future
└── aws/
    ├── client_factory.py      # Environment-based auth
    └── config.py
```

### Key Features
1. **Environment-based auth**: Team members configure via `AWS_PROFILE` env var
2. **Installable like serena**: `uvx --from git+...` pattern
3. **Selective tools**: Only expose tools needed for debugging (no CRUD operations)
4. **Time-based correlation**: Search logs around Step Function execution times

## Scope

### Phase 1: MVP (CloudWatch Logs)
- ✅ `describe_log_groups` - List available log groups
- ✅ `search_logs` - CloudWatch Insights queries
- ✅ `get_logs_for_timerange` - Raw logs in time window

### Phase 2: Expand Services
- ⏳ Step Functions execution details
- ⏳ ECS task logs
- ⏳ Lambda invocation logs

### Phase 3: Polish
- ⏳ Comprehensive tests
- ⏳ Error handling
- ⏳ Community documentation

### Out of Scope (for now)
- AI-powered log analysis
- Multi-cloud support (AWS only)
- Write operations (read-only debugging)

## Success Metrics

1. **Personal productivity**: Reduce debugging time by 50%
2. **Team adoption**: 3+ team members using it within 1 month
3. **Open source**: 10+ GitHub stars, 1+ external contributor
4. **Learning**: Deepen MCP development expertise for Barley work

## Dependencies & Constraints

### Technical Dependencies
- AWS credentials (profile-based or env vars)
- Python 3.10+
- uvx/uv installed

### Organizational Dependencies
- Time to develop (estimate: 2-3 days for MVP)
- Barley team's MCP roadmap (alignment opportunity)

### Constraints
- Must not interfere with Barley team priorities
- Keep scope tight (no feature creep)
- Sustainable maintenance burden

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AWS changes APIs | M | M | Use stable boto3, version constraints |
| Low adoption | M | L | Start with personal use, expand organically |
| Maintenance burden | M | M | Keep scope minimal, good docs |
| Competes with AWS MCPs | H | L | Focus on debugging-specific UX, not generic AWS access |

## Implementation Phases

### Phase 1: Bootstrap & MVP ✅ COMPLETE
- [x] Create bootstrap documentation
- [x] Implement project structure
- [x] Implement CloudWatch Logs tools
- [x] Test locally with Claude Code
- [x] Write README and examples
- [ ] Push to GitHub repo
- [ ] Tag v0.1.0 release

**Success criteria**: Can search CloudWatch logs from Claude Code ✅

### Phase 2: Expand Tools (1-2 days)
- [ ] Add Step Functions tools
- [ ] Add ECS tools
- [ ] Add Lambda tools
- [ ] Improve time correlation logic

**Success criteria**: Can trace request across services

### Phase 3: Open Source Ready (1-2 days)
- [ ] Comprehensive tests (pytest)
- [ ] CI/CD setup (GitHub Actions)
- [ ] CONTRIBUTING.md
- [ ] License (MIT)
- [ ] Example use cases
- [ ] Blog post / LinkedIn article

**Success criteria**: External developers can contribute

## Next Steps

1. **Immediate**: Review BOOTSTRAP.md, validate approach
2. **Create GitHub repo**: `github.com/your-username/aws-debug-mcp`
3. **Initialize project**: Follow bootstrap guide
4. **Implement MVP**: CloudWatch Logs tools
5. **Test locally**: Configure in Claude Code
6. **Share with team**: Get early feedback

## Related Projects

- **Barley MCP Development**: This project teaches MCP patterns applicable to Barley
- **search_aws_executions.py**: Existing Step Functions debugging script (can complement)
- **Serena**: Reference implementation for uvx installation pattern

## Learning Outcomes

By building this, you'll gain:
- Deep MCP server development expertise
- Python packaging/distribution skills (uv, pyproject.toml)
- Open source project management
- FastMCP framework knowledge
- Reusable patterns for Barley's MCP roadmap

---

**Owner**: Evgenii Basmov
**Repository**: (to be created)
**Status**: Ready to bootstrap - see [BOOTSTRAP.md](./BOOTSTRAP.md)