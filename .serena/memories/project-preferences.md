# Project Preferences

## Testing Philosophy
- **No unit tests** - the project owner considers them useless for this project
- **Verification via real tool calls** - test tools by actually calling them against real services
- **Manual integration testing** - use the MCP server locally and verify tool outputs

## Test Command (`.claude/commands/test.md`)
- Assumes all credentials are correctly configured
- If tools don't work or aren't exposed, it indicates a credentials setup issue
- Used during development to verify tools are functioning
