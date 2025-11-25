"""Entry point for aws-debug-mcp."""
from .server import mcp


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
