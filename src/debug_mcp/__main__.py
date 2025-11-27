"""Entry point for debug-mcp."""

import argparse

from .mcp_proxy import init_proxy
from .server import mcp


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="Debug MCP Server")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--aws-profile", default="", help="AWS profile (default: empty)")

    args = parser.parse_args()

    # Initialize proxy with credentials from CLI args only
    init_proxy(aws_profile=args.aws_profile, aws_region=args.aws_region)

    mcp.run()


if __name__ == "__main__":
    main()
