"""Entry point for debug-mcp."""

import argparse


def main():
    """Run the MCP server."""
    import sys

    parser = argparse.ArgumentParser(description="Debug MCP Server")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--aws-profile", default="", help="AWS profile (default: empty)")

    args = parser.parse_args()

    # DEBUG LOGGING
    print(f"[DEBUG] sys.argv: {sys.argv}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Parsed region: {args.aws_region}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Parsed profile: {args.aws_profile}", file=sys.stderr, flush=True)

    # Initialize proxy with credentials from CLI args BEFORE importing server
    from .mcp_proxy import init_proxy

    init_proxy(aws_profile=args.aws_profile, aws_region=args.aws_region)

    # Import server AFTER proxy is initialized
    from .server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
