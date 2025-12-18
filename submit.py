#!/usr/bin/env python3
"""
Needle in Haystack Assignment Submission Tool

This is the command-line entry point for submitting your agent.
The core logic is in submit_core module (which can be compiled to .so).

Usage:
    python submit.py --agent agent_template:ExampleAgent
    python submit.py --agent my_agent:MyAgent --api-key sk-xxx --base-url https://api.openai.com/v1

Required environment variables:
    STUDENT_ID, STUDENT_NAME, STUDENT_NICKNAME, MAIN_CONTRIBUTOR, API_KEY, BASE_URL
    (API_KEY and BASE_URL can also be passed as command arguments)
"""

import sys
import argparse
from typing import Optional

from submit_core import run_submission



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Submit your Needle in Haystack agent for evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires API_KEY and BASE_URL in environment)
  python submit.py --agent agent_template:ExampleAgent

  # With API credentials as arguments
  python submit.py --agent my_agent:MyAgent --api-key sk-xxx --base-url https://api.openai.com/v1

  # Short form
  python submit.py -a agent_template:ExampleAgent

Environment Variables Required:
  STUDENT_ID         Your student ID
  STUDENT_NAME       Your full name
  STUDENT_NICKNAME   Your nickname for the leaderboard
  MAIN_CONTRIBUTOR   'human' or 'ai' (who did most of the work)
  API_KEY            API key for LLM (optional if passed as argument)
  BASE_URL           API base URL (optional if passed as argument)
        """
    )

    parser.add_argument(
        '-a', '--agent',
        type=str,
        required=True,
        help='Agent specification in format "module.path:ClassName" (e.g., agent_template:ExampleAgent)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM service (overrides API_KEY environment variable)'
    )

    parser.add_argument(
        '--base-url',
        type=str,
        default=None,
        help='API base URL (overrides BASE_URL environment variable)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    try:
        args = parse_arguments()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n❌ Error parsing arguments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("Needle in Haystack Assignment - Submission Tool")
    print("=" * 70)
    print(f"\nAgent: {args.agent}")

    if args.api_key:
        print("API Key: Provided via command line")
    if args.base_url:
        print(f"Base URL: {args.base_url}")

    print("\nStarting submission process...\n")

    try:
        exit_code = run_submission(
            agent_spec=args.agent,
            api_key=args.api_key,
            base_url=args.base_url
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()