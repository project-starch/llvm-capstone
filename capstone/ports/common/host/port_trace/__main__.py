"""Run with PYTHONPATH=capstone/ports/common/host python3 -m port_trace."""

import argparse
import json
import sys
from .formats import FORMATS
from .reader import inspect_trace
from .model import TraceError


def main():
    parser = argparse.ArgumentParser(
        description="Inspect allocator traces without changing their bytes."
    )
    parser.add_argument("command", choices=("validate", "summary", "inspect"))
    parser.add_argument("trace")
    parser.add_argument("--format", choices=[f.name for f in FORMATS])
    parser.add_argument(
        "--replay-input",
        action="store_true",
        help="also reject report fields and unresolved process prefixes",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="inspect preview records (0..10000)"
    )
    args = parser.parse_args()
    if not 0 <= args.limit <= 10000:
        parser.error("--limit must be 0..10000")
    try:
        result = inspect_trace(
            args.trace,
            expected_format=args.format,
            replay=args.replay_input,
            limit=args.limit if args.command == "inspect" else 0,
        )
    except (TraceError, OSError) as error:
        print(
            json.dumps({"schema": "capstone.trace-error/v1", "error": str(error)}),
            file=sys.stderr,
        )
        return 1
    if args.command == "validate":
        result = {k: result[k] for k in ("schema", "trace", "validation")}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
