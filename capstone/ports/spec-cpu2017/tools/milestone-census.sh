#!/usr/bin/env bash
# Run on the measurement host. Configuration and artifact layout: profile.md.
set -euo pipefail
exec python3 "$(dirname "$0")/milestone-census.py" "$@"
