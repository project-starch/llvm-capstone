#!/usr/bin/env bash
set -euo pipefail

# Additive optimization sweep for authority probes. The canonical suite remains
# -O0; this wrapper reports whether eligible probes preserve their oracle class
# at -O1, -O2, and -O3.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

exec python3 "$SCRIPT_DIR/run-authority-opt-matrix.py" "$@"
