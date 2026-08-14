#!/usr/bin/env bash
# Fetch, prepare and survey musl against the Capstone pure-cap compiler.
# Exits non-zero on a regression, on a flipped positive control, or on no data.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../tests/capstone-test-env.sh"

MUSL_SRC_DIR=$(bash "$SCRIPT_DIR/prepare-musl-capstone.sh" | tail -1)
exec python3 "$SCRIPT_DIR/survey-musl-capstone.py" "$MUSL_SRC_DIR" "$@"
