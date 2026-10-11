#!/usr/bin/env bash
# Prepare, configure and survey CPython 3.13.7 against the Capstone pure-cap compiler.
# Exits 1 on a regression, 2 on a flipped control, a harness error or no data.
# CPY_PATCHES=none surveys upstream as released (the controls then report ERROR).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null

BUILD_DIR=$(bash "$SCRIPT_DIR/prepare-cpython-capstone.sh" | tail -1)
source "$BUILD_DIR/capstone-env.sh"
exec python3 "$SCRIPT_DIR/survey-cpython-capstone.py" "$BUILD_DIR" "$@"
