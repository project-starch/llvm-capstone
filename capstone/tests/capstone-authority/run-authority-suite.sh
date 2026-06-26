#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around run-authority-suite.py, matching the BEEBS/RV8 run-*.sh
# convention. Builds the loader + domains, boots the Capstone QEMU guest once per
# domain, and classifies each runtime outcome against oracle.tsv.
#
# Env:
#   AUTHORITY_ONLY=name[,name]  run only the listed domains
#   AUTHORITY_NO_BUILD=1        skip the loader/domain rebuild
#
# Prints a PASS/FAIL table and "__CAPSTONE_AUTHORITY_SUITE_PASSED__" on success.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

exec python3 "$SCRIPT_DIR/run-authority-suite.py" "$@"
