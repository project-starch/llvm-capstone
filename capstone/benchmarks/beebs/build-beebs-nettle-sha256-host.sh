#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=nettle-sha256
source "$SCRIPT_DIR/build-beebs-simple-host-common.sh"
