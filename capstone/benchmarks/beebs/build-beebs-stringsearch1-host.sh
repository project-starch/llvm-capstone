#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BEEBS_BENCHMARK=stringsearch1
source "$SCRIPT_DIR/build-beebs-simple-host-common.sh"
