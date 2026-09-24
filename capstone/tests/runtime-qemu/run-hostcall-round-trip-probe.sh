#!/usr/bin/env bash
# Pointers computed through uintptr_t in a musl domain (CapstoneRecoverProvenance), with the pass-off control. Lives in round-trip/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/round-trip/run.sh" "$@"
