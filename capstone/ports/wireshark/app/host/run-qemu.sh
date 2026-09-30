#!/usr/bin/env bash
# The VM is managed by capstone-vm; CAPSTONE_VM_STATE selects a running instance.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$HERE/run.py" "$@"
