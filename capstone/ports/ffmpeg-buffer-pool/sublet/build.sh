#!/usr/bin/env bash
# Sublet is mode 2 of the shared Capstone domain binary.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$HERE/../capstone/build.sh" "$@"
