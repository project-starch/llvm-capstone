#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRACE=${1:?usage: run-qemu.sh TRACE RESULT-DIR [EXPECTED-STATUS]}
RESULT=${2:?missing result directory}
exec bash "$HERE/../capstone/run-qemu.sh" "$TRACE" "$RESULT" sublet "${3:-0}"
