#!/usr/bin/env bash
# Full source rebuild; the output directory includes workload and safety images.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../../tests/capstone-test-env.sh"
[[ $# == 1 ]] || { echo 'usage: build-virtual.sh NEW_OUTPUT' >&2; exit 2; }
export MC_WORK; MC_WORK=$(realpath -m "$1")
[[ ! -e "$MC_WORK" ]] || { echo "use a fresh output directory: $MC_WORK" >&2; exit 2; }
export CAPSTONE_APPLICATION_PROFILE=virtual
bash "$HERE/../deps/build-libevent.sh"
bash "$HERE/build-domain.sh"
bash "$HERE/build-native.sh"
bash "$HERE/build-marker.sh"
bash "$HERE/build-safety.sh"
source "$HERE/../deps/env.sh"
"$CAPSTONE_SDK/capstone-cc" -O1 -I"$CAPSTONE_REPO_ROOT/capstone/runtime/include" \
  "$CAPSTONE_REPO_ROOT/capstone/runtime/virtual/pthread-contract.c" \
  -o "$MC_WORK/pthread.dom"
echo "virtual memcached build: $MC_WORK"
