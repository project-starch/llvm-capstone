#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
peer=${1:?native context source tree or snapshot}
out=${2:?output directory}
mkdir -p "$out"
ref=$peer/capstone/runtime/virtual/tests/native-context/reference.c
inc=$peer/capstone/runtime/include
sha256sum "$ref" "$inc/capstone/native-context.h" > "$out/peer.sha256"
for kind in normal mutant; do
  flags=(); [[ $kind != mutant ]] || flags+=(-DCV_BRIDGE_CAUSE_ONLY)
  cc -std=c11 -O2 -Wall -Wextra -Werror -I"$inc" "${flags[@]}" \
    "$here/context-fault-probe.c" "$ref" -o "$out/$kind"
  rc=0
  (ulimit -c 0; "$out/$kind") > "$out/$kind.log" 2>&1 || rc=$?
  if [[ $kind == normal ]]; then [[ $rc == 0 ]]; else [[ $rc == 134 ]]; fi
  echo "context-fault-peer $kind rc=$rc"
done
