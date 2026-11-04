#!/usr/bin/env bash
# Native blocking-serve regression against exec.c; an optional exec.c selects
# the negative control (00a2ac6bdc15 deadlocks when the raw context exits).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
src=$(realpath "${1:-$here/exec.c}")
out=$(mktemp -d "${TMPDIR:-/tmp}/detach-deadlock.XXXXXX")
trap 'rm -rf "$out"' EXIT
${CC:-gcc} -O1 -g -pthread -I"$here" -I"$here/../include" -DEXEC_C="\"$src\"" \
    "$here/test-detach-deadlock.c" \
    "$here/../linux/"{application-image,image-hash,delegate-service,spawner,signals,park,domain-fault}.c \
    "$here/../common/"{launch,delegate,spawn,msghdr}.c -o "$out/test"
"$out/test"
"$out/test" exit
