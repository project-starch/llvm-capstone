#!/bin/sh
# test-load-bounds.sh [EXEC_C]: build test-load-bounds.c against EXEC_C (default exec.c) and run it.
# Host test (x86 or riscv, any Linux with gcc). Exit 0 only if every case is refused cleanly.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-$here/exec.c}
out=${TMPDIR:-/tmp}/test-load-bounds.$$
trap 'rm -f "$out"' EXIT
${CC:-gcc} -O1 -g -w -I"$here" -I"$here/../include" -I"$here/.." -DEXEC_C="\"$src\"" \
    "$here/test-load-bounds.c" -o "$out" -pthread -Wl,--unresolved-symbols=ignore-all
"$out"
