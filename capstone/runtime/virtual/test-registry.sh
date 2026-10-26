#!/bin/sh
# test-registry.sh [EXEC_C]: build test-registry.c against EXEC_C (default exec.c) and run it.
# Host test (x86 or riscv, any Linux with gcc). Exit 0 only if every check passes.
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
src=${1:-$here/exec.c}
case $src in /*) ;; *) src=$(pwd)/$src ;; esac
out=${TMPDIR:-/tmp}/test-registry.$$
trap 'rm -f "$out"' EXIT
trap 'exit 1' INT TERM
${CC:-gcc} -O1 -g -w -I"$here" -I"$here/../include" -I"$here/.." -DEXEC_C="\"$src\"" \
    "$here/test-registry.c" -o "$out" -pthread -Wl,--unresolved-symbols=ignore-all
"$out"
