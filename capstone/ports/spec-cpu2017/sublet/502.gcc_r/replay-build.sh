#!/bin/bash
# Rebuild a copy of the 502.gcc_r build directory with the commands of the build it was copied from.
# usage: replay-build.sh TREE [JOBS]
#   TREE  a copy of the benchmark's build directory, with runcpu's make.out in it
# Every compile command and the link command of make.out run again in TREE. Compiler, flags and
# SDK are therefore those of the original build. An unchanged copy must give the original
# executable byte for byte, which is the check that the replay is complete.
set -euo pipefail

[ $# -ge 1 ] || { echo "usage: replay-build.sh TREE [JOBS]" >&2; exit 2; }
TREE=$(cd "$1" && pwd)
JOBS=${2:-16}
cd "$TREE"
[ -f make.out ] || { echo "replay-build: $TREE has no make.out" >&2; exit 2; }

grep -E '^[^ ]+ -c -o [^ ]+\.o ' make.out > replay-compile.cmds || true
grep -E ' -o cpugcc_r *$' make.out | grep -v -- ' -c ' > replay-link.cmd || true
[ -s replay-compile.cmds ] || { echo "replay-build: make.out has no compile command" >&2; exit 1; }
[ "$(wc -l < replay-link.cmd)" -eq 1 ] || { echo "replay-build: make.out does not have exactly one link command" >&2; exit 1; }

find . -name '*.o' -delete
rm -f cpugcc_r

if ! xargs -d '\n' -P "$JOBS" -I COMMAND sh -c 'COMMAND' < replay-compile.cmds > replay-compile.log 2>&1; then
  echo "replay-build: a compile command failed:" >&2
  grep -E 'error' replay-compile.log | head -20 >&2
  exit 1
fi
if ! sh replay-link.cmd > replay-link.log 2>&1; then
  echo "replay-build: the link command failed:" >&2
  tail -20 replay-link.log >&2
  exit 1
fi

echo "compiled $(wc -l < replay-compile.cmds) files"
sha256sum cpugcc_r
