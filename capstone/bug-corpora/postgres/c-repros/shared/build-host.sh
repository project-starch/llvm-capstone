#!/bin/bash
# Build the client-repros cases on the host under ASan. This is not one of the
# three arms: it is the check that a reduction REPRODUCES the defect at all,
# before anything is claimed about what a mechanism does or does not see.
# A case that ASan does not fault on is not yet a reproduction.
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
S="$R/shared"
OUT=${OUT:-/tmp/pgclient-host}
mkdir -p "$OUT"
CC=${CC:-cc}
FLAGS="-O0 -g -fsanitize=address -fno-omit-frame-pointer -I$S"

built=0; failed=0
for d in "$R"/[0-9][0-9]_*/; do
  [ -f "$d/case.c" ] || continue
  tag=$(basename "$d")
  n=$(printf '%s' "$tag" | cut -c1-2 | sed 's/^0//')
  if $CC $FLAGS "$d/case.c" "$S/driver.c" "$S"/upstream_*.c -o "$OUT/$tag" 2>"$OUT/$tag.err"; then
    printf '  %-52s BUILD OK\n' "$tag"; built=$((built+1))
  else
    printf '  %-52s BUILD FAIL\n' "$tag"; head -5 "$OUT/$tag.err" | sed 's/^/      /'; failed=$((failed+1))
  fi
done
echo "built=$built failed=$failed  (binaries in $OUT)"
