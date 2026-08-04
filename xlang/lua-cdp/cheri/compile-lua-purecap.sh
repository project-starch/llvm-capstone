#!/usr/bin/env bash
# Compile the Lua-CDP shims as CHERI-RISC-V *purecap* binaries for the CHERI
# baseline column. This is the CHERI half of the fair comparison; the shims are
# the SAME source the Capstone column compiles (../shims/*.c).
#
# Output: $OUT holds one ELF per row (named after the shim stem) plus cheri_status.
# Bake $OUT into the image with provision-cheri-vehicle.sh (OVERLAY_SRC) and run
# under CHERI-QEMU with run-in-guest.sh.
#
# -O0 on purpose, exactly as the mruby/sqlite baselines: a use-after-free is
# undefined behaviour and at -O1+ the compiler may hoist the load before the free
# or elide the dangling access, so the access CHERI must police is never emitted.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
ROOTFS=${ROOTFS:-$CHERI_ROOT/rootfs-purecap}
CC=${CC:-$SDK/bin/clang}
OUT=${OUT:-$CHERI_ROOT/lua-cdp-cheri}
ROWS=${ROWS_FILE:-$HERE/rows.tsv}
BASELINE=${BASELINE:-$HERE/../../../capstone/tests/cheri-baseline}

for p in "$CC" "$ROOTFS/usr/include" "$ROWS"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 2; }
done
mkdir -p "$OUT"

# The recipe verified on this SDK: --sysroot is honoured for the LINK (libc/crt);
# the shims need no sysroot headers (only <stdint.h>, a clang builtin), and
# mock_report.c forward-declares printf, so the host-header search anomaly on
# this SDK does not matter. Compile source+source in ONE invocation.
CFLAGS="--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d --sysroot=$ROOTFS -mno-relax"
OPT=${OPT:--O0}
echo "[*] purecap flags: $CFLAGS ; $OPT"

fail=0
# Column 2 basenames from rows.tsv are the single source of truth.
NAMES=$(awk -F'\t' '!/^#/ && NF>=2 { sub(/^shims\//,"",$2); sub(/\.c$/,"",$2); print $2 }' "$ROWS")
for n in $NAMES; do
  src="$HERE/../shims/$n.c"
  [ -f "$src" ] || { echo "  [FAIL] $n (no source $src)"; fail=$((fail+1)); continue; }
  if "$CC" $CFLAGS $OPT -g -o "$OUT/$n" "$src" "$HERE/mock_report.c" 2>"$OUT/$n.build.log"; then
    echo "  [ok]   $n"
  else
    echo "  [FAIL] $n (see $OUT/$n.build.log)"; fail=$((fail+1))
  fi
done

# cheri_status: reports the revocation policy each config ACTUALLY applied, so
# config reality is recorded rather than assumed.
if [ -f "$BASELINE/cheri_status.c" ]; then
  "$CC" $CFLAGS -o "$OUT/cheri_status" "$BASELINE/cheri_status.c" 2>/dev/null \
    || echo "  [warn] cheri_status did not build; config reality will be unrecorded"
fi

cp "$HERE/run-in-guest.sh" "$OUT/"
cp "$ROWS" "$OUT/rows.tsv"
rm -f "$OUT"/*.build.log
echo "[*] output: $OUT (failures: $fail)"
[ "$fail" -eq 0 ]
