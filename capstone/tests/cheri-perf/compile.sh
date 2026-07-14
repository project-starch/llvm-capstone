#!/usr/bin/env bash
# Compile the CHERI-side revoke-cost microbenchmark as a CheriBSD *purecap*
# binary into the rootfs overlay, alongside run-in-guest.sh. Bake the overlay
# into the disk image (run.sh does this) and boot under CHERI-QEMU.
#
# Two variants are built -- rc_instret (rdinstret) and rc_cycle (rdcycle) -- so
# the guest can fall back if one CSR is not user-readable. See revoke_cost_cheri.c.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHERI_SDK=${CHERI_SDK:-/home/alexey/cheri/output/sdk}
SYSROOT=${SYSROOT:-$CHERI_SDK/sysroot-riscv64-purecap}
CC="$CHERI_SDK/bin/clang"
OUT=${OUT:-/home/alexey/cheri-ws/rootfs-overlay/root/cheri-perf}

for p in "$CC" "$SYSROOT"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 2; }
done
mkdir -p "$OUT"

# Probe the working purecap triple/arch for this SDK (same logic as the baseline).
BASEFLAGS="--sysroot=$SYSROOT -mno-relax"
PURECAP=""
probe=$(mktemp /tmp/cheri-perf-probe-XXXX.c); echo 'int main(void){return 0;}' > "$probe"
for spec in \
  "--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64imafdcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64gc_xcheri -mabi=l64pc128d"; do
  if "$CC" $spec $BASEFLAGS "$probe" -o "$probe.out" 2>/dev/null; then PURECAP="$spec"; break; fi
done
rm -f "$probe" "$probe.out"
[ -n "$PURECAP" ] || { echo "no working purecap -march for $CC" >&2; exit 3; }
CFLAGS="$PURECAP $BASEFLAGS"
# -O2: we WANT a realistic optimized allocator loop here (unlike the cheri-baseline
# shims, which are -O0 to preserve UB accesses). The malloc/free calls are real
# library calls the optimizer cannot elide.
OPT=${OPT:--O2}
echo "[*] purecap flags: $CFLAGS ; $OPT"

"$CC" $CFLAGS $OPT -g "$SCRIPT_DIR/revoke_cost_cheri.c" -o "$OUT/rc_instret" \
  2>"$OUT/rc_instret_cc.log" || { echo "rc_instret build FAILED"; cat "$OUT/rc_instret_cc.log" >&2; exit 4; }
"$CC" $CFLAGS $OPT -g -DUSE_CYCLE "$SCRIPT_DIR/revoke_cost_cheri.c" -o "$OUT/rc_cycle" \
  2>"$OUT/rc_cycle_cc.log" || { echo "rc_cycle build FAILED"; cat "$OUT/rc_cycle_cc.log" >&2; exit 4; }

# Real-workload arm: the shared binary-search-tree build/lookup/destroy workload.
# TW_ROUNDS=10 (20k frees) bounds eager's cost (each free is a ~14M-instr sweep)
# while keeping a 2000-node live-set for the sweep to scan.
TREE_DEFS=${TREE_DEFS:--DTW_KEYS=2000 -DTW_ROUNDS=10}
"$CC" $CFLAGS $OPT -g $TREE_DEFS "$SCRIPT_DIR/tree_cheri.c" -o "$OUT/rc_tree" \
  2>"$OUT/rc_tree_cc.log" || { echo "rc_tree build FAILED"; cat "$OUT/rc_tree_cc.log" >&2; exit 4; }
"$CC" $CFLAGS $OPT -g $TREE_DEFS -DUSE_CYCLE "$SCRIPT_DIR/tree_cheri.c" -o "$OUT/rc_tree_cycle" \
  2>"$OUT/rc_tree_cycle_cc.log" || { echo "rc_tree_cycle build FAILED"; cat "$OUT/rc_tree_cycle_cc.log" >&2; exit 4; }

# Revocation-status probe (reused from the cheri-baseline): confirms the config
# the sysctl set is actually what the process runtime enforces.
STATUS_SRC="$SCRIPT_DIR/../cheri-baseline/cheri_status.c"
if [ -f "$STATUS_SRC" ]; then
  "$CC" $CFLAGS -O1 "$STATUS_SRC" -o "$OUT/cheri_status" \
    2>"$OUT/cheri_status_cc.log" && echo "  cheri_status: built" \
    || echo "  cheri_status: BUILD-FAIL (see cheri_status_cc.log)" >&2
fi

cp "$SCRIPT_DIR/run-in-guest.sh" "$OUT/run-in-guest.sh"
chmod +x "$OUT/run-in-guest.sh"
echo "[*] built rc_instret + rc_cycle -> $OUT"
