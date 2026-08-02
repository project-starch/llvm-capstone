#!/usr/bin/env bash
# Compile the xlang corpus shims as CHERI-RISC-V *purecap* binaries for the
# CHERI baseline column (plan: agent-handoff/plans/cheri-baseline-xlang.md).
#
# Output: $OUT holds one ELF per row named `row<N>` plus a sanity probe. Bake
# $OUT into the CheriBSD disk image (--disk-image/extra-files) or 9p-share it,
# then run under CHERI-QEMU with run-in-guest.sh.
#
# This is the *baseline* column (CHERI), not our system. Measurement only.
#
# PREREQUISITE: check_shim_fidelity.py must pass first. A shim that does not
# reproduce its row's ASan class natively measures nothing, so its purecap
# verdict would be noise.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Workspace layout: cheribuild conventions, overridable per machine.
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
CHERI_SDK=${CHERI_SDK:-$CHERI_ROOT/output/sdk}
SYSROOT=${SYSROOT:-$CHERI_SDK/sysroot-riscv64-purecap}
CC=${CC:-$CHERI_SDK/bin/clang}
OUT=${OUT:-$CHERI_ROOT/rootfs-overlay/root/cheri-baseline-xlang}

for p in "$CC" "$SYSROOT"; do
  [ -e "$p" ] || { echo "MISSING: $p (set CHERI_ROOT/CHERI_SDK/SYSROOT)" >&2; exit 2; }
done
mkdir -p "$OUT"

# --- probe the purecap target triple/arch for this SDK -------------------------
BASEFLAGS="--sysroot=$SYSROOT -mno-relax"
PURECAP=""
probe=$(mktemp /tmp/cheri-xlang-probe-XXXX.c); echo 'int main(void){return 0;}' > "$probe"
for spec in \
  "--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64imafdcxcheri -mabi=l64pc128d" \
  "--target=riscv64-unknown-freebsd -march=rv64gc_xcheri -mabi=l64pc128d"; do
  if "$CC" $spec $BASEFLAGS "$probe" -o "$probe.out" 2>/dev/null; then PURECAP="$spec"; break; fi
done
rm -f "$probe" "$probe.out"
[ -n "$PURECAP" ] || { echo "could not find a working purecap -march for $CC" >&2; exit 3; }
CFLAGS="$PURECAP $BASEFLAGS"

# -O0 on purpose, exactly as the sqlite baseline: a use-after-free is undefined
# behaviour and at -O1+ the compiler may hoist the load before the free or elide
# the dangling access, so the access we want CHERI to police is never emitted.
OPT=${OPT:--O0}
echo "[*] purecap flags: $CFLAGS ; measured binaries at $OPT"

# Derived from rows.tsv, NOT hardcoded. It was a hardcoded list until
# 2026-08-02, and row 7 was added to rows.tsv and to the fidelity gate but not
# here -- so its shim was never compiled, the guest reported BIN-MISSING, and
# the row classified as NO-DATA. A silent miss, not an error. rows.tsv is the
# single list; keep it that way.
DEFECTS=$(awk -F'\t' '!/^#/ && NF >= 2 { sub(/^shims\//, "", $2); sub(/\.c$/, "", $2); print $2 }' "$SCRIPT_DIR/rows.tsv")
MOCK="$SCRIPT_DIR/mock-mruby/mock_mruby.c"

fail=0
for n in $DEFECTS; do
  src="$SCRIPT_DIR/shims/$n.c"
  if "$CC" $CFLAGS $OPT -g -o "$OUT/$n" "$src" "$MOCK" 2>"$OUT/$n.build.log"; then
    echo "  [ok]   $n"
  else
    echo "  [FAIL] $n (see $OUT/$n.build.log)"; fail=$((fail+1))
  fi
done

# Sanity probe: the harness itself must run clean purecap, or every row's
# verdict is about the harness rather than the defect (the sanity_mock role in
# the sqlite baseline).
cat >"$OUT/sanity_mock.c" <<'EOF'
#include "mock_mruby.h"
int main(void) {
  mrb_state *mrb = mrb_open(1024);
  mrb_value *regs = mrb->c->stack;
  regs[0] = 1;
  mrb_stack_extend(mrb, 4096);          /* a legal, in-bounds move */
  mrb->c->stack[0] = 2;                 /* through the NEW pointer */
  void *p = mrb_gc_alloc(mrb, 48);
  (void)p; mrb_full_gc(mrb);            /* alloc + sweep, no stale use */
  mock_report("sanity_mock", "clean");
  mrb_close(mrb);
  return 0;
}
EOF
if "$CC" $CFLAGS $OPT -g -I"$SCRIPT_DIR/mock-mruby" \
     -o "$OUT/sanity_mock" "$OUT/sanity_mock.c" "$MOCK" 2>"$OUT/sanity_mock.build.log"; then
  echo "  [ok]   sanity_mock"
else
  echo "  [FAIL] sanity_mock (see $OUT/sanity_mock.build.log)"; fail=$((fail+1))
fi
rm -f "$OUT/sanity_mock.c"

cp "$SCRIPT_DIR/run-in-guest.sh" "$OUT/" 2>/dev/null || true
echo "[*] output: $OUT (failures: $fail)"
[ "$fail" -eq 0 ]
