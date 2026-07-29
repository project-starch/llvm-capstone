#!/usr/bin/env bash
# Re-snapshot the uncommitted submodule edits. See README.md for why these exist.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT="$ROOT/capstone/tests/vendor-patches"

snap() {  # snap <label> <repo-relative dir> <output file>
  local label=$1 dir=$2 out=$3
  if [[ ! -d "$ROOT/$dir" ]]; then echo "  SKIP $label (missing $dir)"; return; fi
  git -C "$ROOT/$dir" diff > "$OUT/$out.new" 2>/dev/null || true
  if [[ -f "$OUT/$out" ]] && cmp -s "$OUT/$out" "$OUT/$out.new"; then
    rm -f "$OUT/$out.new"; echo "  same $label"
  else
    mv "$OUT/$out.new" "$OUT/$out"; echo "  UPDATED $label ($(wc -c < "$OUT/$out") B)"
  fi
}

snap "capstone-qemu"        capstone/capstone-qemu                                     capstone-qemu.patch
snap "opensbi component"    capstone/caplifive-buildroot/components/opensbi            opensbi-component.patch
snap "opensbi capstone-sbi" capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi opensbi-capstone-sbi.patch
snap "capstone-sbi package" capstone/caplifive-buildroot/package/capstone-sbi-domain/capstone-sbi buildroot-capstone-sbi-package.patch

# caplifive-SYSTEM: the tree the FPGA firmware is actually built from. It was missing
# here, which meant the FPGA monitor's uncommitted edits had NO backup at all -- including
# the 2026-07-29 C-13 root-cause fix (scalar blob copy). Submodule source is deliberately
# not committed, so an unmirrored edit is one `git checkout` from being lost. Same
# wrong-tree gap that left the FPGA rootfs overlay carrying a day-old SQLite domain.
snap "SYSTEM opensbi"       capstone/caplifive-system/sw/buildroot/components/opensbi   system-opensbi-component.patch
snap "SYSTEM capstone-sbi"  capstone/caplifive-system/sw/buildroot/components/opensbi/lib/sbi/capstone-sbi system-opensbi-capstone-sbi.patch

DIAG="$ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/capstone-diag.c"
if [[ -f "$DIAG" ]]; then
  if cmp -s "$DIAG" "$OUT/capstone-diag.c"; then echo "  same capstone-diag.c"
  else cp "$DIAG" "$OUT/capstone-diag.c"; echo "  UPDATED capstone-diag.c"; fi
else
  echo "  SKIP capstone-diag.c (missing -- restore it FROM here: cp $OUT/capstone-diag.c $DIAG)"
fi
echo "done."
