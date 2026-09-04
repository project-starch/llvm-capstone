#!/usr/bin/env bash
# Re-snapshot the uncommitted submodule edits. See README.md for why these exist.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT="$ROOT/capstone/tests/vendor-patches"

# snap <label> <repo-relative dir> <output file> [base] [path...]
#
# Without <base> this mirrors the WORKING TREE only (git diff). That was right when
# submodule source was deliberately left uncommitted; since 2026-08-05 the rule is the
# opposite -- submodule work SHOULD be committed -- and a committed edit is invisible to
# a bare `git diff`. So entries whose repo can be diffed against its remote pass a <base>
# and thereby capture committed-but-unpushed work as well. capstone-qemu deliberately
# does not: its origin/master is upstream QEMU and the fork is ~1M lines ahead, so a
# remote-based diff there is useless rather than merely large.
snap() {
  local label=$1 dir=$2 out=$3 base=${4:-}; shift 4 2>/dev/null || shift $#
  if [[ ! -d "$ROOT/$dir" ]]; then echo "  SKIP $label (missing $dir)"; return; fi
  if [[ -n "$base" ]] && ! git -C "$ROOT/$dir" rev-parse --verify -q "$base" >/dev/null; then
    echo "  SKIP $label (base $base not resolvable -- fetch first)"; return
  fi
  git -C "$ROOT/$dir" diff ${base:+"$base"} ${1:+-- "$@"} > "$OUT/$out.new" 2>/dev/null || true
  # A mirror is a BACKUP. Replacing a non-empty one with an empty diff is how this tool
  # would erase what it exists to protect: every mirrored tree was clean on 2026-08-19,
  # so a run on that day would have blanked all of them at once and reported UPDATED for
  # each. An empty result now means "nothing to mirror", never "delete the mirror".
  if [[ ! -s "$OUT/$out.new" && -s "${OUT}/$out" ]]; then
    rm -f "$OUT/$out.new"
    echo "  KEPT $label -- diff is now EMPTY but the mirror is not; refusing to blank it."
    echo "       (the edit was probably committed; give this entry a <base> so committed"
    echo "        work is captured, or delete the mirror deliberately if it is obsolete)"
    return
  fi
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

# The domain-sizing series. All three repos are committed but none of them accepts a
# push from this account, so these mirrors are the only way the change travels with the
# parent. They pass a base so COMMITTED work is captured, not just the working tree.
snap "domain sizing (module)" capstone/caplifive-buildroot caplifive-buildroot-domain-sizing.patch origin/capstone-bootstrap package/modcapstone Makefile configs
snap "domain sizing (monitor)" capstone/caplifive-buildroot/components/opensbi/lib/sbi/capstone-sbi capstone-sbi-two-regions.patch origin/master sbi_capstone.c

DIAG="$ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/capstone-diag.c"
if [[ -f "$DIAG" ]]; then
  if cmp -s "$DIAG" "$OUT/capstone-diag.c"; then echo "  same capstone-diag.c"
  else cp "$DIAG" "$OUT/capstone-diag.c"; echo "  UPDATED capstone-diag.c"; fi
else
  echo "  SKIP capstone-diag.c (missing -- restore it FROM here: cp $OUT/capstone-diag.c $DIAG)"
fi
echo "done."
