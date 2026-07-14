#!/usr/bin/env bash
set -euo pipefail

# Real-workload (binary-search-tree) temporal-safety overhead -- Capstone side.
# Boots the BST build/lookup/destroy workload under three allocator configs (one
# boot each) and reports per-node-lifecycle instruction count + the overhead
# breakdown. Counterpart to tests/cheri-perf/ (tree arm). See revoke_cost_tree.c.
#
#   bump     = unprotected baseline (broad heap cap, no per-object caps)
#   norevoke = revoke-on-free allocator, revoke suppressed (alloc-side cost)
#   revoke   = full revoke-on-free (alloc-side + free-time revoke)
#
# FUNCTIONAL-MODEL PROXY: rdcycle under -icount is a deterministic dynamic
# instruction count, an honest overhead proxy, NOT cycle-accurate timing.
#
# Requires the rootfs.ext2 write lock: announce in agent-handoff/COORDINATION.md.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-tree-cost-share}
ICOUNT=${ICOUNT:-shift=0,sleep=off}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
bash "$SCRIPT_DIR/build-tree-cost-probe.sh" "$SHARE_DIR"

run_mode() { # $1=name
  local name="$1"
  local log="$SHARE_DIR/tree_cost_$name.log"
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
    --share-dir "$SHARE_DIR" \
    --log-file "$log" \
    --qemu-extra-arg=-icount \
    --qemu-extra-arg="$ICOUNT" \
    --guest-command "cp /mnt/host/revoke_cost_probe.user /tmp/tc.user && chmod 0755 /tmp/tc.user && /tmp/tc.user /mnt/host/tree_cost_$name.dom" \
    --success-marker "revoke-cost-probe: measurement complete" >/dev/null 2>&1 \
    || { echo "run-tree-cost-probe.sh: $name run FAILED; see $log" >&2; return 1; }
}

for m in bump norevoke revoke; do
  echo "== booting mode: $m =="
  run_mode "$m" || exit 1
done

echo
echo "=== tree-cost: real-workload temporal-safety overhead (functional-model proxy) ==="
python3 - "$SHARE_DIR" <<'PY'
import re, sys, os
share = sys.argv[1]
def perop(name):
    log = open(os.path.join(share, f"tree_cost_{name}.log"), encoding="utf-8", errors="replace").read()
    c = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"counter\[(\d+)\]\s*=\s*(\d+)", log)}
    ops, _, instr = c[0], c[1], c[2]
    return instr / ops, ops
bump, ops = perop("bump")
norev, _ = perop("norevoke")
rev, _ = perop("revoke")
print(f"node lifecycles (build+lookup+destroy) : {ops}")
print(f"bump      (baseline)  per-op   : {bump:.2f} instr")
print(f"norevoke  (alloc-side) per-op  : {norev:.2f} instr")
print(f"revoke    (full)       per-op  : {rev:.2f} instr")
print()
print(f"alloc-side overhead (norevoke-bump) : +{norev-bump:.2f} instr/op  ({norev/bump:.2f}x)")
print(f"revoke overhead     (revoke-norevoke): +{rev-norev:.2f} instr/op")
print(f"total temporal cost (revoke-bump)    : +{rev-bump:.2f} instr/op  ({rev/bump:.2f}x over baseline)")
PY

echo
echo "run-tree-cost-probe.sh completed. Serial logs: $SHARE_DIR/tree_cost_*.log"
