#!/usr/bin/env bash
set -euo pipefail

# Temporal-safety overhead microbenchmark -- Capstone side (PI 2026-07-14).
#
# Boots the malloc/touch/free loop under three allocator configs (one boot each)
# and reports the per-op dynamic instruction count + the overhead breakdown:
#   bump     = unprotected baseline (broad heap cap, no per-object caps)
#   norevoke = revoke-on-free allocator, revoke suppressed (alloc-side cost)
#   revoke   = full revoke-on-free (alloc-side + free-time revoke)
# See revoke-cost-probe/revoke_cost_probe.h. This is the Capstone half of the
# QEMU-to-QEMU CHERI-vs-Capstone comparison (plans/perf-cheri-vs-capstone-qemu.md);
# the CHERI half runs on the task-015 CHERI-QEMU stack.
#
# FUNCTIONAL-MODEL PROXY: csrdicount under -icount is a deterministic dynamic
# instruction count, an honest overhead proxy, NOT cycle-accurate timing.
#
# Requires the rootfs.ext2 write lock: announce in agent-handoff/COORDINATION.md.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-revoke-cost-share}
ICOUNT=${ICOUNT:-shift=0,sleep=off}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
bash "$SCRIPT_DIR/build-revoke-cost-probe.sh" "$SHARE_DIR"

run_mode() { # $1=name -> echoes "empty allocfree" from the serial counters
  local name="$1"
  local log="$SHARE_DIR/revoke_cost_$name.log"
  python3 "$SCRIPT_DIR/run-domain-smoke.py" \
    --share-dir "$SHARE_DIR" \
    --log-file "$log" \
    --qemu-extra-arg=-icount \
    --qemu-extra-arg="$ICOUNT" \
    --guest-command "cp /mnt/host/revoke_cost_probe.user /tmp/rc.user && chmod 0755 /tmp/rc.user && /tmp/rc.user /mnt/host/revoke_cost_$name.dom" \
    --success-marker "revoke-cost-probe: measurement complete" >/dev/null 2>&1 \
    || { echo "run-revoke-cost-probe.sh: $name run FAILED; see $log" >&2; return 1; }
}

for m in bump norevoke revoke; do
  echo "== booting mode: $m =="
  run_mode "$m" || exit 1
done

echo
echo "=== revoke-cost: temporal-safety overhead (functional-model proxy) ==="
python3 - "$SHARE_DIR" <<'PY'
import re, sys, os
share = sys.argv[1]
def perop(name):
    log = open(os.path.join(share, f"revoke_cost_{name}.log"), encoding="utf-8", errors="replace").read()
    c = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"counter\[(\d+)\]\s*=\s*(\d+)", log)}
    iters, empty, allocfree = c[0], c[1], c[2]
    return (allocfree - empty) / iters, iters
bump, iters = perop("bump")
norev, _ = perop("norevoke")
rev, _ = perop("revoke")
print(f"iterations (malloc+touch+free) : {iters}")
print(f"bump      (baseline)  per-op   : {bump:.2f} instr")
print(f"norevoke  (alloc-side) per-op  : {norev:.2f} instr")
print(f"revoke    (full)       per-op  : {rev:.2f} instr")
print()
print(f"alloc-side overhead (norevoke-bump) : +{norev-bump:.2f} instr/op  ({norev/bump:.2f}x)")
print(f"revoke overhead     (revoke-norevoke): +{rev-norev:.2f} instr/op")
print(f"total temporal cost (revoke-bump)    : +{rev-bump:.2f} instr/op  ({rev/bump:.2f}x over baseline)")
PY

echo
echo "run-revoke-cost-probe.sh completed. Serial logs: $SHARE_DIR/revoke_cost_*.log"
