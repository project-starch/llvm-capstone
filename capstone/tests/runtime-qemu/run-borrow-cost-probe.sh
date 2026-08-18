#!/usr/bin/env bash
set -euo pipefail

# Borrow-path cost measurement (task-014, paper deliverable 2).
#
# ONE domain receives a real monitor-granted LINEAR arena and measures the
# dynamic instruction count of three variants of the same borrow-one-word
# boundary operation -- raw pointer, capability borrow (mrev+delin+access+
# revoke), and a TRANSIENT-style defensive copy -- via the csrdicount emulator
# readout under -icount. See borrow-cost-probe/borrow_cost_probe.h and RESULTS.md.
#
# FUNCTIONAL-MODEL PROXY: csrdicount is a deterministic dynamic-instruction
# count, an honest overhead proxy, NOT cycle-accurate silicon timing (QEMU has
# no pipeline/cache/cycle model).
#
# Requires the rootfs.ext2 write lock: the suites must be SERIALIZED (never two at once)
# and confirm the other agent is not mid-run before invoking this. (The boot
# uses -snapshot, but the lock convention still applies.)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-runtime-qemu-share}
LOG_FILE=${LOG_FILE:-$TMP_ROOT/capstone-runtime-qemu-borrow-cost-probe.log}
# shift=0: 1 icount tick == 1 retired instruction, so csrdicount deltas are an
# exact dynamic instruction count. sleep=off keeps the vCPU from idling the
# virtual clock while we are not measuring.
ICOUNT=${ICOUNT:-shift=0,sleep=off}

mkdir -p "$TMP_ROOT" "$SHARE_DIR"
rm -f "$SHARE_DIR"/borrow_cost.dom "$SHARE_DIR"/borrow_cost_probe.user

bash "$SCRIPT_DIR/build-borrow-cost-probe.sh" "$SHARE_DIR"

python3 "$SCRIPT_DIR/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --qemu-extra-arg=-icount \
  --qemu-extra-arg="$ICOUNT" \
  --guest-command "cp /mnt/host/borrow_cost_probe.user /tmp/bc.user && chmod 0755 /tmp/bc.user && /tmp/bc.user /mnt/host/borrow_cost.dom" \
  --success-marker "borrow-cost-probe: measurement complete" \
  || { echo "run-borrow-cost-probe.sh: domain run FAILED; see $LOG_FILE" >&2; exit 1; }

echo
echo "=== borrow-cost-probe: instruction-count results (functional-model proxy) ==="
python3 - "$LOG_FILE" <<'PY'
import re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
c = {}
for m in re.finditer(r"counter\[(\d+)\]\s*=\s*(\d+)", log):
    c[int(m.group(1))] = int(m.group(2))
need = [0, 1, 2, 3, 4, 5, 6, 7]
if not all(k in c for k in need):
    print("could not parse all counters from log; got:", c, file=sys.stderr)
    sys.exit(1)
iters, empty, raw, borrow = c[0], c[1], c[2], c[3]
copy, cbytes, copy2, cbytes2 = c[4], c[5], c[6], c[7]
# Per-operation instruction count = (variant_total - empty_loop_total)/iters:
# subtracting the empty loop removes the shared loop-control + bracket overhead.
def per(x): return (x - empty) / iters
pr, pb, pc, pc2 = per(raw), per(borrow), per(copy), per(copy2)
print(f"iterations              : {iters}")
print(f"empty-loop total        : {empty}")
print(f"raw     total/perop     : {raw} / {pr:.2f}")
print(f"borrow  total/perop     : {borrow} / {pb:.2f}")
print(f"copy {cbytes:>4}B total/perop : {copy} / {pc:.2f}")
print(f"copy {cbytes2:>4}B total/perop : {copy2} / {pc2:.2f}")
print()
print(f"borrow / raw ratio      : {pb/pr:.2f}x  (+{pb-pr:.0f} instr/op, payload-independent)")
print(f"copy {cbytes}B / raw ratio  : {pc/pr:.2f}x")
print(f"copy {cbytes2}B / raw ratio  : {pc2/pr:.2f}x")
print(f"borrow vs copy {cbytes}B     : borrow {pc/pb:.1f}x cheaper")
print(f"borrow vs copy {cbytes2}B     : borrow {pc2/pb:.1f}x cheaper")
print(f"copy grows with payload : {pc:.0f} -> {pc2:.0f} instr for {cbytes}B -> {cbytes2}B "
      f"(O(payload)); borrow/raw stay flat (O(1))")
PY

echo
echo "run-borrow-cost-probe.sh completed. Full serial log: $LOG_FILE"
