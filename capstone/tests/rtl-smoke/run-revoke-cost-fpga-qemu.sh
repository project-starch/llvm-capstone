#!/usr/bin/env bash
set -euo pipefail

# QEMU plumbing-validation for the RTL/FPGA temporal-safety port (task-016):
# build the three .dom configs + controller, run each under the functional model,
# and parse the UART `RESULT` lines into the temporal-safety breakdown. The SAME
# parser (parse_results below) is what turns the board's pasted UART output into
# the paper number -- so this script both validates under QEMU and documents the
# on-board readout. To parse a pasted board capture instead of running QEMU:
#
#     ./run-revoke-cost-fpga-qemu.sh --parse-uart <file-with-RESULT-lines>
#
# FUNCTIONAL-MODEL PROXY: under -icount, mcycle advances one-per-retired-
# instruction, so the "cycles" here equal dynamic instruction counts (QEMU has no
# pipeline/cache model). On silicon these become real cycles. The number's value
# is not the deliverable here -- the plumbing (mcycle-in-domain + the two-region
# result hand-off) is. See RESULTS.md.
#
# Requires the rootfs.ext2 write lock (serialize with other QEMU suites).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"

TMP_ROOT=${TMP_ROOT:-$CAPSTONE_TMP_ROOT}
SHARE_DIR=${SHARE_DIR:-$TMP_ROOT/capstone-rtl-smoke-share}
ICOUNT=${ICOUNT:-shift=0,sleep=off}

# parse_results <logfile...>: extract the per-op cycles/op for each mode from the
# `revoke-cost-fpga: RESULT cycles/op  mode=<m>  alloc_free=<n>` UART lines and
# print the temporal-safety breakdown. Works on QEMU serial logs OR a pasted
# board UART capture (same RESULT line format).
parse_results() {
  python3 - "$@" <<'PY'
import re, sys
per = {}
for path in sys.argv[1:]:
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"RESULT cycles/op\s+mode=(\w+)\s+alloc_free=(\d+)", text):
        per[m.group(1)] = int(m.group(2))
need = ("bump", "norevoke", "revoke")
missing = [k for k in need if k not in per]
if missing:
    print(f"could not find RESULT lines for: {', '.join(missing)} (got {per})",
          file=sys.stderr)
    sys.exit(1)
bump, norev, rev = per["bump"], per["norevoke"], per["revoke"]
print("=== revoke-cost-fpga: temporal-safety overhead (cycles/op) ===")
print(f"bump      (unprotected baseline) : {bump}")
print(f"norevoke  (alloc-side)           : {norev}")
print(f"revoke    (full temporal safety) : {rev}")
print(f"alloc-side overhead (norevoke-bump)  : +{norev-bump}  ({norev/bump:.2f}x)")
print(f"revoke-at-free op   (revoke-norevoke): +{rev-norev}  (the O(1) op)")
print(f"total temporal cost (revoke-bump)    : +{rev-bump}  ({rev/bump:.2f}x over baseline)")
PY
}

if [[ "${1:-}" == "--parse-uart" ]]; then
  [[ -n "${2:-}" ]] || { echo "usage: $0 --parse-uart <file>" >&2; exit 2; }
  parse_results "$2"
  exit 0
fi

mkdir -p "$SHARE_DIR"
bash "$SCRIPT_DIR/build-revoke-cost-fpga.sh" "$SHARE_DIR" >/dev/null

logs=()
for mode in bump norevoke revoke; do
  LOG="$TMP_ROOT/rtl-revoke-cost-fpga-$mode.log"
  echo "== booting mode: $mode =="
  python3 "$SCRIPT_DIR/../runtime-qemu/run-domain-smoke.py" \
    --share-dir "$SHARE_DIR" \
    --log-file "$LOG" \
    --qemu-extra-arg=-icount \
    --qemu-extra-arg="$ICOUNT" \
    --guest-command "cp /mnt/host/revoke_cost_fpga.user /tmp/rc.user && chmod 0755 /tmp/rc.user && /tmp/rc.user /mnt/host/revoke_cost_fpga_$mode.dom" \
    --success-marker "revoke-cost-fpga: measurement complete" \
    || { echo "run-revoke-cost-fpga-qemu.sh: mode $mode FAILED; see $LOG" >&2; exit 1; }
  logs+=("$LOG")
done

echo
parse_results "${logs[@]}"
echo
echo "run-revoke-cost-fpga-qemu.sh completed. Serial logs: $TMP_ROOT/rtl-revoke-cost-fpga-*.log"
