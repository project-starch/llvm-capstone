#!/bin/bash
# One board session on a PRIVATE firmware (no bake: the payload Image is the shared board build's current one, copied).
# Usage: [SUPMON_SPEEDTESTS=n] board-supmon.sh <tag> <fw_payload.bin>. Stages: k800, n x the staged speedtest1.dom, k800.
# Mirrors board-r1e4.sh's runner + watchdog section (the stages runner is the driver of record).
main() {
  set -u
  local TAG=$1 FW=$2 R=$HOME/dev/llvm-capstone OUT=$HOME/capstone-artifacts/unify/board-$1
  mkdir -p "$OUT"; cp -f "$FW" "$OUT/fw_payload.bin"
  export FPGA_URL="$(cat "${CAPSTONE_FPGA_URL_FILE:-$HOME/.claude-kisp/secrets/fpga-console-url}")"
  export FPGA_FW="$OUT/fw_payload.bin" FPGA_BITSTREAM=${SUPMON_BITSTREAM:-caplifive_supcall_36a641e0b.bit} REFUSAL_RECORD=1
  export PREFLIGHT_ALLOW_SHORT=1 PREFLIGHT_ALLOW_SLOTS=1
  # The image carries the k800 RELINKED at 0x20000 (589ceee3; speedtest1.dom enters at 0x10000, R-3 / preflight C15),
  # so preflight reads that control's own QEMU-pass record.
  export PREFLIGHT_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc K800_ORACLES=$HOME/capstone-artifacts/k800-relinked-0x20000/orc K800_HASH=589ceee3853c6092
  export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
  export SQLITE_HOST=/test-domains/sqlite_host_rr.user SQLITE_STAGE_TIMEOUT=600 SQLITE_IDLE_S=600
  local K="/test-domains/lpc|k800:/test-domains/k800.dom"
  local S="/test-domains/sqlite_host_rr.user|/test-domains/speedtest1.dom:--speedtest1 --testset main --size 1 --verify"
  # SUPMON_SPEEDTESTS: how many speedtest runs between the two k800 controls (default 1; C5q runs 3).
  local D="$K" i
  for i in $(seq 1 "${SUPMON_SPEEDTESTS:-1}"); do D="$D,$S"; done
  export SQLITE_STAGE_DOMS="$D,$K"
  export PROBE_SCOPED_OUT=$OUT/boot.txt PROBE_RAW_OUT=$OUT/boot-raw.txt
  echo "$(date +%H:%M:%S) === $TAG fw $(sha256sum "$FPGA_FW" | cut -c1-12) ===" | tee -a $OUT/log
  cd $R/capstone/tests/rtl-smoke
  timeout 3600 python3 -m fpga_driver.run_sqlite_stages_fpga > $OUT/driver.log 2>&1 &
  local RUNNER=$!
  ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=420 bash board-watchdog.sh "$OUT/driver.log" 900 "$RUNNER" > $OUT/watchdog.log 2>&1 &
  local WD=$!
  wait $RUNNER; local rc=$?
  kill $WD 2>/dev/null; wait $WD 2>/dev/null
  echo "$(date +%H:%M:%S) runner rc=$rc" | tee -a $OUT/log
  return $rc
}
main "$@"; exit $?
