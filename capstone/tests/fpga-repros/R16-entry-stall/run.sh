#!/usr/bin/env bash
# R-16 bitstream acceptance test: does the resident bitstream carry the capability
# operand-forwarding fix (capstone-ariane 7aac52f93)?
#
#   VERDICT: R-16 ABSENT  -> the reproducer entered; the bitstream has the fix
#   VERDICT: R-16 PRESENT -> the reproducer entry-stalled; the bitstream LACKS the fix,
#                            which also means R-14 is back (see ../ARCHIVED/R14-frame-pad/)
#
# Why the control gate matters: the known-entering control fails roughly 1 in 5 boots, and 2
# of 3 in the 2026-08-04 session. A boot whose control fails carries NO verdict about
# anything, so this script discards it and retries rather than reporting a false stall.
#
# Ordering is not a style choice: a stalled or wedged domain takes the core with it, so the
# control goes FIRST and the single expected-to-stall image goes LAST.
set -uo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)" || exit 1
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1

BUILD=${R16_BUILD:-1}            # 0 = reuse whatever is already staged
ATTEMPTS=${R16_ATTEMPTS:-3}
OUT_DIR_1=/tmp/capstone/r16-sb1
CTL=${R16_CONTROL:-f10}          # known-entering control already baked in the image

O=capstone/caplifive-system/sw/buildroot/overlay/test-domains
T=capstone/caplifive-system/sw/buildroot/build/target/test-domains

: "${FPGA_FW:?set FPGA_FW to the built fw_payload.bin}"
# FPGA_URL is a CREDENTIAL: read it from the secret file, never hardcode, never echo it.
export FPGA_URL="${FPGA_URL:-$(cat ~/.claude-c/secrets/fpga-console-url)}"

if [[ "$BUILD" == "1" ]]; then
  echo "== building the reproducer (SQLITE_STATIC_BUILTINS=1)"
  SQLITE_STATIC_BUILTINS=1 OUT_DIR="$OUT_DIR_1" \
    bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh >/tmp/capstone/r16-build.log 2>&1 || {
      echo "BUILD FAILED -- tail:"; tail -20 /tmp/capstone/r16-build.log; exit 1; }

  src="$OUT_DIR_1/sqlite_silicon.dom"
  sz=$(stat -c%s "$src")
  echo "   built: $(sha256sum "$src" | cut -c1-16)  size=$sz"
  # Identify by SIZE, not filename: a stale staged image of the other shape reads as a pass.
  [[ "$sz" == "1551512" ]] || echo "   NOTE: size differs from the pinned 1551512 (see IMAGE-HASHES.txt)"

  cp -f "$src" "$O/sb1.dom" && cp -f "$src" "$T/sb1.dom"   # BOTH -- buildroot packs $T
  ( cd capstone/caplifive-system/sw/buildroot
    # linux-rebuild FIRST: buildroot does not track overlay/ -> cpio, so an OpenSBI-only
    # relink silently ships the OLD initramfs.
    make build LINUX_PAYLOAD=1 A=linux-rebuild  CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)" >/tmp/capstone/r16-lin.log 2>&1 \
      && make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)" >/tmp/capstone/r16-sbi.log 2>&1
  ) || { echo "IMAGE REBUILD FAILED"; exit 1; }
fi

export SQLITE_STAGE_TIMEOUT=${SQLITE_STAGE_TIMEOUT:-150}

for a in $(seq 1 "$ATTEMPTS"); do
  OUT=/tmp/capstone/r16-run-$a.txt; LOG=/tmp/capstone/r16-run-$a.log; WD=/tmp/capstone/r16-run-$a.wd
  rm -f "$OUT" "$LOG" "$WD"
  export PROBE_SCOPED_OUT="$OUT"
  export SQLITE_STAGE_DOMS="/test-domains/$CTL.dom:0,/test-domains/sb1.dom:0"

  python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py >"$LOG" 2>&1 &
  R=$!
  # ENTRY_STALL_S >= 260: the JTAG upload is 133-227s of legitimate UART silence.
  ABORT_ON_ENTRY_STALL=1 bash capstone/tests/rtl-smoke/board-watchdog.sh "$LOG" 300 "$R" >"$WD" 2>&1 &
  wait $R

  echo "-- attempt $a"
  awk '/^===== /{if(l)printf "   %-30s G/enter=%-3s last=%s\n",l,(g?"YES":"NO"),s; l=$2; g=0; s="-"}
       /SQ: G\/enter/{g=1} /SQ: /{s=$0}
       END{if(l)printf "   %-30s G/enter=%-3s last=%s\n",l,(g?"YES":"NO"),s}' "$OUT" 2>/dev/null

  # A boot is VALID only if the control both ENTERED and RETURNED.
  if ! awk "/^===== .*$CTL/,/^===== .*sb1/" "$OUT" 2>/dev/null | grep -q "SQ: obs="; then
      echo "   control did not return -> boot VOID, retrying"; continue
  fi
  if ! grep -q "sb1" "$OUT" 2>/dev/null; then
      echo "   reproducer block absent -> boot VOID, retrying"; continue
  fi
  if awk '/===== .*sb1/,0' "$OUT" | grep -q "SQ: G/enter"; then
      echo "   VERDICT: R-16 ABSENT  -- reproducer ENTERED; bitstream HAS the forwarding fix"
  else
      echo "   VERDICT: R-16 PRESENT -- reproducer entry-stalled; bitstream LACKS the fix."
      echo "            R-14 is therefore also back: see ../ARCHIVED/R14-frame-pad/"
  fi
  exit 0
done

echo "no valid boot in $ATTEMPTS attempts (control never returned) -- no verdict"
exit 75          # 75 = infra, per the driver convention; retry, do not record a result
