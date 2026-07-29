#!/usr/bin/env bash
# One-line, path-safe launcher for a silicon-ladder board session.
#
# WHY THIS EXISTS. Every board run needs the same five-step preamble: source the test
# env, read the secret console URL, point FPGA_FW at a firmware, cd to the driver, run.
# Hand-assembling that on the command line has broken THREE TIMES IN ONE DAY, always the
# same way: a relative path used AFTER a `cd`, which fails, which aborts the `&&` chain,
# which means THE BOARD RUN NEVER STARTS. Twice that was only noticed because a log file
# was missing; once it was read as a board failure. Every path here is derived from this
# script's own location, so there is nothing relative left to get wrong.
#
#   usage:  bash capstone/tests/rtl-smoke/run-board-ladder.sh <rung> [rung...]
#
#   env:
#     FPGA_FW        firmware to load   (default: the caplifive-system fpga/ariane build)
#     LADDER_ONE_BOOT / LADDER_DISTINCT_VA   default 1/1 (R-3 workaround; see ISSUES)
#     DOMAIN_GLUE    default interp
#     LOG            where to write     (default: /tmp/capstone/board-<date>.log)
#     anything else  passed straight through (INTERP_*, LADDER_EXEC_*, ...)
#
# The console URL is read from ~/.claude-c/secrets/fpga-console-url and is NEVER echoed,
# logged, or exported into anything that gets captured. It is secret.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)      # .../capstone/tests/rtl-smoke
REPO=$(cd -- "$HERE/../../.." && pwd)                          # repo root

[[ $# -ge 1 ]] || { echo "usage: $0 <rung> [rung...]" >&2; exit 2; }

# ---- preconditions, each fatal and each reported by NAME -----------------------------
ENVSH="$REPO/capstone/tests/capstone-test-env.sh"
[[ -f "$ENVSH" ]] || { echo "FATAL: missing $ENVSH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENVSH" >/dev/null 2>&1
[[ -n "${CAPSTONE_CLANG:-}" ]] || { echo "FATAL: env did not export CAPSTONE_CLANG" >&2; exit 1; }

URLFILE="$HOME/.claude-c/secrets/fpga-console-url"
[[ -f "$URLFILE" ]] || { echo "FATAL: no board URL at $URLFILE" >&2; exit 1; }
FPGA_URL=$(tr -d '[:space:]' < "$URLFILE"); export FPGA_URL
[[ -n "$FPGA_URL" ]] || { echo "FATAL: board URL file is empty" >&2; exit 1; }

FPGA_FW=${FPGA_FW:-$REPO/capstone/caplifive-system/sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin}
[[ -f "$FPGA_FW" ]] || { echo "FATAL: firmware not found: $FPGA_FW" >&2; exit 1; }
export FPGA_FW
export FPGA_FW_NAME=${FPGA_FW_NAME:-$(basename "$FPGA_FW")}

# ---- firmware sanity, BEFORE spending a board session --------------------------------
# A payload-less image (~2 MB instead of ~17 MB) boots to nothing AND drops the baked
# initramfs; that was shipped once by omitting LINUX_PAYLOAD=1. Warn loudly rather than
# discover it on the board.
FW_SIZE=$(stat -c %s "$FPGA_FW")
printf 'firmware : %s\n' "$FPGA_FW"
printf 'size     : %s bytes%s\n' "$FW_SIZE" \
  "$( (( FW_SIZE < 10000000 )) && echo '   <-- WARNING: suspiciously small, kernel payload may be missing' )"
if command -v "${CAPSTONE_LLVM_BIN:-}/llvm-nm" >/dev/null 2>&1; then
  ELF=${FPGA_FW%.bin}.elf
  if [[ -f "$ELF" ]]; then
    LBL=$("$CAPSTONE_LLVM_BIN/llvm-nm" "$ELF" 2>/dev/null |
          grep -oE '_create_domain\.[0-9]+' | sed 's/.*\.//' | sort -n | tail -1)
    # <=9 means the monitor was built WITHOUT the globals blob copy: the domain then
    # reads uninitialised dom_data and fails in a way that looks like a silicon bug.
    printf 'monitor  : create_domain labels up to %s%s\n' "${LBL:-?}" \
      "$( [[ -n "$LBL" && "$LBL" -le 9 ]] && echo '   <-- WARNING: no blob-copy block (stale .c.S?)' )"
  fi
fi

# ---- PAYLOAD FRESHNESS: does the firmware embed the kernel that was actually built? --
# Measured 2026-07-30: the documented rebuild recipe links fw_payload.bin ~25 s BEFORE the
# kernel image finishes, because BR2_TARGET_OPENSBI_LINUX_PAYLOAD is commented out and the
# payload is injected out-of-band via LINUX_PAYLOAD=1 -- so buildroot has no ordering edge
# from opensbi to the kernel. The firmware then carries a kernel (and initramfs, and every
# baked domain) one generation old, at the correct SIZE, which is why a size-only guard
# cannot see it. Compare CONTENT.
IMGF="$REPO/capstone/caplifive-system/sw/buildroot/build/images/Image"
if [[ -f "$IMGF" && -r "$FPGA_FW" ]]; then
  if ! python3 - "$IMGF" "$FPGA_FW" <<'PYEOF'
import sys, hashlib
img = open(sys.argv[1], 'rb').read()
fw  = open(sys.argv[2], 'rb').read()
emb = fw[0x200000:0x200000 + len(img)]
ok = emb == img
print("payload  : %s (kernel %s)" % (
    "matches build/images/Image" if ok else "STALE -- embeds a DIFFERENT kernel",
    hashlib.sha256(img).hexdigest()[:16]))
sys.exit(0 if ok else 1)
PYEOF
  then
    echo "           ^^ rebuild with: make -C buildroot ... linux-rebuild-with-initramfs" >&2
    echo "              THEN relink opensbi. Results from this firmware may not reflect" >&2
    echo "              your latest kernel/initramfs/baked domains." >&2
  fi
fi

export LADDER_ONE_BOOT=${LADDER_ONE_BOOT:-1}
export LADDER_DISTINCT_VA=${LADDER_DISTINCT_VA:-1}
export DOMAIN_GLUE=${DOMAIN_GLUE:-interp}
export LADDER_RUNGS="$*"

LOG=${LOG:-/tmp/capstone/board-$(date +%Y%m%d-%H%M%S).log}
mkdir -p "$(dirname "$LOG")"

printf 'rungs    : %s\n' "$LADDER_RUNGS"
printf 'glue     : %s   one-boot=%s distinct-va=%s\n' \
  "$DOMAIN_GLUE" "$LADDER_ONE_BOOT" "$LADDER_DISTINCT_VA"
printf 'log      : %s\n\n' "$LOG"

cd "$HERE" || exit 1
# USE THE VENV INTERPRETER IF IT HAS websocket-client. Without it python-socketio falls
# back to the POLLING transport, which batches and reorders -- and measured on 2026-07-30
# the fast transfer tier (burst=16) failed its sha check 5 times out of 5, i.e. 100%, so
# every transfer degraded to 1 char/emit and one degraded further still. The audit costed
# the retry ladder at 47-90% of a session's wall clock. A real WebSocket is the cheapest
# thing that might fix the corruption at its source rather than working around it.
PY=python3
if [[ -x /tmp/capstone/fpga-venv/bin/python ]] && \
   /tmp/capstone/fpga-venv/bin/python -c "import websocket" 2>/dev/null; then
  PY=/tmp/capstone/fpga-venv/bin/python
  echo "python   : venv (websocket transport available)"
else
  echo "python   : system python3 -- POLLING transport only, transfers will be slow"
fi
timeout "${BOARD_TIMEOUT:-5400}" "$PY" fpga_driver/run_ladder_perf_fpga.py > "$LOG" 2>&1
rc=$?

echo
sed -n '/silicon-ladder FPGA perf/,$p' "$LOG" | head -20
echo
# The board must ALWAYS end powered off and unlocked. A timeout that kills the runner
# skips its cleanup, which has stranded the board powered-on before.
if grep -q "powered off" "$LOG" && grep -q "unlocked" "$LOG"; then
  echo "board: powered off + unlocked (clean)"
else
  echo "board: WARNING - no clean shutdown in the log. RELEASE IT MANUALLY."
fi
echo "exit=$rc  log=$LOG"
exit $rc
