#!/usr/bin/env bash
# Build the MicroPython domain and run it under Capstone QEMU. One command, one verdict.
#
#   bash capstone/ports/micropython/run-micropython.sh
#   MPY_EXPECT=0x20046 DOMAIN_EXTRA_DEFS=-DMPY_RETURN_OUTPUT ... run-micropython.sh
#
# WHY THIS EXISTS. Before it, "does the port still run" meant building by hand, copying
# into the 9p share by hand, calling run-domain-smoke.py by hand and comparing the retval
# by eye. A move or a refactor could not be checked, so it was not checked.
#
# IT DOES NOT SOURCE repro-lib.sh, which can already do most of this. That driver belongs
# to the bug corpora, and a port that depends on its corpus cannot be reviewed or merged
# without it.
#
# Exit 0 = the expected retval. Exit 75 = infrastructure, NO verdict (the smoke runner's
# own convention, passed through unchanged). Exit 1 = the domain returned something else.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# 0x4D500000 is `0x4D500000 | rc` with rc = 0: "MP" plus a zero status from do_str().
EXPECT=${MPY_EXPECT:-0x4D500000}
DOM_NAME=${DOM_NAME:-micropython}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/micropython-silicon}
SHARE=${CAPSTONE_TMP_ROOT:-/tmp/capstone}/capstone-runtime-qemu-share
LOG=${MPY_RUN_LOG:-$CAPSTONE_TMP_ROOT/$DOM_NAME-run.log}

if [[ -z "${MPY_SKIP_BUILD:-}" ]]; then
  echo "== building $DOM_NAME"
  bash "$SCRIPT_DIR/build-micropython-silicon.sh" || { echo "build failed" >&2; exit 1; }
fi
[[ -f "$OUT_DIR/$DOM_NAME.dom" ]] || { echo "no image at $OUT_DIR/$DOM_NAME.dom" >&2; exit 1; }

mkdir -p "$SHARE"
cp -f "$OUT_DIR/$DOM_NAME.dom" "$SHARE/"
echo "== running (md5 $(md5sum "$SHARE/$DOM_NAME.dom" | cut -c1-12))"

# --timeout-multiplier 3: the 9p mount and the login have both timed out on an otherwise
# healthy boot. A flake costs a rerun; a too-short timeout costs a wrong verdict.
python3 "$REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --timeout-multiplier "${MPY_TIMEOUT_MULT:-3}" --log-file "$LOG" \
  "$SHARE/$DOM_NAME.dom" >"$LOG.runner" 2>&1
rc=$?
if [[ $rc -eq 75 ]]; then
  echo "INFRA FLAKE, no verdict -- rerun. See $LOG.runner" >&2; exit 75
fi

# The retval, read from the UART log rather than from the runner's exit status: a domain
# that returns the WRONG value still exits 0 there.
got=$(grep -a -o 'retval = [0-9-]*' "$LOG" | tail -1 | awk '{print $3}')
[[ -n "$got" ]] || { echo "no retval in $LOG -- treating as infrastructure" >&2; exit 75; }
if [[ "$got" -eq $((EXPECT)) ]]; then
  printf 'PASS  retval = %s (0x%x), as expected\n' "$got" "$got"; exit 0
fi
printf 'FAIL  retval = %s (0x%x), expected %s (0x%x)\n' "$got" "$got" "$((EXPECT))" "$((EXPECT))" >&2
exit 1
