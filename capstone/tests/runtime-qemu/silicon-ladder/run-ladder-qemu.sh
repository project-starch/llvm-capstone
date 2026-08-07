#!/usr/bin/env bash
# Generic Stage-0 QEMU runner for a ladder rung <base>: builds <base>_host.c as the
# native oracle and <base>_app.c as a silicon-config domain, runs it in a pure-cap
# domain on QEMU (gp fabrication OFF), and asserts the domain returns the oracle
# value.  usage: run-ladder-qemu.sh <base>   (e.g. matmult_int, init_probe)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../capstone-test-env.sh"

[[ $# -eq 1 ]] || { echo "usage: $0 <base>" >&2; exit 1; }
BASE=$1
APP="$SCRIPT_DIR/${BASE}_app.c"
HOST="$SCRIPT_DIR/${BASE}_host.c"
[[ -f "$APP" && -f "$HOST" ]] || { echo "missing ${BASE}_app.c / ${BASE}_host.c" >&2; exit 1; }

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/silicon-ladder}
mkdir -p "$OUT_DIR"
# RUNG_NAME names the artifacts when one <base> is built at many parameterisations
# (fdreg is built at a dozen FDREG_* settings per session). Without it every arm
# overwrites $OUT_DIR/<base>.dom and the marker written below vouches for whichever
# build ran last -- the stale-artifact failure this gate exists to prevent.
RUNG_NAME=${RUNG_NAME:-$BASE}
DOM="$OUT_DIR/${RUNG_NAME}.dom"

# HOST_EXTRA_CFLAGS defaults to DOMAIN_EXTRA_CFLAGS so a parameterised rung's NATIVE
# ORACLE is computed at the same parameterisation as the domain.
#
# It was previously a bare `cc -O0`, which is a silent wrong-oracle bug for any rung
# whose kernel is selected by -D: fdreg_host.c and fdreg_app.c include the same
# fdreg_kernel.h, so the domain built at -DFDREG_STAGE=32 was being compared against a
# host binary built at the header's DEFAULT stage 1. The comparison then either fails
# for the wrong reason or -- worse -- passes because two unrelated stages happen to
# return the same number. Split as a separate variable because DOMAIN_EXTRA_CFLAGS may
# legitimately carry target-only flags that the host `cc` would reject.
HOST_EXTRA_CFLAGS=${HOST_EXTRA_CFLAGS:-${DOMAIN_EXTRA_CFLAGS:-}}
# shellcheck disable=SC2086
cc -O0 $HOST_EXTRA_CFLAGS -o "$OUT_DIR/${RUNG_NAME}_host" "$HOST"
EXPECT_DEC=$("$OUT_DIR/${RUNG_NAME}_host")
echo "oracle: $RUNG_NAME = $EXPECT_DEC   (host flags: ${HOST_EXTRA_CFLAGS:-none})"

bash "$SCRIPT_DIR/build-ladder-domain.sh" "$APP" "$DOM"

: "${CAPSTONE_GP_FABRICATE:=0}"; export CAPSTONE_GP_FABRICATE
[[ -n "${CAPSTONE_GP_STANDIN:-}" ]] && export CAPSTONE_GP_STANDIN
python3 "$SCRIPT_DIR/../run-domain-smoke.py" "$DOM"

LOG="$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
MARK="__CAPSTONE_LADDER_$(echo "$BASE" | tr '[:lower:]' '[:upper:]')_PASSED__"
if grep -aqF "retval = $EXPECT_DEC" "$LOG"; then
  echo "$MARK (retval = $EXPECT_DEC)"
else
  echo "FAIL: expected 'retval = $EXPECT_DEC' not found" >&2
  grep -aE 'retval|Cap mem|halt|fault' "$LOG" | tail -8 >&2 || true
  exit 1
fi
