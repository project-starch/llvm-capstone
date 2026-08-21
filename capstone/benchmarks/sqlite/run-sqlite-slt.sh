#!/usr/bin/env bash
# Run one SQLLogicTest file inside the silicon-config SQLite domain, under QEMU.
#
# Stage 1 of the regression-suite plan: one call_dom per .test file, the file delivered
# whole in the payload region. See slt/slt_runner.h for why the runner is shared with the
# native baseline and why the DIFFERENCE between them is the result rather than the
# absolute rate.
#
# ONE ASSIGNMENT SETS THE REGION SIZE FOR BOTH BUILDS. That is deliberate and is the whole
# reason this script exists rather than two build invocations: host and domain are separate
# compilations of one #define, and a drift between them is silent. The domain also refuses
# to run on a mismatch, but that gate is the backstop, not the mechanism.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# 1 MiB is MEASURED, not assumed: 4 KiB and 1 MiB both work end-to-end, 64 MiB fails at
# map_region with no markers at all. The input gets the top half, so a .test file must be
# under REGION/2 -- select1 and select2 fit at 1 MiB; select3/4/5 need 2 MiB or 4 MiB and
# that ceiling has NOT been measured, so raising it is an experiment, not a setting.
REGION=${SLT_REGION_SIZE:-1048576}
export SQLITE_HC_REGION_SIZE=$REGION

# OUT_DIR resolved and exported BEFORE the builds -- see run-sqlite-silicon.sh:19 for the
# failure this ordering prevents (the two builds otherwise fall back to different defaults).
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-slt}
export OUT_DIR

SLT_TEST=${SLT_TEST:-}
if [[ -z "$SLT_TEST" ]]; then
  CORPUS=$(bash "$SCRIPT_DIR/fetch-sqllogictest.sh")
  SLT_TEST="$CORPUS/select1.test"
fi
[[ -f "$SLT_TEST" ]] || { echo "ERROR: no such test file: $SLT_TEST" >&2; exit 1; }

# REFUSE AN OVERSIZED FILE HERE, not in the guest. The host also refuses it, but finding
# out after a QEMU boot costs minutes and the arithmetic is available now.
MAXIN=$(( REGION / 2 ))
SZ=$(stat -c%s "$SLT_TEST")
if (( SZ >= MAXIN )); then
  echo "ERROR: $SLT_TEST is $SZ bytes; the input half of a $REGION-byte region holds $MAXIN." >&2
  echo "       Raise SLT_REGION_SIZE (1 MiB is the largest MEASURED size) or pick a smaller file." >&2
  exit 1
fi

DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_SQLITE_SLT=1 -DSQLITE_HC_REGION_SIZE=${REGION}UL"
HOST_EXTRA_DEFS="${HOST_EXTRA_DEFS:-} -DSQLITE_HC_REGION_SIZE=${REGION}UL"
export DOMAIN_EXTRA_DEFS HOST_EXTRA_DEFS

bash "$SCRIPT_DIR/build-sqlite-silicon.sh"
bash "$SCRIPT_DIR/build-sqlite-host.sh"

SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/sqlite-slt-share}
rm -rf "$SHARE"; mkdir -p "$SHARE"
DOM="$OUT_DIR/sqlite_silicon.dom"
[[ -f "$DOM" ]] || { echo "ERROR: $DOM does not exist -- the build produced nothing" >&2; exit 1; }
cp -f "$DOM" "$SHARE/"
cp -f "$OUT_DIR/sqlite_host.user" "$SHARE/"
cp -f "$SLT_TEST" "$SHARE/case.test"
echo "== domain $(sha256sum "$DOM" | cut -c1-16)  region=$REGION  case=$(basename "$SLT_TEST") ($SZ bytes)"

# SLT-SUMMARY is the only marker asserted, and `completed=1` is part of it on purpose: a
# summary without it is a run that stopped early, which must never read as a pass.
python3 "$ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" \
  --log-file "${SLT_LOG_FILE:-$OUT_DIR/sqlite-slt.log}" \
  --timeout-multiplier 12 \
  --guest-command \
    'cp /mnt/host/sqlite_host.user /tmp/h.user && chmod 0755 /tmp/h.user && /tmp/h.user /mnt/host/sqlite_silicon.dom --slt /mnt/host/case.test' \
  --success-marker 'SLT-SUMMARY' \
  --success-marker 'completed=1'

echo "__CAPSTONE_SQLITE_SLT_RAN__"
