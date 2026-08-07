#!/usr/bin/env bash
# Verify a ladder rung under QEMU and stage it for the board, writing the `.qemu-pass`
# marker that preflight-board-run.sh's C13 gate demands.
#
#   usage: RUNG=<name> [DOMAIN_EXTRA_CFLAGS=...] [DOMAIN_BASE_VA=...] \
#            bash verify-and-stage-rung.sh <base>          # e.g. fdreg
#
# WHY THIS EXISTS. C13 blocks a rung that has no recorded QEMU pass, and its comment says
# "`.qemu-pass` is written by the verify step next to the oracle". That verify step did not
# exist. Nothing in the tree wrote a marker, so every marker present had been produced by
# hand -- four of them transcribed out of a SHA256SUMS file minutes AFTER the boots they
# vouched for. A gate whose evidence is hand-typed is a gate that records an intention, so
# this script is the step the comment was describing.
#
# THE HONEST PART, stated in the marker itself. Two different binaries are involved:
#
#   <base>_app.c       writes *res only (4 B)     -- this is what QEMU runs
#   <base>_fpga_app.c  writes res[0..2] (24 B)    -- this is what the BOARD runs
#
# The QEMU harness hands the domain an 8-byte shared region, so the _fpga build faults at
# `sd a1, 0x8(a2)` (res[1]) before it can report anything: running it under QEMU produces a
# capability fault, not a verdict. The two builds share the kernel header and the -D flags,
# so QEMU validates the COMPUTE, and the board image is built from the same source at the
# same parameterisation -- but they are not the same bytes, and the marker says so rather
# than implying an image was run that was not. C13 compares against the staged (board)
# hash, so that is what goes in first; the QEMU-tested hash is recorded beside it.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../capstone-test-env.sh"

[[ $# -eq 1 ]] || { echo "usage: RUNG=<name> $0 <base>" >&2; exit 1; }
BASE=$1
RUNG=${RUNG:-$BASE}
APP="$SCRIPT_DIR/${BASE}_app.c"
FPGA_APP="$SCRIPT_DIR/${BASE}_fpga_app.c"
HOST="$SCRIPT_DIR/${BASE}_host.c"
for f in "$APP" "$FPGA_APP" "$HOST"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/silicon-ladder}
ORACLES=${LADDER_FPGA_DIR:-$CAPSTONE_TMP_ROOT/ladder-fpga}
OVERLAY=${OVERLAY_DIR:-$SCRIPT_DIR/../../../caplifive-system/sw/buildroot/overlay/test-domains}
TARGET=${TARGET_DIR:-$SCRIPT_DIR/../../../caplifive-system/sw/buildroot/build/target/test-domains}
mkdir -p "$OUT_DIR" "$ORACLES"

QDOM="$OUT_DIR/${RUNG}.qemu.dom"
BDOM="$OUT_DIR/${RUNG}.dom"

# Delete before building. A build that fails leaves the PREVIOUS artifact in place, and a
# stale .dom read as a fresh one has already cost this project a run.
rm -f "$QDOM" "$BDOM" "$ORACLES/$RUNG.qemu-pass"

# ---- 1. native oracle, at the SAME parameterisation as the domain ------------------
HOST_EXTRA_CFLAGS=${HOST_EXTRA_CFLAGS:-${DOMAIN_EXTRA_CFLAGS:-}}
# shellcheck disable=SC2086
cc -O0 $HOST_EXTRA_CFLAGS -o "$OUT_DIR/${RUNG}_host" "$HOST"
ORACLE=$("$OUT_DIR/${RUNG}_host")
echo "oracle: $RUNG = $ORACLE"

# ---- 2. QEMU build + run ----------------------------------------------------------
bash "$SCRIPT_DIR/build-ladder-domain.sh" "$APP" "$QDOM" >/dev/null
[[ -f "$QDOM" ]] || { echo "QEMU build produced no artifact" >&2; exit 1; }

: "${CAPSTONE_GP_FABRICATE:=0}"; export CAPSTONE_GP_FABRICATE
python3 "$SCRIPT_DIR/../run-domain-smoke.py" "$QDOM" >/dev/null 2>&1 || true

LOG="$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
if ! grep -aqF "retval = $ORACLE" "$LOG"; then
  echo "QEMU FAIL: expected 'retval = $ORACLE'" >&2
  grep -aE 'retval|Cap mem|halt|fault' "$LOG" | tail -6 >&2 || true
  exit 1
fi
echo "qemu: PASS (retval = $ORACLE)"

# ---- 3. board build, same source + same flags -------------------------------------
bash "$SCRIPT_DIR/build-ladder-domain.sh" "$FPGA_APP" "$BDOM" >/dev/null
[[ -f "$BDOM" ]] || { echo "board build produced no artifact" >&2; exit 1; }

BSHA=$(sha256sum "$BDOM" | cut -d' ' -f1)
QSHA=$(sha256sum "$QDOM" | cut -d' ' -f1)

# ---- 4. geometry, recorded so a wrong-layout build is visible before it is staged ---
GEO=$(python3 "$SCRIPT_DIR/extract-frame-layout.py" "$BDOM" 2>/dev/null | tr -d '\n ' || echo "unavailable")

# ---- 5. stage into BOTH trees; buildroot packs $TARGET ----------------------------
mkdir -p "$OVERLAY" "$TARGET"
cp -f "$BDOM" "$OVERLAY/$RUNG.dom"
cp -f "$BDOM" "$TARGET/$RUNG.dom"

echo "$ORACLE" > "$ORACLES/$RUNG.oracle"
cat > "$ORACLES/$RUNG.qemu-pass" <<EOF
$BSHA  staged board image ($RUNG.dom, ${BASE}_fpga_app.c) -- this is what C13 compares
$QSHA  image actually RUN under QEMU (${BASE}_app.c, same header + same -D flags)
oracle=$ORACLE
flags=${DOMAIN_EXTRA_CFLAGS:-none}
base_va=${DOMAIN_BASE_VA:-default}
glue=${DOMAIN_GLUE:-generated}
geometry=$GEO
verified_at=$(date -Is)
NOTE: the board image was NOT itself run under QEMU -- the _fpga variant writes 24 B into
the QEMU harness's 8-byte shared region and faults on res[1]. QEMU validates the compute;
the board image is the same source at the same parameterisation.
EOF

echo "staged: $RUNG.dom -> overlay + target   sha ${BSHA:0:12}"
echo "marker: $ORACLES/$RUNG.qemu-pass"
echo "geometry: $GEO"
echo
echo "REMINDER: the .dom is not in the initramfs until you rebuild it --"
echo "  make build LINUX_PAYLOAD=1 A=linux-rebuild ...   # FIRST, then opensbi-rebuild"
