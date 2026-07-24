#!/usr/bin/env bash
# gp-free domain proof: build a real integer app (data globals + a non-inlined
# call graph) as a gp-free / cjalr-free domain and run it in a pure-capability
# domain on QEMU, with the gp = PCC(cursor 0) fabrication DISABLED.
#
# This is the functional (Experiment A) existence proof that a real globals-using
# program runs correctly under pervasive capabilities with the silicon-shaped
# gp-free ABI: intra-domain calls/returns are plain jal/jalr within PCC (no
# cjalr), and global data is addressed with `scc gp` (absolute cursor) off an
# image-covering gp the MONITOR delivers via the cscratch stack region -- no
# hardware gp fabrication (the RTL never establishes gp, and an image-bounded cap
# forced to cursor 0 is unrepresentable). See README.md.
#
# Requires: clang/lld with -capstone-gp-free; QEMU with the CAPSTONE_GP_FABRICATE
# guard; and fw_jump.elf built from the create_domain gp-delivery change (both
# documented in README.md). Env CAPSTONE_GP_STANDIN=1 substitutes a QEMU-delivered
# representable gp when the monitor image is not rebuilt.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/gp-free-domain}
mkdir -p "$OUT_DIR"
DOM="$OUT_DIR/gpfree_app.dom"

# Build: gp-free entry glue + the app compiled with -capstone-gp-free at -O0
# (so the helper() call is a real, non-inlined plain call).
START_SRC="$SCRIPT_DIR/start-gpfree-cscratch.S" \
EXTRA_CLANG_FLAGS="-mllvm -capstone-gp-free" \
DOMAIN_OPT_LEVEL=-O0 \
  bash "$SCRIPT_DIR/../build-domain.sh" "$SCRIPT_DIR/gpfree_app.c" "$DOM"

# Static gate: the domain must be strictly gp-free / cjalr-free.
NCJALR=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$DOM" | grep -cE '\bcjalr\b' || true)
NCINCGP=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$DOM" | grep -cE 'cincoffset[[:space:]]+[a-z0-9]+, gp' || true)
NSCCGP=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$DOM" | grep -cE 'scc[[:space:]]+[a-z0-9]+, gp' || true)
echo "static: cjalr=$NCJALR cincoffset-gp=$NCINCGP scc-gp=$NSCCGP"
if [[ "$NCJALR" != "0" || "$NCINCGP" != "0" || "$NSCCGP" -lt 1 ]]; then
  echo "FAIL: domain is not gp-free/cjalr-free (expected cjalr=0, cincoffset-gp=0, scc-gp>=1)" >&2
  exit 1
fi

# Run with the fabrication OFF. If the monitor image carries the create_domain
# gp-delivery change, real cscratch delivery works with no stand-in; otherwise
# set CAPSTONE_GP_STANDIN=1 to have QEMU deliver a representable gp.
: "${CAPSTONE_GP_FABRICATE:=0}"
export CAPSTONE_GP_FABRICATE
[[ -n "${CAPSTONE_GP_STANDIN:-}" ]] && export CAPSTONE_GP_STANDIN

python3 "$SCRIPT_DIR/../run-domain-smoke.py" "$DOM"

EXPECT="Called dom (1-th time) retval = 554745961"
LOG="$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
if grep -aqF "$EXPECT" "$LOG"; then
  echo "__CAPSTONE_GPFREE_DOMAIN_PASSED__ ($EXPECT)"
else
  echo "FAIL: expected '$EXPECT' not found in $LOG" >&2
  grep -aE 'retval|Cap mem|halt' "$LOG" | tail -5 >&2 || true
  exit 1
fi
