#!/usr/bin/env bash
# End-to-end CHERI baseline for the corpus's 31 ASAN-blind PHP defects.
#
#   1. compile the allocator-pattern rows purecap, in both allocator arms
#   2. stage them into the CheriBSD purecap rootfs and rebuild the disk image
#   3. boot CHERI-QEMU and run all rows under all three revocation configs
#   4. classify, and print measured verdicts against the pre-registered ones
#
# Prerequisite: the vehicle from capstone/tests/cheri-baseline/provision-cheri-vehicle.sh.
# Run ./selftest-classify.sh at least once first: a classifier that has never
# produced a catch cannot be trusted to report a miss.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROVISION="$SCRIPT_DIR/../cheri-baseline/provision-cheri-vehicle.sh"

CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
CHERI_WS=${CHERI_WS:-$HOME/cheri-ws}
LOCAL_DEPS=${LOCAL_DEPS:-$CHERI_WS/local-deps}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
OVERLAY=${OVERLAY:-$CHERI_ROOT/overlay-php-blind/cheri-php-blind}
WORK=${WORK:-$CHERI_ROOT/php-blind-run}
mkdir -p "$OVERLAY" "$WORK"

echo "== [0/4] kernel revocation preflight =="
bash "$SCRIPT_DIR/preflight-kernel.sh" > "$WORK/preflight-kernel.log" 2>&1 \
  || { echo "kernel preflight FAILED (see $WORK/preflight-kernel.log)"; \
       tail -6 "$WORK/preflight-kernel.log"; exit 1; }
grep -E '^\[\*\] (ident|config)' "$WORK/preflight-kernel.log"
echo "   revocation knobs present"

[ -x "$SDK/bin/clang" ] || { echo "no CHERI SDK at $SDK -- run $PROVISION first" >&2; exit 2; }

echo "== [0/4] classifier selftest =="
bash "$SCRIPT_DIR/selftest-classify.sh" > "$WORK/selftest.log" 2>&1 \
  || { echo "classifier selftest FAILED (see $WORK/selftest.log)"; exit 1; }
echo "   ok"

echo "== [1/4] compile rows purecap (both arms) =="
CHERI_SDK="$SDK" OUT="$OVERLAY" bash "$SCRIPT_DIR/compile-purecap.sh" \
  || { echo "compile failed"; exit 1; }

echo "== [2/4] stage overlay + rebuild disk image =="
CHERI_ROOT="$CHERI_ROOT" CHERI_WS="$CHERI_WS" LOCAL_DEPS="$LOCAL_DEPS" \
  OVERLAY_SRC="$OVERLAY" bash "$PROVISION" > "$WORK/provision.log" 2>&1 \
  || { echo "image rebuild failed (see $WORK/provision.log)"; tail -20 "$WORK/provision.log"; exit 1; }
# Resolve the argv ourselves -- provision's own argv names qemu-system-riscv64cheri
# while cheribuild installs qemu-system-riscv64Xcheri. See make-qemu-argv.sh.
ARGV="$WORK/qemu-argv.txt"
CHERI_ROOT="$CHERI_ROOT" SDK="$SDK" bash "$SCRIPT_DIR/make-qemu-argv.sh" "$ARGV" \
  || { echo "could not resolve the qemu argv"; exit 1; }

echo "== [3/4] boot + run all three configs =="
# qemu links libslirp.so.0 from the no-sudo prefix with no rpath, so the RUN
# needs that environment too, not just the build.
[ -f "$SCRIPT_DIR/vehicle-env-no-sudo.sh" ] && . "$SCRIPT_DIR/vehicle-env-no-sudo.sh"
python3 -I "$SCRIPT_DIR/cheri-run.py" "$ARGV" "$WORK/serial.log" /root/cheri-php-blind
echo "   cheri-run rc=$?; serial log: $WORK/serial.log"

echo "== [4/4] classify =="
python3 -I "$SCRIPT_DIR/classify.py" "$WORK/serial.log" "$SCRIPT_DIR/rows.tsv" \
  | tee "$WORK/verdicts.txt"
