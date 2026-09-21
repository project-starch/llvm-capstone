#!/usr/bin/env bash
# CRASH-008 matched pair: PHP 5.0.0's Zend allocator, unpatched, in a Capstone domain.
#
#   capstone/container/run.sh capstone/ports/php/zend-alloc/run-crash008.sh
#
# CONTROL FIRST. A fault on its own proves nothing: at -O0 an unrelated spill or a
# broken build can fault too. The control is the same program with one constant
# changed -- the capability bound comes from REAL_SIZE(size) instead of size -- so
# it reproduces stock PHP, where the overflow lands in the rounding slack. If the
# control does NOT complete, this script exits 75 with NO verdict rather than
# reporting a result, following the rule in capstone/bug-corpora/README.md.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"

H="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu"
OUT="$CAPSTONE_TMP_ROOT/zend-crash008"
SRC="$HERE/crash008_domain.c"
JOBSRC="$HERE/zend_capstone_alloc.h"
rm -rf "$OUT"; mkdir -p "$OUT"

# -O0 is REQUIRED, not a preference. At -O1+ the store into a buffer nothing
# reads can be eliminated, and the access under test is never emitted.
export DOMAIN_OPT_LEVEL=-O0

echo "==> building both arms (-O0)"
bash "$H/build-domain.sh" "$SRC" "$OUT/crash008_fault.dom" >/dev/null
EXTRA_CLANG_FLAGS="-DZEND_CAP_BOUNDS_REAL_SIZE" \
  bash "$H/build-domain.sh" "$SRC" "$OUT/crash008_control.dom" >/dev/null

# BUILD-TIME GATE. If the bounds narrowing is not in the image, both arms are the
# same program and both would "pass" -- a MISS indistinguishable from a broken
# build. Modelled on xlang/capstone/build-xlang-capstone.sh:87-96, which counts
# `revoke` for the same reason.
n=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT/crash008_fault.dom" 2>/dev/null | grep -ci 'shrink')
if [ "${n:-0}" -eq 0 ]; then
  echo "BUILD-INVALID: no shrink instruction in the fault arm; bounds are never narrowed" >&2
  exit 3
fi
echo "    shrink instructions in image: $n"
if cmp -s "$OUT/crash008_fault.dom" "$OUT/crash008_control.dom"; then
  echo "BUILD-INVALID: the two arms are byte-identical; -DZEND_CAP_BOUNDS_REAL_SIZE did nothing" >&2
  exit 3
fi

run_arm() {   # $1 = arm name, $2 = dom, $3... = success markers
  local arm=$1 dom=$2; shift 2
  local share="$OUT/share-$arm" log="$CAPSTONE_TMP_ROOT/crash008-$arm.log"
  rm -rf "$share"; mkdir -p "$share"
  cp "$dom" "$share/"
  bash "$H/build-capstone-test-user.sh" "$share/capstone-test.user" >/dev/null
  local markers=()
  for m in "$@"; do markers+=(--success-marker "$m"); done
  # One domain per boot: a faulted domain poisons later create_dom in the same
  # guest session (HOW-TO-RUN-ON-QEMU.md:131-132). $CAPSTONE_QEMU_LOCK, because
  # every suite shares one rootfs.ext2.
  flock "$CAPSTONE_QEMU_LOCK" python3 "$H/run-domain-smoke.py" \
    --share-dir "$share" --log-file "$log" \
    --guest-command "/mnt/host/capstone-test.user /mnt/host/$(basename "$dom")" \
    "${markers[@]}" >/dev/null 2>&1
  echo $?
}

echo
echo "==> CONTROL arm  (bound = header + REAL_SIZE(11) = 64) -- stock PHP"
rc=$(run_arm control "$OUT/crash008_control.dom" "Created domain ID = 0" "retval = 200")
if [ "$rc" != 0 ]; then
  echo "CONTROL DID NOT COMPLETE (rc=$rc). No verdict -- this is infrastructure, not a result." >&2
  exit 75
fi
echo "    PASS  completed, retval = 200 (survived) -- reproduces the corpus asan-stock OK row"

echo
echo "==> FAULT arm    (bound = header + 11 = 59) -- true request"
export CAPSTONE_DEBUG_PRINT=1
run_arm fault "$OUT/crash008_fault.dom" "__never__" >/dev/null
FL="$CAPSTONE_TMP_ROOT/crash008-fault.log"
oob=$(grep -a "Cap mem access OOB" "$FL" | tail -1)
hal=$(grep -a "domain halted by capability fault" "$FL" | tail -1)
if [ -z "$hal" ]; then
  echo "    FAIL  fault arm did not halt -- the overflow was NOT caught" >&2
  exit 1
fi
echo "    PASS  $hal"
[ -n "$oob" ] && echo "          $oob"

echo
echo "VERDICT: CAUGHT. Allocator unpatched (REAL_SIZE still rounds, cache still on);"
echo "         the 5-byte overflow is stopped by the capability bound."
