#!/usr/bin/env bash
# Temporal axis: revoke-on-free in PHP 5.0.0's Zend allocator.
#
#   capstone/container/run.sh capstone/ports/php/zend-alloc/run-uaf.sh
#
# SYNTHETIC trigger, not a CRASH-nnn corpus case -- see uaf_domain.c.
#
# CONTROL FIRST, and it matters more here than on the spatial arm. At -O0 the
# alias is spilled and reloaded, so a caught use-after-free arrives as "tag gone"
# (cause 24), which is exactly what an unrelated spill of something else looks
# like. Only the control -- the same program with the single revoke removed --
# separates them (ALLOCATOR-CONTRACT.md §4).
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"

H="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu"
OUT="$CAPSTONE_TMP_ROOT/zend-uaf"
SRC="$HERE/uaf_domain.c"
rm -rf "$OUT"; mkdir -p "$OUT"

export DOMAIN_OPT_LEVEL=-O0
# The emulator sizes the revocation-node pool at startup and reuses no node.
export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-65536}

echo "==> building both arms (-O0)"
EXTRA_CLANG_FLAGS="-DZEND_TEMPORAL" \
  bash "$H/build-domain.sh" "$SRC" "$OUT/uaf_fault.dom" >/dev/null
EXTRA_CLANG_FLAGS="-DZEND_TEMPORAL -DZEND_NO_REVOKE" \
  bash "$H/build-domain.sh" "$SRC" "$OUT/uaf_control.dom" >/dev/null

# BUILD GATE. If the revoke is not in the binary it measures nothing, and a MISS
# is indistinguishable from a broken build
# (xlang/capstone/build-xlang-capstone.sh:87-96).
nf=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT/uaf_fault.dom"   2>/dev/null | grep -ci revoke)
nc=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$OUT/uaf_control.dom" 2>/dev/null | grep -ci revoke)
echo "    revoke instructions: fault=$nf control=$nc"
if [ "${nf:-0}" -eq 0 ]; then
  echo "BUILD-INVALID: no revoke in the fault arm" >&2; exit 3
fi
if [ "${nc:-1}" -ne 0 ]; then
  echo "BUILD-INVALID: control arm still contains a revoke; it is not a control" >&2; exit 3
fi

run_arm() {
  local arm=$1 dom=$2; shift 2
  local share="$OUT/share-$arm" log="$CAPSTONE_TMP_ROOT/uaf-$arm.log"
  rm -rf "$share"; mkdir -p "$share"
  cp "$dom" "$share/"
  bash "$H/build-capstone-test-user.sh" "$share/capstone-test.user" >/dev/null
  local markers=()
  for m in "$@"; do markers+=(--success-marker "$m"); done
  # One domain per boot: a faulted domain poisons later create_dom in the same
  # guest session. $CAPSTONE_QEMU_LOCK, because every suite shares rootfs.ext2.
  flock "$CAPSTONE_QEMU_LOCK" python3 "$H/run-domain-smoke.py" \
    --share-dir "$share" --log-file "$log" \
    --guest-command "/mnt/host/capstone-test.user /mnt/host/$(basename "$dom")" \
    "${markers[@]}" >/dev/null 2>&1
  echo $?
}

echo
echo "==> CONTROL arm  (-DZEND_NO_REVOKE) -- stock PHP lifetime behaviour"
rc=$(run_arm control "$OUT/uaf_control.dom" "Created domain ID = 0" "retval = 213")
if [ "$rc" != 0 ]; then
  echo "CONTROL DID NOT COMPLETE (rc=$rc). No verdict -- infrastructure, not a result." >&2
  exit 75
fi
echo "    PASS  completed, retval = 213 -- the use-after-free SURVIVED, as it does on stock PHP"

echo
echo "==> FAULT arm    (revoke-on-free)"
export CAPSTONE_DEBUG_PRINT=1
run_arm fault "$OUT/uaf_fault.dom" "__never__" >/dev/null
FL="$CAPSTONE_TMP_ROOT/uaf-fault.log"
hal=$(grep -a "domain halted by capability fault" "$FL" | tail -1)
diag=$(grep -aE "Cap mem access (requires capability|on revoked capability)" "$FL" | tail -1)
oob=$(grep -a "Cap mem access OOB" "$FL" | tail -1)

if [ -z "$hal" ]; then
  echo "    FAIL  fault arm did not halt -- the stale access was NOT caught" >&2; exit 1
fi
# Separating the axes: a bounds fault here would mean the trigger overflowed,
# which would make this a spatial result wearing a temporal label.
if [ -n "$oob" ]; then
  echo "    FAIL  halted on an OUT-OF-BOUNDS access, not a revoked one:" >&2
  echo "          $oob" >&2
  echo "          The trigger is supposed to stay inside its 32 bytes." >&2
  exit 1
fi
echo "    PASS  $hal"
[ -n "$diag" ] && echo "          $diag"

echo
echo "VERDICT: CAUGHT. The stale alias is dead at _efree, with no shadow memory,"
echo "         no quarantine and no redzone."
echo "NOTE:    cause 24 (tag gone) rather than 25 (revoked) is expected at -O0 --"
echo "         the alias is spilled and reloaded. That is why the control is required."
