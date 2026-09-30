#!/usr/bin/env bash
# Phase 3: real PHP corpus triggers, executed by the real engine, in a Capstone domain.
#
#   capstone/container/run.sh capstone/ports/php/engine/run-triggers.sh [CRASH-110|CRASH-073|both]
#
#   CRASH-110   var_dump(parse_url("file:///"))   ASAN: READ of size 1, url.c:132
#   CRASH-073   var_dump(parse_url('a:/'))        ASAN: READ of size 3, _estrndup from url.c:292
#
# var_dump is not linked: both over-reads are inside php_url_parse, before formatting.
#
# CONTROL FIRST, and the control decides whether there is a verdict at all. The control arm
# bounds every capability to REAL_SIZE(size) -- stock PHP's rounding -- so the over-read
# lands in the slack and the program must COMPLETE. That is the corpus's
# "crashes_on_pristine_build: false". If the control does not complete this exits 75 with NO
# verdict, per capstone/bug-corpora/README.md: at -O0 an unrelated spill or a broken build
# can fault too, so a fault alone proves nothing.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"

H="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu"
O="$CAPSTONE_TMP_ROOT/php-engine"
TMO=${CAPSTONE_TIMEOUT_MULTIPLIER:-10}
WHICH=${1:-both}

# Stage bits from rung_E_domain.c.
declare -A BIT=( [ENTERED]=1 [SINKED]=2 [STARTED]=4 [COMPILER]=8 [EXECUTOR]=16
                 [REGISTERED]=32 [EVALED]=64 [ISARRAY]=128 [BAILED]=256 [ERRORS]=512 )

build_arm() {   # $1 = out tag, $2... = extra flags
  local tag=$1; shift
  EXTRA_CF="$*" "$HERE/build-engine.sh" E >/dev/null 2>&1 || { echo "BUILD-FAIL $tag" >&2; return 1; }
  cp "$O/php_rungE.dom" "$O/trig_$tag.dom"
}

run_arm() {     # $1 = tag -> echoes "retval|cause|oob"
  # Separate statements on purpose: `local a=$1 b=$a` leaves b empty under set -u,
  # because local declares every name before the assignments expand.
  local tag=$1
  local share="$O/share-trig-$tag"
  local log="$CAPSTONE_TMP_ROOT/trig-$tag.log"
  rm -rf "$share"; mkdir -p "$share"; cp "$O/trig_$tag.dom" "$share/"
  bash "$H/build-capstone-test-user.sh" "$share/capstone-test.user" >/dev/null 2>&1
  local v=""
  # Retry: a lost guest boot is not a verdict. See FINDINGS.md on the unexplained
  # boot-phase flake -- it dies before "Domain requirement" and passes on a rerun.
  for i in 1 2 3; do
    flock "$CAPSTONE_QEMU_LOCK" python3 "$H/run-domain-smoke.py" \
      --share-dir "$share" --log-file "$log" --timeout-multiplier "$TMO" \
      --guest-command "/mnt/host/capstone-test.user /mnt/host/trig_$tag.dom" \
      --success-marker "retval" >/dev/null 2>&1
    v=$(grep -a "retval = " "$log" 2>/dev/null | tail -1 | sed 's/.*retval = //')
    [ -n "$v" ] && break
    grep -aq "cause = " "$log" 2>/dev/null && break     # a real fault, not a lost boot
  done
  local cause oob
  cause=$(grep -aoE "cause = [0-9]+" "$log" 2>/dev/null | tail -1 | sed 's/cause = //')
  oob=$(grep -a "Cap mem access OOB" "$log" 2>/dev/null | tail -1 | sed 's/.*OOB: //')
  printf '%s|%s|%s\n' "${v:-}" "${cause:-}" "${oob:-}"
}

flags_of() {    # $1 = retval -> prints set stage names
  local v=$1 out=""
  for n in ENTERED SINKED STARTED COMPILER EXECUTOR REGISTERED EVALED ISARRAY BAILED ERRORS; do
    (( v & BIT[$n] )) && out="$out $n"
  done
  echo "$out"
}

one_case() {    # $1 = case id, $2 = extra -D for the source select
  local id=$1
  local sel=$2
  local rc=0
  echo
  echo "################ $id"

  build_arm "${id}_control" $sel -DZEND_CAP_BOUNDS_REAL_SIZE || return 3
  build_arm "${id}_fault"   $sel                             || return 3

  # BUILD GATES. Without these a MISS is indistinguishable from a broken build.
  local n
  n=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$O/trig_${id}_fault.dom" 2>/dev/null | grep -ci 'shrink')
  if [ "${n:-0}" -eq 0 ]; then
    echo "BUILD-INVALID: no shrink in the fault arm; bounds are never narrowed" >&2; return 3
  fi
  if cmp -s "$O/trig_${id}_fault.dom" "$O/trig_${id}_control.dom"; then
    echo "BUILD-INVALID: arms byte-identical; -DZEND_CAP_BOUNDS_REAL_SIZE did nothing" >&2; return 3
  fi
  echo "    shrink instructions in fault arm: $n"

  # ---- CONTROL
  local c v cause oob
  c=$(run_arm "${id}_control"); v=${c%%|*}; cause=$(echo "$c" | cut -d'|' -f2)
  echo "==> CONTROL arm (bound = REAL_SIZE) -- stock PHP lifetime/rounding"
  if [ -z "$v" ]; then
    echo "    CONTROL DID NOT COMPLETE (cause=${cause:-none}). No verdict -- infrastructure, not a result." >&2
    return 75
  fi
  local isarr=$(( v & BIT[ISARRAY] ))
  echo "    retval=$v flags:$(flags_of "$v")"
  if [ "$isarr" -eq 0 ]; then
    echo "    CONTROL completed but parse_url did NOT return an array. No verdict." >&2
    return 75
  fi
  echo "    PASS  completed, parse_url returned an array -- reproduces the corpus pristine-OK row"

  # ---- FAULT
  local f
  f=$(run_arm "${id}_fault"); v=${f%%|*}; cause=$(echo "$f" | cut -d'|' -f2); oob=$(echo "$f" | cut -d'|' -f3)
  echo "==> FAULT arm   (bound = true request)"
  if [ -n "$cause" ] && [ -z "$v" ]; then
    echo "    CAUGHT  cause = $cause"
    [ -n "$oob" ] && echo "            $oob"
    echo "VERDICT: $id CAUGHT. url.c is byte-identical from the corpus; the over-read is"
    echo "         stopped by the capability bound on the zval string."
  elif [ -n "$v" ]; then
    echo "    retval=$v flags:$(flags_of "$v")"
    echo "VERDICT: $id MISSED -- the fault arm completed. The read is inside the bound."
    rc=1
  else
    echo "    no retval and no fault recorded -- lost boot after retries. No verdict." >&2
    rc=75
  fi
  return $rc
}

worst=0
case "$WHICH" in
  CRASH-110) one_case CRASH-110 ""                  ; worst=$? ;;
  CRASH-073) one_case CRASH-073 "-DRUNG_E_CRASH073" ; worst=$? ;;
  both)
    one_case CRASH-110 ""                  ; a=$?
    one_case CRASH-073 "-DRUNG_E_CRASH073" ; b=$?
    worst=$(( a > b ? a : b )) ;;
  *) echo "usage: $0 [CRASH-110|CRASH-073|both]" >&2; exit 2 ;;
esac
echo
exit $worst
