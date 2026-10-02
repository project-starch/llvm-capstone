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

build_arm() {   # $1 = out tag, $2 = allocator mode (hier|seam), $3... = extra flags
  local tag=$1; local mode=$2; shift 2
  PHP_CAP_ALLOC_MODE="$mode" EXTRA_CF="$*" "$HERE/build-engine.sh" E >/dev/null 2>&1 \
    || { echo "BUILD-FAIL $tag" >&2; return 1; }
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
  # Retry: a lost guest boot is not a verdict. See FINDINGS.md on the boot-phase flake -- the
  # guest wedges mid-boot with QEMU still spinning, and a rerun passes.
  #
  # The LOGIN timeout is capped separately from the run timeout, and the distinction matters: the
  # multiplier exists because the DOMAIN is slow (1.4 MB to load and execute under a capability
  # monitor), while booting Linux to a login prompt takes the same ~30 s whatever we are about to
  # run. Left at the default 120 * multiplier, every flaked boot cost 24 MINUTES of a 3-retry
  # budget before being retried. Overridable, since a loaded machine does boot slower.
  local login=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-300}
  for i in 1 2 3; do
    CAPSTONE_QEMU_LOGIN_TIMEOUT="$login" \
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

# WHERE DID IT FAULT? A fault in the fault arm is not a catch unless it is the defect under test.
# Scoring "the fault arm faulted" alone reported a CAUGHT for CRASH-110/073 whose actual fault was
# _estrndup's NUL write at zend_alloc.c:404 -- an allocator bug of ours, in the right arm for the
# wrong reason. Every verdict now prints the faulting source location so that cannot pass silently.
fault_site() {  # $1 = log -> "file:line" of the faulting pc, or empty
  local log=$1 pc base elf
  pc=$(grep -aoE "pc = 101c[0-9a-f]+" "$log" 2>/dev/null | tail -1 | sed 's/pc = //')
  base=$(grep -aoE "pcc_base = 101c[0-9a-f]+" "$log" 2>/dev/null | tail -1 | sed 's/pcc_base = //')
  [ -z "$pc" ] && return 0
  [ -z "$base" ] && base=101c00000
  elf=$(( 0x10000 + (0x$pc - 0x$base) ))
  "$CAPSTONE_LLVM_BIN/llvm-symbolizer" --obj="$O/trig_${2}.dom" "$elf" 2>/dev/null \
    | sed -n 2p | sed 's|.*/||'
}

# CASES WHOSE OVER-READ IS PERFORMED BY A SHARED COPY PRIMITIVE.
#
# CRASH-073 is `estrndup(s, (ue-s))` at url.c:292 with the length three bytes too long, and ASAN
# reports it with frame #0 in MEMCPY, frame #1 in _estrndup, frame #2 in php_url_parse -- because
# the over-read is performed by the copy, not by url.c. So for these cases a fault inside our memcpy
# is the CORRECT shape of the catch, and the url.c-only gate below would reject the real thing.
#
# It cannot simply be loosened: a fault in memcpy is also exactly what a bound the PORT got wrong
# looks like, which is how _erealloc's narrow-pointer bug once scored as a catch. The faulting pc
# cannot separate them, so the attribution is taken at the ALLOCATOR BOUNDARY instead: a third,
# diagnostic arm compares the length the caller asked for against what the source capability
# authorises, and halts with both numbers if the request overruns. A caller asking for more than its
# source holds is the defect whoever performs the load.
#
# The audit arm is built only to answer this question and is never the arm that is scored.
declare -A COPYCASE=( [CRASH-073]=estrndup )

# The copy primitives, by source file: a fault here is attributable but never self-attributing.
is_copy_site() { case "$1" in beebs_freestanding_string.c:*) return 0 ;; *) return 1 ;; esac; }

attribute_overread() {  # $1 = case id, $2 = source-select flags -> echoes "who|requested|avail|overrun"
  local id=$1 sel=$2 tag="${id}_audit" bad v
  build_arm "$tag" hier $sel -DZEND_CAP_ESTRNDUP_AUDIT >/dev/null 2>&1 || return 1
  run_arm "$tag" >/dev/null 2>&1
  bad=$(grep -aoE "badaddr = 0x[0-9a-f]+" "$CAPSTONE_TMP_ROOT/trig-$tag.log" 2>/dev/null \
        | tail -1 | sed 's/.*0x//')
  [ -z "$bad" ] && return 1
  # php_fault_report encodes its argument in badaddr, offset by the 64-byte deliberate over-store.
  v=$(( 0x$bad - 64 ))
  [ $(( (v >> 56) & 0xFF )) -eq 227 ] || return 1          # 0xE3 marks an audit report
  printf '%s|%s|%s|%s\n' $(( (v >> 48) & 0xFF )) $(( (v >> 24) & 0xFFFF )) \
                          $(( (v >> 8) & 0xFFFF )) $(( v & 0xFF ))
  return 0
}

flags_of() {    # $1 = retval -> prints set stage names
  local v=$1 out=""
  for n in ENTERED SINKED STARTED COMPILER EXECUTOR REGISTERED EVALED ISARRAY BAILED ERRORS; do
    (( v & BIT[$n] )) && out="$out $n"
  done
  echo "$out"
}

# AXIS of each case, because a matched pair is only meaningful when the CONTROL varies the thing
# that masks the bug:
#   heap    -- a heap over-read. Allocator rounding masks it, so -DZEND_CAP_BOUNDS_REAL_SIZE is
#              the right control and the pair is valid.
#   temporal-- a use-after-free. Nothing about BOUNDS masks or reveals it; it needs revoke-on-free
#              (-DZEND_TEMPORAL / -DZEND_NO_REVOKE). On the spatial axis the correct report is
#              NOT APPLICABLE, not MISSED -- calling it a miss would be a false negative claim.
#   global  -- an overflow of a .rodata/.bss table. The compiler bounds globals identically in
#              both arms, so the allocator control cannot mask it and there is no pair to run.
#              The control is the corpus's own crashes_on_pristine_build=False row.
axis_of() {
  case "$1" in
    CRASH-110|CRASH-073) echo heap ;;
    CRASH-003|CRASH-004|CRASH-010|CRASH-067) echo temporal ;;
    CRASH-005) echo global ;;
    *) echo heap ;;
  esac
}

# THE TEMPORAL AXIS NEEDS A CONTROL THAT LEAVES REVOCATION ON.
#
# The per-case control is -DZEND_NO_REVOKE, which turns revocation OFF. That is the right control
# for "does the bug survive stock PHP lifetimes", and it is useless for the question that actually
# decides the verdict: does revoke-on-free survive NORMAL OPERATION? If it does not, every temporal
# case faults in the fault arm and completes in the control, and the suite reports a clean sweep of
# catches having caught nothing.
#
# That is not hypothetical -- it happened. Four use-after-free cases reported CAUGHT, all faulting at
# the same line inside the allocator, and a benign script with revocation on faulted identically
# (zend_find matching a free slot on an untagged pointer). The per-case pair could not see it,
# because the thing that was broken was removed by its own control.
#
# So: one benign arm, revocation ON, spatial over-read masked by REAL_SIZE exactly as stock PHP
# masks it, run once before any temporal case is scored. It must COMPLETE.
REVSANE=""
temporal_sanity() {
  [ -n "$REVSANE" ] && return "$REVSANE"
  echo "==> REVOCATION SANITY (benign workload, revoke-on-free ON)"
  if ! build_arm revsane seam -DZEND_TEMPORAL -DZEND_CAP_BOUNDS_REAL_SIZE; then
    echo "    BUILD-FAIL" >&2; REVSANE=3; return 3
  fi
  local r v cause
  r=$(run_arm revsane); v=${r%%|*}; cause=$(echo "$r" | cut -d'|' -f2)
  if [ -z "$v" ]; then
    echo "    FAILED (cause=${cause:-none}) at $(fault_site "$CAPSTONE_TMP_ROOT/trig-revsane.log" revsane)" >&2
    echo "    Revocation does not survive a workload with no lifetime error in it. Every temporal" >&2
    echo "    verdict below would be this fault, not a catch. NO VERDICT on the temporal axis." >&2
    REVSANE=75; return 75
  fi
  echo "    PASS  completed (retval=$v) -- revocation is sound on a clean workload"
  REVSANE=0; return 0
}

one_case() {    # $1 = case id, $2 = extra -D for the source select
  local id=$1
  local sel=$2
  local rc=0
  local axis
  axis=$(axis_of "$id")
  if [ "$axis" = temporal ]; then
    echo
    echo "################ $id   (TEMPORAL axis: revoke-on-free)"
    temporal_sanity || return $?
    # SEAM mode, not hier: in the hierarchy PHP's AG(cache) absorbs most frees, so an efree never
    # reaches the arena and revoke-on-free cannot fire per object. Collapsing the seam puts the
    # capability allocator in PHP's place, so every emalloc/efree pair is seen. This is the one
    # axis where the two allocator modes are not interchangeable.
    build_arm "${id}_control" seam $sel -DZEND_TEMPORAL -DZEND_NO_REVOKE || return 3
    build_arm "${id}_fault"   seam $sel -DZEND_TEMPORAL                  || return 3
    local n
    n=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$O/trig_${id}_fault.dom" 2>/dev/null | grep -ci 'revoke')
    if [ "${n:-0}" -eq 0 ]; then
      echo "BUILD-INVALID: no revoke instruction in the fault arm; lifetimes are never enforced" >&2
      return 3
    fi
    if cmp -s "$O/trig_${id}_fault.dom" "$O/trig_${id}_control.dom"; then
      echo "BUILD-INVALID: arms byte-identical; -DZEND_NO_REVOKE did nothing" >&2; return 3
    fi
    echo "    revoke instructions in fault arm: $n"
    local c v cause oob
    c=$(run_arm "${id}_control"); v=${c%%|*}; cause=$(echo "$c" | cut -d'|' -f2)
    echo "==> CONTROL arm (-DZEND_NO_REVOKE) -- stock PHP lifetime behaviour"
    if [ -z "$v" ]; then
      echo "    CONTROL DID NOT COMPLETE (cause=${cause:-none}). No verdict -- infrastructure." >&2
      return 75
    fi
    echo "    PASS  completed (retval=$v) -- the use-after-free SURVIVED, as on stock PHP"
    local f
    f=$(run_arm "${id}_fault"); v=${f%%|*}; cause=$(echo "$f" | cut -d'|' -f2); oob=$(echo "$f" | cut -d'|' -f3)
    echo "==> FAULT arm   (revoke-on-free)"
    if [ -n "$cause" ] && [ -z "$v" ]; then
      echo "    CAUGHT  cause = $cause"
      [ -n "$oob" ] && echo "            $oob"
      local tsite
      tsite=$(fault_site "$CAPSTONE_TMP_ROOT/trig-${id}_fault.log" "${id}_fault")
      echo "            faulting site: ${tsite:-<unresolved>}"
      # A use-after-free faults where the stale reference is USED -- in Zend, which is where ASAN
      # reports all four (zend_assign_to_variable, zend_assign_to_variable_reference,
      # _zval_ptr_dtor). A fault inside the ALLOCATOR is the allocator tripping over its own
      # revocation, which is what the sanity arm exists to rule out; refuse it here as well, since
      # the two checks fail independently.
      case "$tsite" in
        zend_capstone_alloc.h:*|php_capstone_*)
          echo "SUSPECT: $id -- the fault is inside the ALLOCATOR, not at a use of the stale" >&2
          echo "         reference. That is a port artefact, not a catch. NO VERDICT." >&2
          return 75 ;;
      esac
      echo "VERDICT: $id CAUGHT on the temporal axis. The stale reference is dead at its USE."
      return 0
    elif [ -n "$v" ]; then
      echo "    retval=$v flags:$(flags_of "$v")"
      echo "VERDICT: $id MISSED -- the fault arm completed despite revoke-on-free."
      return 1
    fi
    echo "    no retval and no fault after retries. No verdict." >&2
    return 75
  fi
  echo
  echo "################ $id"

  build_arm "${id}_control" hier $sel -DZEND_CAP_BOUNDS_REAL_SIZE || return 3
  build_arm "${id}_fault"   hier $sel                             || return 3

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
  c=$(run_arm "${id}_control"); v=${c%%|*}; cause=$(echo "$c" | cut -d'|' -f2); oob=$(echo "$c" | cut -d'|' -f3)
  echo "==> CONTROL arm (bound = REAL_SIZE) -- stock PHP lifetime/rounding"
  if [ -z "$v" ] && [ "$axis" = global ]; then
    # EXPECTED for a global overflow: the allocator control cannot mask a .rodata/.bss bound, so
    # both arms fault. That is the bug, not infrastructure.
    echo "    control also faults (cause = ${cause:-none}) -- EXPECTED on the global axis"
    [ -n "$oob" ] && echo "            $oob"
    echo "VERDICT: $id CAUGHT (no pair). The overflowed object is a GLOBAL, bounded by the"
    echo "         compiler in both arms, so allocator rounding cannot mask it and there is no"
    echo "         control to vary. Control is the corpus row: stock PHP survives this input."
    return 0
  fi
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
  echo "    PASS  completed -- reproduces the corpus crashes_on_pristine_build=False row"

  # ---- FAULT
  local f
  f=$(run_arm "${id}_fault"); v=${f%%|*}; cause=$(echo "$f" | cut -d'|' -f2); oob=$(echo "$f" | cut -d'|' -f3)
  echo "==> FAULT arm   (bound = true request)"
  if [ -n "$cause" ] && [ -z "$v" ]; then
    local site
    site=$(fault_site "$CAPSTONE_TMP_ROOT/trig-${id}_fault.log" "${id}_fault")
    echo "    CAUGHT  cause = $cause"
    [ -n "$oob" ] && echo "            $oob"
    echo "            faulting site: ${site:-<unresolved>}"
    case "$site" in
      url.c:*) : ;;   # the defect under test, reading through its own frame
      *)
        # A copy primitive, for a case whose over-read ASAN also reports inside memcpy: attributable
        # on evidence, never on the pc alone.
        if [ -n "${COPYCASE[$id]:-}" ] && is_copy_site "$site"; then
          echo "            (the copy inside ${COPYCASE[$id]} performs the read, as ASAN reports"
          echo "             it; attributing at the allocator boundary)"
          local a who req avail over
          a=$(attribute_overread "$id" "$sel") || {
            echo "SUSPECT: $id -- fault is in the copy primitive and the audit arm did not report an" >&2
            echo "         over-length request. Unattributed, so NO VERDICT." >&2
            return 75; }
          who=${a%%|*}; req=$(echo "$a" | cut -d'|' -f2)
          avail=$(echo "$a" | cut -d'|' -f3); over=$(echo "$a" | cut -d'|' -f4)
          if [ "${over:-0}" -eq 0 ]; then
            echo "SUSPECT: $id -- audit reports no overrun. NO VERDICT." >&2; return 75
          fi
          echo "            ATTRIBUTED: _estrndup asked for $req bytes from a source authorising"
          echo "                        $avail -- over by $over (audit who=$who)"
          echo "VERDICT: $id CAUGHT. url.c is byte-identical from the corpus; its length arithmetic"
          echo "         asks for more than the source holds, and the capability stops the copy at"
          echo "         the first byte past the end -- the same frame ASAN reports (#0 memcpy,"
          echo "         #1 _estrndup, #2 php_url_parse)."
          return 0
        fi
        echo "SUSPECT: $id -- the fault arm faulted, but NOT in url.c. That is an artefact of" >&2
        echo "         this port, not a corpus catch. Treating as NO VERDICT." >&2
        return 75 ;;
    esac
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

# The reachable corpus cases: those needing no ext/standard function, since the whole Zend engine
# is linked but only ext/standard/url.c is. The four use-after-free cases are pure language;
# CRASH-005 is a PARSE ERROR (`a(1`) whose ASAN class is global-buffer-overflow, so it exercises
# the scanner/parser rather than the heap.
#
# AXIS MATTERS. -DZEND_CAP_BOUNDS_REAL_SIZE vs nothing is the SPATIAL pair, which is what the
# bounds arm varies. A use-after-free is a TEMPORAL defect and needs revoke-on-free
# (-DZEND_TEMPORAL, controlled by -DZEND_NO_REVOKE); the spatial pair cannot catch one, and
# reporting a spatial MISS on a UAF would be meaningless. `lang` runs the reachable cases on the
# spatial axis only, which is honest about what it can show and is the cheap first measurement.
worst=0
case "$WHICH" in
  CRASH-110) one_case CRASH-110 ""                  ; worst=$? ;;
  CRASH-073) one_case CRASH-073 "-DRUNG_E_CRASH073" ; worst=$? ;;
  both)
    one_case CRASH-110 ""                  ; a=$?
    one_case CRASH-073 "-DRUNG_E_CRASH073" ; b=$?
    worst=$(( a > b ? a : b )) ;;
  lang)
    for c in CRASH-003 CRASH-004 CRASH-010 CRASH-067 CRASH-005; do
      one_case "$c" "-DRUNG_E_${c/CRASH-/CRASH}" ; r=$?
      [ "$r" -gt "$worst" ] && worst=$r
    done ;;
  *) echo "usage: $0 [CRASH-110|CRASH-073|both|lang]" >&2; exit 2 ;;
esac
echo
exit $worst
