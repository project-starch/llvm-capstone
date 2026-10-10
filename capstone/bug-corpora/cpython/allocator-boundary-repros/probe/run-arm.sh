#!/bin/bash
# run-arm.sh <arm> [out]: boot once with that arm's image, run every case (ONLY= selects a subset).
#
# The arm is the heap the image links. Three things are checked before any case
# runs, each because skipping it has produced wrong rows in this lane:
#
#   the image hash   three rounds ran bounds-only under a sublet label, because
#                    the label came from an argument and nothing checked the
#                    binary
#   the interpreter  an image that cannot run a real workload gives meaningless
#                    defect results, and "no module named encodings" reads
#                    exactly like a silent mechanism
#   the discipline   an arm named sublet that reports mode=0 is measuring the
#                    spatial discipline under a sublet label, which is worse
#                    than no measurement
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../../.." && pwd)}
CORPUS=$(cd "$HERE/.." && pwd)
KIT=${KIT:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/cpython-arms}
# The share IS the guest's /mnt/host: capstone_vm has no --share on `up`, the
# directory is fixed by the VM state and the domain reads it directly.
SHARE=${SHARE:-$HOME/arms/cpython/shared/domain/share}
ST=${ST:-$HOME/arms/cpython/shared/domain/vm-state}
# --negative-control stages a trigger that reaches no defect, runs every selected
# case anyway, and exits 0 only if NONE of them is scored DETECTED: a detection
# with no defect executed is a fault that does not depend on the defect.
#
# Detect BEFORE filtering. The first version of this scanned "$@" further down,
# after the flag had already been removed, so NEGCTL was always 0 and the
# "control" ran the real triggers -- it reported a detection on case 01 and
# looked like the base arm faulting without a defect.
NEGCTL=0
args=()
for a in "$@"; do
  if [ "$a" = --negative-control ]; then NEGCTL=1; else args+=("$a"); fi
done
set -- "${args[@]+"${args[@]}"}"
arm=${1:?usage: run-arm.sh <arm> [out] [--negative-control]}
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=${2:-$KIT/results/$arm-$STAMP}
export PYTHONPATH=$REPO/capstone/runtime/host
# capstone_vm needs Python 3.11 or later. cli.py opens with
# "from __future__ import annotations", which rules out 3.6, and
# cli.py:190 calls hashlib.file_digest, which exists only from 3.11.
# This box has 3.6.9 on both the non-interactive and the login PATH and 3.9.12
# in miniconda's base, so BOTH of those fail -- and the first version of this
# check said 3.7, let 3.9 through, and produced four boots that died in one
# second each with the AttributeError buried in boot.log.
# /home/miniconda/miniconda3/envs/cheri-deps/bin/python3 is 3.11.16.
pyv=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)
pyok=$(python3 -c 'import hashlib; print(1 if hasattr(hashlib, "file_digest") else 0)' 2>/dev/null || echo 0)
if [ "$pyok" != 1 ]; then
  echo "REFUSING: python3 is ${pyv} and has no hashlib.file_digest; capstone_vm needs >= 3.11" >&2
  echo "  try PATH=/home/miniconda/miniconda3/envs/cheri-deps/bin:\$PATH" >&2
  exit 2
fi
CASE_TIMEOUT=${CASE_TIMEOUT:-120}
# CAP selects a capacity VARIANT of the arm's image, not another arm. The arm
# name still carries the oracle and still has to be in corpus.json; only the
# image and the sha the gate demands change. Four cases were excluded for
# capacity and two of them turned out to need a child process rather than a
# bigger heap, so the raised image must NOT silently become the image for
# everything: a bigger application heap moves when pymalloc returns an emptied
# arena, which is exactly what the cause-24 detections depend on.
#   base   the image the other rows were measured with
#   hicap  heap 192 MiB, PYM_ARENA 128 MiB, PYM_META 64 MiB, ARENA_COUNT 64,
#          LARGE_COUNT 65536 -- for the rows that cannot run at all otherwise
#   mod    default capacity plus _testinternalcapi and _testlimitedcapi, which
#          cases 15 and 23 die at import without. Capacity is deliberately NOT
#          raised here: the rows this image is compared against were measured at
#          the defaults, so the modules stay the only difference between them
#   mod96  the same modules with the application heap at 96 MiB instead of 48.
#          `mod` is unusable on the sublet arm: that arm revokes on every free
#          and the two extra builtin modules put the 48 MiB heap over DURING
#          IMPORT, so cases 15, 19, 23 and 31 all died in importlib -- and 19
#          and 31 were detections on the base image, so it lost more rows than
#          it recovered. Only the application arena is raised; PYM_ARENA_BYTES
#          and the rest stay at their defaults, to keep the confound small.
#   modL   the modules with LARGE_COUNT raised from 4096 to 65536 and nothing
#          else. `mod96` showed the application arena is NOT what runs out:
#          48 and 96 MiB fail identically, same traceback, same line. The
#          sublet port's own `large` table is the limit, and it is a separate
#          table from the arenas, so raising it leaves arena-return timing --
#          what the cause-24 detections depend on -- untouched. Verified at the
#          instruction level: the only difference in block-lifetimes.o is
#          lui a0, 0x1 becoming lui a0, 0x10.
CAP=${CAP:-base}
case $CAP in
  base)  IMGKEY=$arm ;;
  hicap) IMGKEY=$arm-hicap ;;
  mod)   IMGKEY=$arm-mod ;;
  mod96) IMGKEY=$arm-mod96 ;;
  modL)  IMGKEY=$arm-modL ;;
  *) echo "CAP must be base, hicap, mod, mod96 or modL, not '$CAP'" >&2; exit 2 ;;
esac
[ "$CAP" = base ] || OUT=${2:-$KIT/results/$arm-$CAP-$STAMP}
[ "$NEGCTL" = 0 ] || OUT=${2:-$KIT/results/$arm-negctl-$STAMP}
[ -z "${TRIGGER:-}" ] || OUT=${2:-$KIT/results/$arm-probe-$STAMP}

# ---- the arm must be one the corpus declares -----------------------------
DECL=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['required_arms']))" \
       "$CORPUS/corpus.json")
case " $DECL " in *" $arm "*) ;; *)
  echo "arm '$arm' is not in corpus.json required_arms: $DECL" >&2; exit 2 ;;
esac
[ "$arm" = cheribsd-revocation ] && {
  echo "cheribsd-revocation is not a domain arm -- use run-cheribsd.sh" >&2; exit 2; }

# Which arms revoke, and so need more than the 65536-node default. An earlier
# round at the default produced 44 cause-30 (INSUF_RESOURCES) rows out of 54,
# which is a resource ceiling and not a verdict about any defect.
case $arm in
  sublet|sysalloc-sublet|sublet-pymalloc|sublet-gc)
    export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-16777216}
    WANT_SUBLET=1 ;;
  *) WANT_SUBLET=0 ;;
esac

# ---- the image must be the one recorded for this arm ---------------------
IMG=$KIT/images/python-$IMGKEY.dom
INPUTS=$KIT/images/inputs.tsv
[ -f "$IMG" ]    || { echo "no image for $IMGKEY at $IMG" >&2; exit 2; }
[ -f "$INPUTS" ] || { echo "no $INPUTS" >&2; exit 2; }
want=$(awk -F'\t' -v a="$IMGKEY" '$1==a{print $3}' "$INPUTS")
have=$(sha256sum "$IMG" | cut -d' ' -f1)
[ -n "$want" ] || { echo "$IMGKEY is not in $INPUTS" >&2; exit 2; }
[ "$want" = "$have" ] || { echo "REFUSING: $IMG is not the image recorded for $IMGKEY" >&2
  echo "  recorded ${want:0:16}  present ${have:0:16}" >&2; exit 2; }

mkdir -p "$OUT"
# One run at a time: two would share the staging directory under the share.
exec 9> "$KIT/.run.lock"
flock -n 9 || { echo "REFUSING: another run holds $KIT/.run.lock" >&2; exit 3; }

# EVERY NAME USED LATER IS ASSIGNED HERE, before the controls read them. An
# earlier version put one of these below the positive control; under `set -u` a
# fresh bash died with "GUESTDOM: unbound variable" and the runner reported
# "POSITIVE CONTROL FAILED" four hours after the edit that caused it.
GUESTDOM=python-$arm.dom
GUESTENV=(-e PYTHONHOME=/mnt/host)
[ "$WANT_SUBLET" = 1 ] && GUESTENV+=(-e CPY_SUBLET_MODE=1)
mkdir -p "$SHARE"
cp "$IMG" "$SHARE/$GUESTDOM"
printf 'arm\t%s\ncapacity\t%s\nimage\t%s\nimage_sha256\t%s\nrev_nodes\t%s\nstarted_utc\t%s\n' \
  "$arm" "$CAP" "$IMGKEY" "$have" "${CAPSTONE_REV_NODES:-65536}" "$(date -u +%FT%TZ)" > "$OUT/run.meta"
cp "$INPUTS" "$OUT/inputs.tsv"
echo "arm=$arm capacity=$CAP image=${have:0:16} rev_nodes=${CAPSTONE_REV_NODES:-65536}"

vm() { python3 -m capstone_vm --state "$ST" "$@"; }
cleanup() { vm down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM HUP        # armed BEFORE the VM exists
# `up` needs --qemu --kernel --firmware --rootfs --share; `restart` needs
# nothing, because the state directory's config.json already records them. Using
# restart keeps those four host paths out of this script and guarantees the image
# boots on the same VM the earlier runs used.
[ -f "$ST/config.json" ] || {
  echo "no $ST/config.json -- this arm needs a VM state that has booted once." >&2
  echo "  Create it with: python3 -m capstone_vm --state $ST up \\" >&2
  echo "    --qemu ... --kernel ... --firmware ... --rootfs ... --share $SHARE" >&2
  exit 2; }
# The share recorded in the state must be the one we staged into, or the guest
# reads a different /mnt/host than the one holding this arm's image.
recshare=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['share'])" "$ST/config.json")
[ "$recshare" = "$SHARE" ] || {
  echo "REFUSING: the VM state's share is $recshare but this run staged into $SHARE" >&2
  exit 2; }
vm down >/dev/null 2>&1 || true
vm restart >"$OUT/boot.log" 2>&1 || { echo "boot failed, see $OUT/boot.log" >&2; exit 2; }

# ---- control 1: can this image run a real workload at all? --------------
echo "--- positive control: objects.py 8 3 0"
[ -f "$SHARE/objects.py" ] || { echo "  no $SHARE/objects.py to control with" >&2; exit 2; }
pcall=$(cd "$SHARE" && timeout 900 python3 -m capstone_vm --state "$ST" run --cwd /mnt/host \
        "${GUESTENV[@]}" "/mnt/host/$GUESTDOM" objects.py 8 3 0 2>&1)
pc=$(printf '%s' "$pcall" | tail -1)
echo "  $pc"
printf '%s\n' "$pcall" > "$OUT/positive-control.log"
case "$pc" in EXP-OK*) ;; *)
  echo "  POSITIVE CONTROL FAILED -- not running defects on an unqualified image" >&2
  exit 2 ;; esac

# ---- control 2: is this arm's discipline actually in force? -------------
pcmode=$(printf '%s' "$pcall" | grep -oE "CPY-SUBLET mode=[0-9]" | head -1)
[ -n "$pcmode" ] && echo "  $pcmode"
if [ "$WANT_SUBLET" = 1 ]; then
  case "$pcmode" in
    "CPY-SUBLET mode=1") echo "  sublet discipline confirmed" ;;
    "") echo "  REFUSING: the image printed no CPY-SUBLET mode line at all" >&2; exit 2 ;;
    *)  echo "  REFUSING: arm '$arm' asked for the sublet discipline, image reports $pcmode" >&2
        exit 2 ;;
  esac
fi
printf 'positive_control\t%s\ndiscipline\t%s\n' "$pc" "${pcmode:-none}" >> "$OUT/run.meta"

# ---- the cases ----------------------------------------------------------
# ONLY=01,10,18 runs just those case numbers. A re-run after raising the
# watchdog must not silently re-measure cases that already produced a verdict
# under the stricter setting, so the subset is explicit and recorded.
ONLY=${ONLY:-}
in_subset() {  # $1 = case directory name
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",${1%%_*},"*) return 0 ;; *) return 1 ;; esac
}
printf 'case\tarm\tverdict\trc\tcause\tlast\tpc\taddress\tuntagged_value\n' \
  > "$OUT/verdicts.tsv"
printf 'case_timeout\t%s\nonly\t%s\ntrigger_override\t%s\n' \
  "$CASE_TIMEOUT" "${ONLY:-all}" "${TRIGGER:-none}" >> "$OUT/run.meta"
n=0
for d in "$CORPUS"/[0-9][0-9]_*/; do
  c=$(basename "$d")
  if [ -n "$ONLY" ]; then
    num=${c%%_*}
    case ",$ONLY," in *",$num,"*) ;; *) continue ;; esac
  fi
  n=$((n+1))
  stage=$SHARE/case-$STAMP; rm -rf "$stage"; mkdir -p "$stage"
  # Every .py the case carries, not just the two known names: a case whose
  # upstream test imports a stdlib test package the guest's pruned Lib does not
  # have (test.test_ast, cases 02 and 13) vendors it beside the trigger, and a
  # staging step that copied only two files left those cases dying at import
  # while the row read as the arm staying quiet.
  cp "$d"/*.py "$stage/" 2>/dev/null
  # TRIGGER=<file> runs a different script from the case directory. It is for
  # diagnostics on a case whose trigger runs a whole upstream file: selecting
  # one test answers which defect a fault belongs to without editing the
  # trigger, which would re-measure the case. It is refused together with
  # --negative-control, because then neither name would say what ran.
  if [ -n "${TRIGGER:-}" ] && [ "$NEGCTL" = 1 ]; then
    echo "REFUSING: TRIGGER= and --negative-control both replace the trigger." >&2
    exit 2
  fi
  if [ -n "${TRIGGER:-}" ]; then
    [ -f "$d/$TRIGGER" ] || {
      printf '%s\t%s\tSTAGING-FAILED\t\t\t\t\t\t\n' "$c" "$arm" >> "$OUT/verdicts.tsv"
      printf '  %-56s STAGING-FAILED (no %s)\n' "${c:0:56}" "$TRIGGER"; continue; }
    cp "$d/$TRIGGER" "$stage/trigger.py"
  elif [ "$NEGCTL" = 1 ] && [ -f "$d/negative_control.py" ]; then
    # The strong control: the case's own allocation and free traffic with only
    # the offending access made valid. This is the one that qualifies a
    # detection -- the stub below only shows the arm does not fault on any
    # input, not that a near-miss is safe.
    cp "$d/negative_control.py" "$stage/trigger.py"
    NCKIND=variant
  elif [ "$NEGCTL" = 1 ]; then
    NCKIND=stub
    # Keep the case's other files -- an import failure would be a different
    # experiment -- and replace only the trigger, so the interpreter starts,
    # loads, and exits without performing the defect.
    printf '%s\n' \
      'import sys' \
      '# negative control: the real trigger is not run. Nothing here reaches a' \
      '# defect, so a DETECTED verdict would be a fault that does not depend on' \
      '# one.' \
      'print("NEGATIVE-CONTROL no defect performed")' \
      'sys.exit(0)' > "$stage/trigger.py"
  fi
  [ -f "$stage/trigger.py" ] || {
    printf '%s\t%s\tSTAGING-FAILED\t\t\t\t\t\t\n' "$c" "$arm" >> "$OUT/verdicts.tsv"
    printf '  %-56s STAGING-FAILED\n' "${c:0:56}"; continue; }
  before=$(wc -l < "$ST/qemu.log" 2>/dev/null || echo 0)
  out=$(cd "$stage" && timeout "$CASE_TIMEOUT" python3 -m capstone_vm --state "$ST" run \
          --cwd "/mnt/host/case-$STAMP" "${GUESTENV[@]}" \
          "/mnt/host/$GUESTDOM" trigger.py 2>&1); rc=$?
  printf '%s\n' "$out" > "$OUT/$c.log"
  tailq=$(tail -n +$((before+1)) "$ST/qemu.log" 2>/dev/null)
  # Keep the slice. qemu.log is opened "wb" on every restart, so whatever the
  # emulator printed about a fault is gone at the next boot unless the run
  # keeps it. That is how the operand value below came to be produced on every
  # run and retained on none.
  [ -n "$tailq" ] && printf '%s\n' "$tailq" > "$OUT/$c.qemu.log"
  # pc is reported for every cause and was being dropped. address is reported
  # for the bounds causes only: RISCV_EXCP_UNEXP_OP_TYPE is raised by
  # riscv_raise_exception(env, ..., GETPC()), which takes no address and sets
  # no badaddr, so a cause-24 line carries address=0x0 as an unset field and
  # not as the faulting value. Recording it anyway is what makes that legible.
  fpc=$(printf '%s' "$out$tailq" | grep -oE "pc ?= ?0x[0-9a-f]+" | tail -1 | grep -oE "0x[0-9a-f]+")
  faddr=$(printf '%s' "$out$tailq" | grep -oE "address ?= ?0x[0-9a-f]+" | tail -1 | grep -oE "0x[0-9a-f]+")
  # What the untagged base register actually held. The emulator prints it
  # unconditionally from _helper_access_with_cap (target/riscv/op_helper.c),
  # and it is the one field that tells a pointer which lost its tag from an
  # integer that never was one.
  fval=$(printf '%s' "$tailq" | grep -E "Cap mem access requires capability" \
         | tail -1 | grep -oE "value = [0-9a-f]+" | grep -oE "[0-9a-f]+$")
  # The launcher relays the fault on the run's own stdout with NO spaces around
  # "=", while qemu.log writes "cause = 24". Read both: a parser that matched
  # only the spaced form once scored a real cause=24 as NORUN.
  cause=$(printf '%s' "$out$tailq" | grep -oE "cause ?= ?[0-9]+" | tail -1 | grep -oE "[0-9]+$")
  last=$(printf '%s' "$out" | tail -2 | tr '\n' ' ')
  # Order matters. A fault is a measurement whatever else the log says; a
  # watchdog kill and an exhausted allocator are not measurements at all and
  # must never be read as "ran and the mechanism was silent".
  # The VM can die mid-run -- a host-side kill, an out-of-memory, a crash in
  # the monitor. Every case after that returns rc 1 with this message, and
  # rc 1 reads as SILENT, so the run would keep going and manufacture silence.
  if printf %s "$out" | grep -qE "VM is not running|use up first"; then
    printf '%s\t%s\tVM-GONE\t%s\t\t%s\t\t\t\n' "$c" "$arm" "$rc" \
      "the VM was not running; nothing after this point executed" >> "$OUT/verdicts.tsv"
    printf '  %-56s VM-GONE\n' "${c:0:56}"
    echo "ABORTING: the VM is not running, so nothing further can be measured." >&2
    printf 'aborted\tVM gone\n' >> "$OUT/run.meta"
    exit 4
  fi
  if [ -n "${cause:-}" ]; then
    v="DETECTED"
  elif [ "$rc" = 124 ]; then
    v="TIMEOUT"
  elif printf '%s' "$out" | grep -qE "cannot allocate application heap|MemoryError|error return without exception set"; then
    v="CAPACITY"
  elif [ "$rc" = 0 ] || [ "$rc" = 1 ]; then
    v="SILENT"
  else
    v="OTHER-rc$rc"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$arm" "$v" "$rc" \
    "${cause:-}" "$last" "${fpc:-}" "${faddr:-}" "${fval:-}" >> "$OUT/verdicts.tsv"
  [ "$NEGCTL" = 1 ] && printf '%s\t%s\n' "$c" "${NCKIND:-stub}" >> "$OUT/control-kind.tsv"
  printf '  %-54s %-9s rc=%-4s %s\n' "${c:0:54}" "$v" "$rc" "${cause:+cause=$cause}"
done
printf 'ended_utc\t%s\ncases\t%s\n' "$(date -u +%FT%TZ)" "$n" >> "$OUT/run.meta"
if [ "$NEGCTL" = 1 ]; then
  # A fault IS evidence that the case ran, so detections are judged first. The
  # first version asked for the marker first and reported a case that faulted
  # before printing as one that never started -- hiding the only outcome that
  # matters.
  bad=$(awk -F'\t' 'NR>1 && $3=="DETECTED"{print $1}' "$OUT/verdicts.tsv")
  # Cases that are silent AND left no marker did not run; they cannot support
  # "no detection" and they void the run. 29 of 32 once came back from a dead VM
  # and the summary read "0 of 32 detected, as required".
  nolog=0
  for d in "$CORPUS"/[0-9][0-9]_*/; do
    c=$(basename "$d"); in_subset "$c" || continue
    case " $bad " in *" $c "*) continue ;; esac
    grep -q "NEGATIVE-CONTROL no defect performed" "$OUT/$c.log" 2>/dev/null || {
      echo "  $c: silent and no marker, so this case did not run" >&2; nolog=$((nolog+1)); }
  done
  if [ -z "$bad" ] && [ "$nolog" != 0 ]; then
    printf 'negative_control\tVOID\nnegative_control_not_run\t%s\n' "$nolog" >> "$OUT/run.meta"
    echo "NEGATIVE CONTROL VOID: $nolog case(s) produced no evidence of running." >&2
    echo "  A control cannot pass on cases that never started." >&2
    exit 5
  fi
  # Not "n": that holds the number of cases this run staged and the summary
  # line below still needs it. Clobbering it printed "rows: 5 / 0".
  ndet=$(printf '%s' "$bad" | grep -c . || true)
  printf 'negative_control\t1\nnegative_control_detections\t%s\n' "$ndet" >> "$OUT/run.meta"
  if [ "$ndet" != 0 ]; then
    echo "NEGATIVE CONTROL FAILED: $ndet case(s) scored DETECTED with no defect performed:" >&2
    printf '  %s\n' $bad >&2
    echo "  those faults do not depend on the defect, so the detections they back are not evidence." >&2
    exit 1
  fi
  echo "negative control: 0 of $(($(wc -l < "$OUT/verdicts.tsv")-1)) cases detected, as required"
fi
echo "rows: $(($(wc -l < "$OUT/verdicts.tsv")-1)) / $n"
echo "out:  $OUT"
