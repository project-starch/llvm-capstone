#!/bin/bash
# run-arm.sh <arm> [out]: boot once with that arm's image, run all 21 cases.
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
arm=${1:?usage: run-arm.sh <arm> [out]}
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=${2:-$KIT/results/$arm-$STAMP}
export PYTHONPATH=$REPO/capstone/runtime/host
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
CAP=${CAP:-base}
case $CAP in
  base)  IMGKEY=$arm ;;
  hicap) IMGKEY=$arm-hicap ;;
  *) echo "CAP must be base or hicap, not '$CAP'" >&2; exit 2 ;;
esac
[ "$CAP" = base ] || OUT=${2:-$KIT/results/$arm-$CAP-$STAMP}

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
printf 'case\tarm\tverdict\trc\tcause\tlast\n' > "$OUT/verdicts.tsv"
printf 'case_timeout\t%s\nonly\t%s\n' "$CASE_TIMEOUT" "${ONLY:-all}" >> "$OUT/run.meta"
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
  [ -f "$stage/trigger.py" ] || {
    printf '%s\t%s\tSTAGING-FAILED\t\t\t\n' "$c" "$arm" >> "$OUT/verdicts.tsv"
    printf '  %-56s STAGING-FAILED\n' "${c:0:56}"; continue; }
  before=$(wc -l < "$ST/qemu.log" 2>/dev/null || echo 0)
  out=$(cd "$stage" && timeout "$CASE_TIMEOUT" python3 -m capstone_vm --state "$ST" run \
          --cwd "/mnt/host/case-$STAMP" "${GUESTENV[@]}" \
          "/mnt/host/$GUESTDOM" trigger.py 2>&1); rc=$?
  printf '%s\n' "$out" > "$OUT/$c.log"
  tailq=$(tail -n +$((before+1)) "$ST/qemu.log" 2>/dev/null)
  # The launcher relays the fault on the run's own stdout with NO spaces around
  # "=", while qemu.log writes "cause = 24". Read both: a parser that matched
  # only the spaced form once scored a real cause=24 as NORUN.
  cause=$(printf '%s' "$out$tailq" | grep -oE "cause ?= ?[0-9]+" | tail -1 | grep -oE "[0-9]+$")
  last=$(printf '%s' "$out" | tail -2 | tr '\n' ' ')
  # Order matters. A fault is a measurement whatever else the log says; a
  # watchdog kill and an exhausted allocator are not measurements at all and
  # must never be read as "ran and the mechanism was silent".
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$arm" "$v" "$rc" "${cause:-}" "$last" >> "$OUT/verdicts.tsv"
  printf '  %-54s %-9s rc=%-4s %s\n' "${c:0:54}" "$v" "$rc" "${cause:+cause=$cause}"
done
printf 'ended_utc\t%s\ncases\t%s\n' "$(date -u +%FT%TZ)" "$n" >> "$OUT/run.meta"
echo "rows: $(($(wc -l < "$OUT/verdicts.tsv")-1)) / $n"
echo "out:  $OUT"
