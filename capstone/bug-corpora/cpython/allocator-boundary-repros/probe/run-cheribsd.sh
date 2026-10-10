#!/bin/bash
# run-cheribsd.sh [out]: the cheribsd-revocation arm.
#
# A different shape from run-arm.sh, because this arm is not a domain image: it
# is one purecap CPython on CheriBSD, and the protection is a RUNTIME sysctl
# rather than a linked-in heap. So what has to be pinned down is the sysctl, not
# an image hash.
#
# THE SYSCTL IS READ BACK INSIDE THE GUEST BEFORE ANY CASE RUNS, and the whole
# subtree is recorded. A round in this lane logged only
# runtime_revocation_default=1 and left every other knob unrecorded; that
# round's temporal semantics are now permanently uninterpretable, because
# runtime_revocation_every_free_default turned out to matter and six scripts on
# this machine write it. Recording one knob is not recording the configuration.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd "$HERE/.." && pwd)
# The interpreter ports/cpython/app/cheribsd/build.sh produced. No default: the
# 2026-10-08 and -09 runs defaulted to a path whose outer and inner binaries
# differed, and the guest-copy check below refused every run until it was named.
PY=${CHERI_PYTHON:?set CHERI_PYTHON to the python build.sh produced}
GUEST=${GUEST:-/root/cpython}
# PYTHONHOME is the pyhome DIRECTORY INSIDE the install, not the install.
# With the wrong one every case dies before any Python runs, with
# "No module named encodings" -- indistinguishable from a silent arm.
PYHOME=${PYHOME:-$GUEST/pyhome}
# Both built from capstone/bug-corpora/tools/cheribsd/sicode.c and
# capstone/bug-corpora/postgres/mmgr-repros/results/20261008-cheribsd/quarantine-probe.c.
SICODE=${SICODE:-/root/sicode.so}
SELFTEST=${SELFTEST:-/root/sicode-selftest}
QPROBE=${QPROBE:-/root/quarantine-probe.so}
GUEST_PORT=${GUEST_PORT:-10086}
# --negative-control: see run-arm.sh. Detect BEFORE filtering -- the first
# version detected after the flag had been removed, so the control ran the real
# triggers.
NEGCTL=0
args=()
for a in "$@"; do
  if [ "$a" = --negative-control ]; then NEGCTL=1; else args+=("$a"); fi
done
set -- "${args[@]+"${args[@]}"}"
OUT=${1:-$HOME/arms/cpython/cheribsd/results/boundary$( [ "$NEGCTL" = 1 ] && echo -negctl )-$(date -u +%Y%m%d-%H%M%S)}
BUDGET=${CASE_BUDGET:-120}
# ONLY=01,10,18 runs just those case numbers. It gates BOTH the staging loop
# and the run loop: a subset that stages 32 cases but runs 3 leaves 29 stale
# directories in the guest that the next full run would silently reuse.
ONLY=${ONLY:-}
in_subset() {  # $1 = case directory name, e.g. 07_gh140594
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",${1%%_*},"*) return 0 ;; *) return 1 ;; esac
}
K="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=120 ${GUEST_KEY:+-i $GUEST_KEY}"
G() { ssh $K -p "$GUEST_PORT" root@localhost "$@" 2>/dev/null; }
P() { scp -q $K -P "$GUEST_PORT" "$1" root@localhost:"$2" 2>/dev/null; }

[[ -f $PY ]] || { echo "no purecap python at $PY" >&2; exit 2; }
mkdir -p "$OUT"

# ---- one connection attempt, no retry loop ------------------------------
# Retry loops from two lanes at once once drove this guest's sshd into
# MaxStartups throttling and dropped 47 connections over 40 minutes. If the
# guest is not up, that is a fact to report, not a thing to hammer.
G true || { echo "guest on port $GUEST_PORT is not reachable; bring it up first" >&2; exit 2; }

# ---- the configuration, whole, before anything runs ---------------------
G 'sysctl security.cheri 2>/dev/null' > "$OUT/sysctl-before.txt"
grep -q . "$OUT/sysctl-before.txt" || {
  echo "REFUSING: cannot read security.cheri -- a run with no configuration record is not worth doing" >&2
  exit 2; }
rev=$(awk -F': ' '/runtime_revocation_default/{print $2}' "$OUT/sysctl-before.txt")
[[ $rev == 1 ]] || {
  echo "REFUSING: security.cheri.runtime_revocation_default is '$rev', not 1." >&2
  echo "  This arm IS revocation. Scoring it with revocation off would record the" >&2
  echo "  platform's default as a result about the defects." >&2
  exit 2; }
ef=$(awk -F': ' '/runtime_revocation_every_free_default/{print $2}' "$OUT/sysctl-before.txt")
printf 'arm\tcheribsd-revocation\npython_sha256\t%s\nrevocation_default\t%s\nevery_free_default\t%s\nstarted_utc\t%s\n' \
  "$(sha256sum "$PY" | cut -d' ' -f1)" "$rev" "${ef:-unrecorded}" "$(date -u +%FT%TZ)" > "$OUT/run.meta"
echo "revocation_default=$rev every_free_default=${ef:-unrecorded}"

# ---- control: can this interpreter run a real workload at all? ---------
# Without this, a wrong PYTHONHOME or a missing sicode.so turns every case into
# a fabricated silence. objects.py is the same workload the Capstone arms use.
G "test -x $GUEST/python" || { echo "no $GUEST/python in the guest" >&2; exit 2; }
# The binary that RUNS is the guest's copy; $PY is only what run.meta records.
# Nothing kept the two in step, so a stale guest copy would have produced
# results attributed to a build that never ran them. Compare them.
hostsha=$(sha256sum "$PY" | cut -d' ' -f1)
guestsha=$(G "sha256sum $GUEST/python 2>/dev/null | cut -d' ' -f1")
[[ -n $guestsha ]] || { echo "could not hash $GUEST/python in the guest" >&2; exit 2; }
[[ $hostsha == "$guestsha" ]] || {
  echo "REFUSING: the guest is running a different binary from the one recorded." >&2
  echo "  host  $PY: ${hostsha:0:16}" >&2
  echo "  guest $GUEST/python: ${guestsha:0:16}" >&2
  echo "  push the intended binary into the guest, or point PY at the one that is there." >&2
  exit 2; }
echo "binary: ${hostsha:0:16} (host and guest agree)"
G "test -f $SICODE" || { echo "no $SICODE in the guest -- push it before running" >&2; exit 2; }
G "test -f $QPROBE" || { echo "no $QPROBE in the guest -- push it before running" >&2; exit 2; }
pcout=$(G "cd $GUEST && env PYTHONHOME=$PYHOME LD_PRELOAD=$SICODE:$QPROBE timeout 300 ./python objects.py 8 3 0 2>&1")
pc=$(printf '%s\n' "$pcout" | grep -a '^EXP-OK' | tail -1)
# The quarantine probe reports from a destructor; a workload that ends normally
# without its line means the preload did not take, and every case would lack it.
qpc=$(printf '%s\n' "$pcout" | grep -a '^QUARANTINE shadow=mapped' | tail -1)
[[ -n $qpc ]] || { echo "REFUSING: the quarantine probe did not report on the workload" >&2
  printf '%s\n' "$pcout" | tail -5 >&2; exit 2; }
echo "quarantine probe: $qpc"
echo "positive control: ${pc:-<no output>}"
case "$pc" in
  EXP-OK*) ;;
  *) echo "REFUSING: the interpreter is not qualified on this guest." >&2
     echo "  Got: ${pc:-<no output>}" >&2
     echo "  A run now would record one silence per case, and each would be the harness, not the arm." >&2
     exit 2 ;;
esac
# si_code reporting is what makes this arm informative at all: without it a
# fault is just "it crashed", and BOUNDS, TAG and PERM are indistinguishable.
# The reporter's own self-test is the control: a read past a 16-byte allocation
# must report PROT_CHERI_BOUNDS, and a read through a freed pointer after a
# forced revocation pass must report PROT_CHERI_TAG. The second is also the
# positive control for revocation itself on this guest.
G "test -x $SELFTEST" || { echo "no $SELFTEST in the guest -- push it before running" >&2; exit 2; }
# The self-test has no handler of its own: the line comes from the preload,
# loaded exactly as it is in front of the interpreter.
sb=$(G "env LD_PRELOAD=$SICODE:$QPROBE timeout 60 $SELFTEST bounds 2>&1" | grep -a -o 'si_code=[0-9]* ([A-Z_]*)' | head -1)
sr=$(G "env LD_PRELOAD=$SICODE:$QPROBE timeout 60 $SELFTEST revoked 2>&1" | grep -a -o 'si_code=[0-9]* ([A-Z_]*)' | head -1)
[[ $sb == "si_code=1 (PROT_CHERI_BOUNDS)" && $sr == "si_code=2 (PROT_CHERI_TAG)" ]] || {
  echo "REFUSING: the si_code self-test gave bounds '${sb:-nothing}', revoked '${sr:-nothing}'" >&2
  echo "  expected si_code=1 (PROT_CHERI_BOUNDS) and si_code=2 (PROT_CHERI_TAG)." >&2
  exit 2; }
sc="bounds $sb, revoked $sr"
echo "si_code control:  $sc"
printf 'positive_control\t%s\nsicode_control\t%s\nquarantine_control\t%s\n' "$pc" "$sc" "$qpc" >> "$OUT/run.meta"

# ---- stage: one directory per case, named as the corpus names it --------
G "rm -rf /root/boundary && mkdir -p /root/boundary"
n=0
for d in "$CORPUS"/[0-9][0-9]_*/; do
  c=$(basename "$d")
  in_subset "$c" || continue
  G "mkdir -p /root/boundary/$c"
  # Every .py the case carries -- see the note in run-arm.sh's staging loop.
  ( cd "$d" && tar -cf - ./*.py ) | G "tar -C /root/boundary/$c -xf -"
  if [ "$NEGCTL" = 1 ] && [ -f "$d/negative_control.py" ]; then
    # The strong control: the case's own allocation and free traffic with only
    # the offending access made valid. It is the one that qualifies a
    # detection, and this arm could not run it until now -- only the stub below
    # was implemented here, so sublet's and this arm's detections were
    # qualified by the weak control alone.
    G "cp /root/boundary/$c/negative_control.py /root/boundary/$c/trigger.py"
    NCKIND=variant
  elif [ "$NEGCTL" = 1 ]; then
    NCKIND=stub
    # Only the trigger is replaced: an import failure would be a different
    # experiment. The interpreter starts, loads and exits without a defect.
    G "printf '%s\n' 'import sys' 'print(\"NEGATIVE-CONTROL no defect performed\")' 'sys.exit(0)' > /root/boundary/$c/trigger.py"
  fi
  [ "$NEGCTL" = 1 ] && printf '%s\t%s\n' "$c" "${NCKIND:-stub}" >> "$OUT/control-kind.tsv"
  # Staging is verified per case: a missing trigger.py produced 10 verdicts in
  # this lane that looked like defect results and were "can't open file".
  G "test -f /root/boundary/$c/trigger.py" || { echo "$c: staging failed" >&2; exit 2; }
  n=$((n+1))
done
echo "staged $n cases"

# ---- run ---------------------------------------------------------------
printf 'case\tarm\trc\tsignal\tsi_code\tquarantine\tlast\n' > "$OUT/verdicts.tsv"
printf 'case_budget\t%s\nonly\t%s\n' "$BUDGET" "${ONLY:-all}" >> "$OUT/run.meta"
for d in "$CORPUS"/[0-9][0-9]_*/; do
  c=$(basename "$d")
  in_subset "$c" || continue
  out=$(G "cd /root/boundary/$c && env PYTHONDONTWRITEBYTECODE=1 PYTHONHOME=$PYHOME \
            LD_PRELOAD=$SICODE:$QPROBE timeout $BUDGET $GUEST/python trigger.py 2>&1; \
          echo RC=\$?")
  # -a, and strip non-printables before anything is matched or recorded. Case
  # 02's upstream test asserts on a GARBLED field name -- the overflow's own
  # output -- so the trigger's stdout contains raw bytes. Without this, grep
  # calls the stream binary and the recorded `last` field reads
  # "Binary file (standard input) matches", which hides the actual last lines.
  out=$(printf '%s' "$out" | tr -d '\000' | LC_ALL=C tr -c '\11\12\15\40-\176' '?')
  rc=$(printf '%s' "$out" | grep -a -oE 'RC=[0-9]+' | tail -1 | cut -d= -f2)
  sig=$(printf '%s' "$out" | grep -a -oE 'signal=[0-9]+' | head -1 | cut -d= -f2)
  code=$(printf '%s' "$out" | grep -a -oE 'si_code=[0-9]+ \([A-Z_]+\)' | head -1)
  q=$(printf '%s' "$out" | grep -a -o 'QUARANTINE .*' | tail -1)
  last=$(printf '%s' "$out" | grep -a -v '^RC=\|^QUARANTINE ' | tail -2 | tr '\n' ' ')
  printf '%s\tcheribsd-revocation\t%s\t%s\t%s\t%s\t%s\n' \
    "$c" "${rc:-?}" "${sig:-}" "${code:-}" "${q:-}" "$last" >> "$OUT/verdicts.tsv"
  printf '  %-56s rc=%-4s %s\n' "${c:0:56}" "${rc:-?}" "${code:-}"
done

# ---- the configuration again, to show it did not move under the run ----
G 'sysctl security.cheri 2>/dev/null' > "$OUT/sysctl-after.txt"
if diff -q <(sed 's/^ *//' "$OUT/sysctl-before.txt") <(sed 's/^ *//' "$OUT/sysctl-after.txt") >/dev/null; then
  echo "security.cheri unchanged across the run"
else
  echo "WARNING: security.cheri CHANGED during the run:" >&2
  diff "$OUT/sysctl-before.txt" "$OUT/sysctl-after.txt" >&2
fi
printf 'ended_utc\t%s\n' "$(date -u +%FT%TZ)" >> "$OUT/run.meta"
if [ "$NEGCTL" = 1 ]; then
  bad=$(awk -F'\t' 'NR>1 && $5 != "" {print $1}' "$OUT/verdicts.tsv")
  m=$(printf '%s' "$bad" | grep -c . || true)
  printf 'negative_control\t1\nnegative_control_faults\t%s\n' "$m" >> "$OUT/run.meta"
  if [ "$m" != 0 ]; then
    echo "NEGATIVE CONTROL FAILED: $m case(s) faulted with no defect performed:" >&2
    printf '  %s\n' $bad >&2
    exit 1
  fi
  echo "negative control: 0 of $n cases faulted, as required"
fi
echo "rows: $(($(wc -l < "$OUT/verdicts.tsv")-1)) / $n"
echo "out:  $OUT"
