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
PY=${CHERI_PYTHON:-$HOME/arms/cpython/cheribsd/build-tgot/python}
GUEST=${GUEST:-/root/cpython}
# PYTHONHOME is the pyhome DIRECTORY INSIDE the install, not the install.
# With the wrong one every case dies before any Python runs, with
# "No module named encodings" -- indistinguishable from a silent arm.
PYHOME=${PYHOME:-$GUEST/pyhome}
SICODE=${SICODE:-/root/sicode.so}
GUEST_PORT=${GUEST_PORT:-10086}
OUT=${1:-$HOME/arms/cpython/cheribsd/results/boundary-$(date -u +%Y%m%d-%H%M%S)}
BUDGET=${CASE_BUDGET:-120}
K="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=120"
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
G "test -f $SICODE" || { echo "no $SICODE in the guest -- push it before running" >&2; exit 2; }
pc=$(G "cd $GUEST && env PYTHONHOME=$PYHOME LD_PRELOAD=$SICODE timeout 300 ./python objects.py 8 3 0 2>&1 | tail -1")
echo "positive control: ${pc:-<no output>}"
case "$pc" in
  EXP-OK*) ;;
  *) echo "REFUSING: the interpreter is not qualified on this guest." >&2
     echo "  Got: ${pc:-<no output>}" >&2
     echo "  A run now would record 21 silences that are the harness, not the arm." >&2
     exit 2 ;;
esac
# si_code reporting is what makes this arm informative at all: without it a
# fault is just "it crashed", and BOUNDS, TAG and PERM are indistinguishable.
#
# The probe is mech-control S_OOB -- an in-bounds pointer read past the end --
# and it carries si_code out through its EXIT STATUS, _exit(100 + si_code), so
# this reads the status directly. Reading it through a pipeline would give the
# pipeline's status instead, which is how an earlier check here reported rc=0
# for a probe that had in fact faulted.
MECH=${MECH:-/root/mech-control}
G "test -x $MECH" || { echo "no $MECH in the guest -- the si_code handler cannot be controlled" >&2
  echo "  build it from ports/cpython/cheribsd and push it, or set MECH=" >&2; exit 2; }
scrc=$(G "cd /root && env LD_PRELOAD=$SICODE timeout 60 $MECH S_OOB >/dev/null 2>&1; echo \$?")
case "$scrc" in
  101) sc="si_code=1 (PROT_CHERI_BOUNDS) via mech-control S_OOB" ;;
  1??) sc="si_code=$((scrc-100)) via mech-control S_OOB"
       echo "WARNING: S_OOB reported si_code $((scrc-100)), expected 1 (BOUNDS)." >&2
       echo "  The handler works, but a spatial probe faulting for another reason" >&2
       echo "  means this arm's si_code values need rechecking before they are read." >&2 ;;
  *)   echo "REFUSING: mech-control S_OOB exited $scrc, not 100+si_code." >&2
       echo "  The si_code handler reported nothing on a deliberate out-of-bounds read," >&2
       echo "  so every fault this arm sees would be unclassifiable." >&2
       exit 2 ;;
esac
echo "si_code control:  $sc"
printf 'positive_control\t%s\nsicode_control\t%s\n' "$pc" "$sc" >> "$OUT/run.meta"

# ---- stage: one directory per case, named as the corpus names it --------
G "rm -rf /root/boundary && mkdir -p /root/boundary"
n=0
for d in "$CORPUS"/[0-9][0-9]_*/; do
  c=$(basename "$d")
  G "mkdir -p /root/boundary/$c"
  tar -C "$d" -cf - trigger.py $( [[ -f $d/upstream_test.py ]] && echo upstream_test.py ) \
    | G "tar -C /root/boundary/$c -xf -"
  # Staging is verified per case: a missing trigger.py produced 10 verdicts in
  # this lane that looked like defect results and were "can't open file".
  G "test -f /root/boundary/$c/trigger.py" || { echo "$c: staging failed" >&2; exit 2; }
  n=$((n+1))
done
echo "staged $n cases"

# ---- run ---------------------------------------------------------------
printf 'case\tarm\trc\tsignal\tsi_code\tlast\n' > "$OUT/verdicts.tsv"
for d in "$CORPUS"/[0-9][0-9]_*/; do
  c=$(basename "$d")
  out=$(G "cd /root/boundary/$c && env PYTHONDONTWRITEBYTECODE=1 PYTHONHOME=$PYHOME \
            LD_PRELOAD=$SICODE timeout $BUDGET $GUEST/python trigger.py 2>&1; \
          echo RC=\$?")
  rc=$(printf '%s' "$out" | grep -oE 'RC=[0-9]+' | tail -1 | cut -d= -f2)
  sig=$(printf '%s' "$out" | grep -oE 'signal=[0-9]+' | head -1 | cut -d= -f2)
  code=$(printf '%s' "$out" | grep -oE 'si_code=[0-9]+ \([A-Z_]+\)' | head -1)
  last=$(printf '%s' "$out" | grep -v '^RC=' | tail -2 | tr '\n' ' ')
  printf '%s\tcheribsd-revocation\t%s\t%s\t%s\t%s\n' \
    "$c" "${rc:-?}" "${sig:-}" "${code:-}" "$last" >> "$OUT/verdicts.tsv"
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
echo "rows: $(($(wc -l < "$OUT/verdicts.tsv")-1)) / $n"
echo "out:  $OUT"
