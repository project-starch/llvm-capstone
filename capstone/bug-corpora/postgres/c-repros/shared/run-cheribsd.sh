#!/bin/bash
# Run the c-repros cases in a booted CheriBSD purecap guest.
#
#   run-cheribsd.sh [out dir]
#
# shared/build-cheri.sh produces one static binary per case; this runs each in
# the guest and classifies it. The guest is expected to be up and reachable on
# PG_CHERI_PORT (default 10086); this script does not boot it.
#
# ONE CONNECTION AT A TIME AND NO RETRY LOOPS. The guest is shared with other
# lanes: a loop that reconnects on failure turns one lane's bad minute into
# everyone's, and a parallel run makes every timing-sensitive case unreadable.
#
# VERDICTS, and why the signal matters rather than just the fact of a fault:
#   detected        exit 162, SIGPROT(34). A capability violation, which is
#                   the mechanism reporting. This is the only one that counts.
#   port-defect     SIGBUS(10) or SIGSEGV(11). An alignment or representability
#                   fault is the port being wrong, not the defect being caught,
#                   and scoring it as a detection would inflate the arm.
#   control-failure exit 75, or the driver's CONTROL-FAILED line: the case
#                   refused its own setup, so it says nothing about the arm.
#   silent          returned, with its PG_DEFECT marker printed, so the
#                   defective access provably ran and nothing trapped.
#   BADRUN          returned without the marker: no evidence it was reached.
set -uo pipefail
C=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
CORPUS=$(cd "$C/.." && pwd)
BIN=${PG_CHERI_BIN:-$CORPUS/out-cheri}
PORT=${PG_CHERI_PORT:-10086}
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=${1:-$CORPUS/results/cheribsd-revocation-$STAMP}

# -n belongs to ssh and not to scp, which rejects it and then copies nothing.
# The first run of this script lost its positive control that way: the binary
# never reached the guest, the gate saw exit 127, and it refused to score --
# which is the gate working, but the cause was here.
K="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no"
K="$K -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
G() { ssh -n $K -p "$PORT" root@localhost "$@" 2>/dev/null; }
# Copies are checked: a silent scp failure reads downstream as a missing
# binary, which is indistinguishable from a build that never produced one.
P() { scp $K -P "$PORT" "$1" root@localhost:"$2" >/dev/null 2>&1 \
      || { echo "scp of $1 to the guest failed" >&2; exit 2; }; }

mkdir -p "$OUT"

# The case list comes from the corpus, never from `ls` of the build directory.
# "What was built" is a different question, and on the SQLite arm the two
# answers once differed without saying so: probes were measured as cases and
# real cases were absent, and both totals came to the same number.
CASES=$(cd "$CORPUS" && ls -d [0-9][0-9]_*/ 2>/dev/null | tr -d / | sort)
[ -n "$CASES" ] || { echo "no cases under $CORPUS" >&2; exit 2; }

missing=""
for tag in $CASES; do [ -x "$BIN/$tag" ] || missing="$missing $tag"; done
if [ -n "$missing" ]; then
  echo "REFUSING TO RUN: no purecap binary in $BIN for:" >&2
  for t in $missing; do echo "  $t" >&2; done
  echo "build them with shared/build-cheri.sh; a partial run would understate" >&2
  echo "the denominator while looking like a complete one" >&2
  exit 2
fi

# Read the guest's own settings back rather than assuming them. A run recorded
# against a revocation setting it did not have is worse than no run.
REVD=$(G 'sysctl -n security.cheri.runtime_revocation_default')
REVF=$(G 'sysctl -n security.cheri.runtime_revocation_every_free_default')
KERN=$(G 'uname -r')
[ -n "$KERN" ] || { echo "no answer from the guest on port $PORT" >&2; exit 2; }
echo "kernel=$KERN revocation_default=$REVD every_free_default=$REVF"

# POSITIVE CONTROL. These are spatial defects, so what has to be armed is
# capability BOUNDS, not revocation. Case 00 writes 4096 bytes into an 8-byte
# malloc'd object: on a guest whose bounds are enforced it cannot survive. If
# it does, nothing below is a measurement and the silence would be the guest's,
# not the mechanism's. It is a corpus case doing double duty, which is recorded
# as such rather than presented as a purpose-built control.
GATE=$(printf '%s\n' $CASES | grep '^00_' | head -1)
[ -n "$GATE" ] || { echo "no case 00 to use as the positive control" >&2; exit 2; }
G "mkdir -p /root/c-repros"
P "$BIN/$GATE" /root/c-repros/
graw=$(G "cd /root/c-repros && timeout 120 ./$GATE 0 2>&1; echo __EXIT=\$?")
gex=$(printf '%s' "$graw" | grep -o '__EXIT=[0-9]*' | tail -1 | cut -d= -f2)
echo "positive control $GATE: exit=${gex:-?}"
if [ "${gex:-0}" != 162 ]; then
  echo "POSITIVE CONTROL FAILED: a 4096-byte write into an 8-byte object did" >&2
  echo "not raise SIGPROT (exit 162); got ${gex:-none}. Not scoring defects on" >&2
  echo "a guest that is not enforcing bounds." >&2
  exit 2
fi

TSV=$OUT/matrix.tsv
printf 'case\tarm\tverdict\tevidence\n' > "$TSV"
n=0
for tag in $CASES; do
  n=$((n+1))
  num=$(printf '%s' "$tag" | cut -d_ -f1 | sed 's/^0*//'); num=${num:-0}
  P "$BIN/$tag" /root/c-repros/
  raw=$(G "cd /root/c-repros && timeout 300 ./$tag $num 2>&1; echo __EXIT=\$?")
  ex=$(printf '%s' "$raw" | grep -o '__EXIT=[0-9]*' | tail -1 | cut -d= -f2)
  body=$(printf '%s' "$raw" | grep -v '__EXIT=')
  printf '%s\n' "$body" > "$OUT/$tag.out"
  fault=$(printf '%s' "$body" | grep -oE 'PG_FAULT [^\n]*' | head -1 | cut -c1-140)
  last=$(printf '%s' "$body" | tr -d '\r' | grep -v '^$' | tail -1 | cut -c1-80)

  # Order matters: a control failure is not a verdict however the process died,
  # and a port defect is not a detection however much it looks like one.
  if printf '%s' "$body" | grep -q 'CONTROL-FAILED' || [ "${ex:-0}" = 75 ]; then
    v=control-failure; why="the case refused its own setup: $last"
  elif [ "${ex:-0}" = 162 ]; then
    v=detected; why="SIGPROT(34)${fault:+; $fault}"
  elif [ "${ex:-0}" = 139 ] || [ "${ex:-0}" = 138 ]; then
    v=port-defect; why="$([ "${ex}" = 139 ] && echo 'SIGSEGV(11)' || echo 'SIGBUS(10)') -- a port defect, not a detection; $last"
  elif [ "${ex:-0}" = 124 ]; then
    v=other; why="timed out; not a measurement"
  elif ! printf '%s' "$body" | grep -q "case $num BEGIN"; then
    v=BADRUN; why="no BEGIN line -- the binary did not run"
  elif ! printf '%s' "$body" | grep -q 'PG_DEFECT'; then
    v=BADRUN; why="ran but printed no PG_DEFECT marker, so there is no evidence the defective access was reached"
  elif printf '%s' "$body" | grep -q "case $num RETURNED"; then
    v=silent; why="reached its marker and returned; the mechanism did not report"
  else
    v=other; why="exit=${ex:-?}; $last"
  fi
  printf '%-52s %-16s %s\n' "$tag" "$v" "${why:0:60}"
  printf '%s\tcheribsd-revocation\t%s\t%s\n' "$tag" "$v" "$why" >> "$TSV"
done

scored=$(awk -F'\t' 'NR>1 && $3!="control-failure" && $3!="BADRUN" && $3!="other"' "$TSV" | wc -l)
cat > "$OUT/inputs.json" <<JSON
{
  "arm": "cheribsd-revocation",
  "kernel": "$KERN",
  "revocation_default": "$REVD",
  "revocation_every_free_default": "$REVF",
  "binaries": "$BIN",
  "started_utc": "$STAMP",
  "positive_control": {
    "case": "$GATE",
    "exit": ${gex:-null},
    "kind": "a corpus case doing double duty: a 4096-byte write into an 8-byte malloc'd object, which must raise SIGPROT if bounds are enforced"
  },
  "cases": $n,
  "scored": $scored
}
JSON
echo
echo '=== summary ==='
awk -F'\t' 'NR>1{c[$3]++} END{for(k in c) printf "  %-16s %d\n", k, c[k]}' "$TSV"
echo "scored $scored of $n"
echo "results: $OUT"
