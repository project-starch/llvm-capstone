#!/bin/bash
# Run the corpus inside the booted CheriBSD guest and classify each case.
#
# Verdicts (same bar as the Capstone arm):
#   PASS  - printed "<tag> RETURNED rc=0": the freed-then-used path ran and nothing trapped.
#   FAULT - died on a signal. SIGPROT(34)=capability violation, SIGSEGV(11)=plain bad
#           address, SIGBUS(10)=alignment/representability. Recorded separately: only
#           SIGPROT is a capability check, and SIGBUS means the port is broken, not caught.
#   ERR   - returned with rc!=0, i.e. a soft-error path before the bug site.
# Tags come from the HOST out/ directory, never from `ls` in the guest: cases can create
# files there (one wrote a garbage-named file from dangling memory used as a filename).
set -uo pipefail
# Paths. C is this directory, inside the repository, and holds the sources:
# cases/, poscontrol.c, repro322_common.h and the sibling scripts. WORK is
# where build output and run logs go, which must NOT be in the repository; it
# defaults to the out-of-tree directory these scripts were developed in, so
# behaviour is unchanged unless it is set. The toolchain and the pinned SQLite
# amalgamation are machine-specific and are overridable for the same reason.
C=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORK=${SQLITE_CHERI_WORK:-$HOME/arms/sqlite/cheribsd}
mkdir -p "$WORK"
SSHK="-n -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
G() { ssh $SSHK -p 10086 root@localhost "$@" 2>/dev/null; }
mkdir -p "$WORK/results"
STAMP=$(date +%Y%m%d-%H%M%S)
TSV="$WORK/results/run-$STAMP.tsv"
printf 'tag\tverdict\texit\tsignal\tlast_line\n' > "$TSV"

echo "revocation_default=$(G 'sysctl -n security.cheri.runtime_revocation_default')"
echo "every_free_default=$(G 'sysctl -n security.cheri.runtime_revocation_every_free_default')"
echo "kernel=$(G 'uname -r')"
echo

# THE CASE LIST COMES FROM THE CORPUS, NOT FROM WHAT HAPPENS TO BE BUILT.
# `ls out/` answers "what was built", which is a different question, and on
# 2026-10-05 the two answers differed without saying so: five baseline probes
# were measured as if they were cases and five real cases were absent, the
# totals both came to 43, and nothing in the output disagreed with the corpus.
# tags.tsv is the accounting -- one row per tag, giving the case directory it
# belongs to or the reason it is not a case.
TAGS=$C/../repro322/tags.tsv
[ -r "$TAGS" ] || { echo "cannot read $TAGS" >&2; exit 2; }
CASES=$(awk -F'\t' '$3=="case"{print $1}' "$TAGS" | sort)
[ -n "$CASES" ] || { echo "no cases in $TAGS" >&2; exit 2; }

# Refuse rather than measure a short set. A missing binary is a build problem,
# and scoring the rest would report a smaller denominator as if it were whole.
missing=""
for tag in $CASES; do [ -x "$WORK/out/$tag" ] || missing="$missing $tag"; done
if [ -n "$missing" ]; then
  echo "REFUSING TO RUN: these cases have no memsys5 binary in $WORK/out:" >&2
  for t in $missing; do echo "  $t" >&2; done
  echo "build them first; a partial run would understate the denominator" >&2
  exit 2
fi
echo "cases from $TAGS: $(printf '%s\n' $CASES | wc -l)"

for tag in $CASES; do
  raw=$(G "cd /root/corpus && timeout 300 ./$tag 2>&1; echo __EXIT=\$?")
  ex=$(printf '%s' "$raw" | grep -o '__EXIT=[0-9]*' | tail -1 | cut -d= -f2)
  body=$(printf '%s' "$raw" | grep -v '__EXIT=')
  last=$(printf '%s' "$body" | tr -d '\r' | grep -v '^$' | tail -1 | cut -c1-80)
  sig=''
  case "${ex:-none}" in
    0)   if printf '%s' "$body" | grep -q 'RETURNED rc=0'; then v=PASS
         elif printf '%s' "$body" | grep -q 'RETURNED rc='; then v=ERR
         else v=NORUN; fi ;;
    124) v=TIMEOUT ;;
    162) v=FAULT; sig='SIGPROT(34)' ;;
    139) v=FAULT; sig='SIGSEGV(11)' ;;
    138) v=FAULT; sig='SIGBUS(10)' ;;
    1[3-9][0-9]) v=FAULT; sig="sig$((ex-128))" ;;
    *)   v=ERR ;;
  esac
  printf '%-16s %-8s exit=%-5s %-12s %s\n' "$tag" "$v" "${ex:-?}" "$sig" "$last"
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$v" "${ex:-?}" "$sig" "$last" >> "$TSV"
done
echo
echo '=== summary ==='
awk -F'\t' 'NR>1{c[$2]++; if($4!="")s[$4]++} END{for(k in c) printf "%-8s %d\n",k,c[k]; print "--- signals ---"; for(k in s) printf "%-12s %d\n",k,s[k]}' "$TSV"
echo "tsv=$TSV"
