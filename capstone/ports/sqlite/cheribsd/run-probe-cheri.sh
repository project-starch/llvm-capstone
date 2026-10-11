#!/bin/bash
# Re-run the SQLite corpus on CheriBSD with the reachability probe armed.
#
# WHAT THIS ADDS over run-new13.sh / run-corpus-cheri.sh. Those report the exit code
# and the si_code of any fault, which answers "did CHERI trap". They cannot answer
# "did the defective access happen at all", and for the 26 cases that came back
# silent that is the only question that matters: a clean run and an input that never
# reached the bug produce the same log.
#
# Each binary here is built from sqlite3-probe.c, which carries one marker at the
# exact sqlite3.c line where the host ASan oracle -- running THIS case source with
# THIS arm's flags -- reported the memory error. LB_SITE arms that one marker.
# Three outcomes, and they are different findings:
#   LB_SUMMARY hits=0          the defect site never executed. No conclusion about
#                              the mechanism; the input did not get there.
#   hits>0, witnesses=0        the site executed; the shadow could not prove this
#                              particular access was stale (memsys5 may have already
#                              reissued the block).
#   witnesses>0                the defective access happened, and we can say which
#                              kind -- UAF, OOB past the REQUESTED size, or CAP-OOB.
#                              With no fault, that is "reached and not detected".
#
# ONE CONNECTION AT A TIME, NO RETRY LOOP. The guest runs sshd with
# MaxStartups 10:30:100 and a hostfwd backlog of 1; a retry loop is a self-inflicted
# denial of service, and this project has wedged the guest that way once already.
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
K="-n -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
G() { ssh $K -p 10086 root@localhost "$@" 2>/dev/null; }
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$WORK/results/probe-$STAMP
mkdir -p "$OUT"

# tag -> armed site. A tag absent from this table has no derived site and is not run
# here: there is nothing to arm, and a run with LB_SITE=0 would print hits=0, which
# is exactly the reading ("never reached") that must not be produced by accident.
declare -A SITE=(
  [mem5design]=1 [blobclose]=2 [fts5rank]=3 [staticbind]=4 [rtreestatic]=4
  [wschema]=5 [fz09]=6 [jsoneachroot]=7 [jsoneachstatic]=7 [a783931794_0]=8
  [fts3destroyoom]=9 [c7def600bd_0]=10 [2c7a73eaea_0]=11 [fts3snipor]=12
  [fts5vocabeof]=13 [fts5structwrite]=14 [fts5inplace]=15 [634ac14488_0]=16
  [33cf194218_0]=17 [bfe33f80dd_0]=18 [8f5b14a5c2_0]=19 [415540ddaa_0]=20
  [adfb203a7d_0]=20 [expertrem]=21
)
# memsys5 arm, then the system-allocator arm for the images that have one.
# Overridable so a gap can be filled without editing the list: MEM5="a b" SYS="" ./run-probe-cheri.sh
MEM5=${MEM5:-"mem5design blobclose fts5rank staticbind rtreestatic wschema fz09 jsoneachroot
      jsoneachstatic a783931794_0 fts3destroyoom c7def600bd_0 2c7a73eaea_0 fts3snipor
      fts5vocabeof fts5structwrite fts5inplace 634ac14488_0 33cf194218_0 bfe33f80dd_0
      8f5b14a5c2_0 415540ddaa_0 adfb203a7d_0 expertrem"}
SYS=${SYS-"634ac14488_0 a783931794_0 bfe33f80dd_0 c7def600bd_0 fz09 2c7a73eaea_0 8f5b14a5c2_0
     33cf194218_0 415540ddaa_0 adfb203a7d_0"}

echo "knobs: $(G 'sysctl -n security.cheri.runtime_revocation_default security.cheri.runtime_revocation_every_free_default' | paste -sd/ -)"
echo

one() {   # tag arm
  # A tag with no entry in SITE has no probe site derived for it. That is not
  # the same as hits=0: the instrument is absent, not negative. Record it as
  # '-' and let the verdict say NO-PROBE rather than NOT-REACHED.
  local tag=$1 arm=$2 site=${SITE[$1]:--} bin=$1 label=$1 envp="" e img v
  [ "$arm" = sys ] && { bin="${tag}_sys"; label="${tag}_sys"; envp="NOMEM5=1 "; }
  if ! G "test -x /root/corpus-probe/$bin"; then
    printf '%-18s %-5s %-4s %s\n' "$label" "$site" "$arm" "NOBIN"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$site" "$arm" NOBIN "" >> "$OUT/probe.tsv"
    return
  fi
  # fresh image: several of these WRITE, and a mutated file silently changes what the
  # other arm is handed
  img=${tag}
  G "cd /root/corpus && [ -f $img.db.orig ] && cp -f $img.db.orig $img.db || true"
  e=$(G "cd /root/corpus && env ${envp}LB_SITE=$site timeout ${CASE_TIMEOUT:-300} /root/corpus-probe/$bin >/tmp/p.txt 2>&1; echo \$?")
  G "cat /tmp/p.txt" > "$OUT/$label.$arm.log" 2>/dev/null
  local L="$OUT/$label.$arm.log"
  local armed hits wit kinds code
  armed=$(grep -c "^LB_ARMED site=$site\$" "$L")
  hits=$(grep -m1 -oh "hits=[0-9]*" "$L" | cut -d= -f2)
  wit=$(grep -c "^LB_WITNESS " "$L")
  kinds=$(grep -oh "kind=[A-Z-]*" "$L" | sort -u | tr '\n' ',' | sed 's/,$//')
  code=$(grep -ao 'si_code=[0-9]* ([^)]*)' "$L" | head -1)
  if [ "$arm" = sys ] && ! grep -q 'allocator=system' "$L" && [ "$tag" != fts3destroyoom ]; then
    v="BADCONFIG"   # fts3destroyoom does its own init and never consults NOMEM5
  elif [ "${e:-?}" = 124 ]; then v="TIMEOUT"
  # A fault is a fact about the MECHANISM and does not depend on the probe being
  # armed. Checking "armed" first threw away two real PROT_CHERI_BOUNDS faults
  # (fz02_r2, fz10_r2) and filed them NOT-ARMED, because those tags have no probe
  # site. The instrument's absence cannot overrule the mechanism's report.
  elif [ -n "$code" ]; then v="FAULT"
  elif [ "$armed" -ne 1 ]; then v="NOT-ARMED"
  elif [ "$site" = "-" ]; then v="NO-PROBE"
  elif [ "${hits:-0}" -eq 0 ]; then v="NOT-REACHED"
  elif [ "$wit" -gt 0 ]; then v="WITNESS"
  else v="REACHED"; fi
  printf '%-18s %-5s %-4s %-12s hits=%-6s wit=%-5s %s %s\n' \
    "$label" "$site" "$arm" "$v" "${hits:-?}" "$wit" "$kinds" "$code"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$site" "$arm" "$v" "${hits:-?}" "$wit" "$kinds" "$code" >> "$OUT/probe.tsv"
}

G 'cd /root/corpus && for f in *.db; do [ -f $f.orig ] || cp $f $f.orig; done'
printf '%-18s %-5s %-4s %-12s %s\n' CASE SITE ARM VERDICT DETAIL
for t in $MEM5; do one "$t" mem5; done
for t in $SYS;  do one "$t" sys;  done
echo
echo "results: $OUT"
