#!/bin/bash
# Rebuild both allocator arms, prove nothing stale survived, then deploy and run.
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
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "rebuilding plain (memsys5)"
"$C/build-corpus-cheri.sh" all           > /tmp/build-plain.log 2>&1 || { log "plain build failed"; exit 1; }
log "rebuilding _sys (system allocator)"
SYSALLOC=1 "$C/build-corpus-cheri.sh" all > /tmp/build-sys.log  2>&1 || { log "sys build failed"; exit 1; }
for f in /tmp/build-plain.log /tmp/build-sys.log; do
  n=$(grep -c FAIL "$f"); [ "$n" = 0 ] || { log "$n FAIL lines in $f"; grep FAIL "$f"; exit 1; }
done

# THE CHECK THAT WAS MISSING. build_group recompiles unconditionally, so a case
# that is not in the group list is never touched and never reported -- it just
# keeps an older binary. That is how 13 of 43 cases ran from 2026-10-03 binaries
# after repro322_common.h changed. Anything older than the header is stale.
log "staleness audit against repro322_common.h"
H=$(stat -c %Y "$C/repro322_common.h"); stale=0
for f in "$WORK"/out/*; do
  [ "$(stat -c %Y "$f")" -lt "$H" ] && { echo "  STALE $(basename "$f") $(stat -c %y "$f" | cut -c1-19)"; stale=$((stale+1)); }
done
[ "$stale" = 0 ] || { log "ABORT: $stale binaries predate the harness header"; exit 1; }
log "no stale binaries; $(ls "$WORK"/out | wc -l) files, $(ls "$WORK"/out | grep -c _sys) _sys"

log "deploying"
"$C/deploy-corpus.sh" || { log "deploy FAILED"; exit 1; }

log "gate: positive control must SIGPROT"
K="-n -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
pc=$(ssh $K -p 10086 root@localhost 'cd /root/corpus && timeout 120 ./poscontrol >/dev/null 2>&1; echo $?' 2>/dev/null)
log "poscontrol exit=$pc (162 = 128+SIGPROT(34))"
[ "$pc" = 162 ] || { log "GATE FAILED: revocation inactive, results void"; exit 1; }

log "arm 3: purecap + memsys5"
"$C/run-corpus-cheri.sh"   > /tmp/arm-mem5.log 2>&1
log "arm 4: purecap + system malloc, revocation forced every free"
"$C/run-sysalloc-cheri.sh" > /tmp/arm-sys.log  2>&1
log "DONE"
