#!/bin/bash
# Ship the corpus binaries into the CheriBSD guest. One connection at a time.
# Sibling of deploy-probe.sh; this one carries out/ -> /root/corpus, which is
# where run-corpus-cheri.sh looks. With SYSALLOC=1 having been built, out/ holds
# both arms: <tag> against memsys5 and <tag>_sys against the system allocator.
set -uo pipefail
K="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=120"
# Paths. C is this directory, inside the repository, and holds the sources:
# cases/, poscontrol.c, repro322_common.h and the sibling scripts. WORK is
# where build output and run logs go, which must NOT be in the repository; it
# defaults to the out-of-tree directory these scripts were developed in, so
# behaviour is unchanged unless it is set. The toolchain and the pinned SQLite
# amalgamation are machine-specific and are overridable for the same reason.
C=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORK=${SQLITE_CHERI_WORK:-$HOME/arms/sqlite/cheribsd}
mkdir -p "$WORK"
tar czf /tmp/corpus.tgz -C "$WORK/out" . || exit 1
echo "tgz $(stat -c %s /tmp/corpus.tgz) bytes  ($(ls "$WORK/out" | wc -l) files, $(ls "$WORK/out" | grep -c _sys) of them _sys)"
scp $K -P 10086 /tmp/corpus.tgz root@localhost:/tmp/ || { echo "scp FAILED"; exit 1; }
ssh -n $K -p 10086 root@localhost 'rm -rf /root/corpus && mkdir -p /root/corpus &&
  tar xzf /tmp/corpus.tgz -C /root/corpus && chmod +x /root/corpus/* &&
  echo "deployed: $(ls /root/corpus | wc -l) files, $(ls /root/corpus | grep -c _sys) _sys"'
