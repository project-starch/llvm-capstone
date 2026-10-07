#!/bin/bash
# Ship the probe binaries into the CheriBSD guest. One connection at a time.
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
tar czf /tmp/corpus-probe.tgz -C "$WORK/out-probe" . || exit 1
echo "tgz $(stat -c %s /tmp/corpus-probe.tgz) bytes"
scp $K -P 10086 /tmp/corpus-probe.tgz root@localhost:/tmp/ || { echo "scp FAILED"; exit 1; }
ssh -n $K -p 10086 root@localhost 'rm -rf /root/corpus-probe && mkdir -p /root/corpus-probe &&
  tar xzf /tmp/corpus-probe.tgz -C /root/corpus-probe && chmod +x /root/corpus-probe/* &&
  ls /root/corpus-probe | wc -l'
