#!/bin/bash
# The Juliet board path's first run (2026-10-21; pre-registered in README.md before the boot): juliet/run.py --board-kit
# from branch juliet-board (a772f0112ae3, worktree /tmp/capstone/wt-jb) on a four-case subset of the compiler lane's cw
# temporal census build, one batch (--pairs 4), with the record's 10 s per-program limit. Board 1, kit bkit7-rev2xDN2.
# Usage: tier3-juliet-sub.sh <out log>. Books the board on the lab hub (refuses on 409) and cancels the booking at the
# end. Then reads every step of observations.json (juliet-sub-read.py). Ends by itself.
# JSUB_RUN / JSUB_BUSY / JSUB_BOOK / JSUB_ROOT stub the dry run.
main() {
  set -u
  local O=${1:?out log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone JB=/tmp/capstone/wt-jb/capstone/bug-corpora/juliet
  local K=$H/t3-smoke/bkit7-rev2xDN2 B=$H/juliet-vp2-board/sub4 R=${JSUB_ROOT:-$H/t3-run-treeA/juliet-sub4}
  local HELPER=$H/juliet-vp2/juliet-timeout rc bid
  [ "$(git -C /tmp/capstone/wt-jb rev-parse --short=12 HEAD)" = a772f0112ae3 ] && [ -z "$(git -C /tmp/capstone/wt-jb status --porcelain)" ] \
    || { echo "JSUB STOP: wt-jb is not a772f0112ae3, clean" >> "$O"; return 1; }
  [ "$(sha256sum $HELPER | cut -c1-16)" = bb65b77ac802a9f3 ] || { echo "JSUB STOP: helper is not bb65b77a" >> "$O"; return 1; }
  [ "$(sha256sum $B/build.json | cut -c1-16)" = "${JSUB_BUILD_SHA:-741cd6b3bf55fb93}" ] || { echo "JSUB STOP: subset build.json is not the pre-registered one" >> "$O"; return 1; }
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=caplifive_vp2c_8522d72fe" $K/board.env \
    && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 54d8a3af1158c3f4 ] \
    || { echo "JSUB STOP: kit $K is not the board-1 54d8a3af kit" >> "$O"; return 1; }
  [ -e "$R" ] && { echo "JSUB STOP: $R exists" >> "$O"; return 1; }
  ${JSUB_BUSY:-python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1} > /dev/null || { echo "JSUB STOP: board 1 is busy" >> "$O"; return 1; }
  bid=$(${JSUB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book 1 45 "vp2-tier12 juliet board-path subset (board 1)"); rc=$?
  [ $rc = 3 ] && { echo "JSUB STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
  [ $rc = 0 ] || { echo "JSUB NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
  echo "JSUB BEGIN $(date +%T) run.py $(sha256sum $JB/run.py | cut -c1-16) build $(sha256sum $B/build.json | cut -c1-16) booking ${bid:-none}" >> "$O"
  taskset -c 16-31 timeout 3600 ${JSUB_RUN:-python3 $JB/run.py} --build $B --board-kit $K --helper $HELPER --out $R --pairs 4 > $R.log 2>&1; rc=$?
  [ -n "$bid" ] && ${JSUB_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
  echo "JSUB END rc=$rc $(date +%T) :: $(tail -1 $R.log | cut -c1-160)" >> "$O"
  python3 $S/juliet-sub-read.py $R >> "$O" 2>&1; echo "JSUB READ rc=$?" >> "$O"
  echo "JSUB DONE $(date +%T)" >> "$O"
}
main "$@"; exit
