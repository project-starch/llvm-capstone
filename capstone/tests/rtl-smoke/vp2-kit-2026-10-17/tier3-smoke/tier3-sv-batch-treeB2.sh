#!/bin/bash
# BOARD 2 (2026-10-19 evening, the lead's own word: tree B flashed onto board 2, both boards used): tier3-sv-batch-treeB.sh with
# FPGA_BOARD=2, board 2's URL file, dev's runner with the lease client (/tmp/capstone/wt-b2), outputs under treeB2-session.
# TREE B (copied from tree A's batch, 2026-10-19): the same seven arms on caplifive_vp2c_8de3c48cf, judged against the RTL
# lane's TREE B column (compare-sv.py --tree B), after tree B's steps 1-3 end CHAIN_TREEB_STEPS123_DONE. Tree A's header:
# S-29 and S-23's fetch face on silicon (2026-10-19): the RTL lane's seven bare arms (dev 8e1df1e05a3c,
# satpstale-bare-2026-10-19) on the resident tree A, one power cycle each, through the 25 arms' runner and infra-retry.
# Each is compared with its tree A vector by compare-sv.py. Order: the five va-satpstale arms, then the satpu2 pair last
# (sv-fetch-c, then sv-fetch), the one construction with S/U and a second chain. The images are matched by
# images/SHA256SUMS, and the batch fails unless the table gives exactly 7 arms. It runs after the tree A Linux steps end
# (TAL DONE or STOP: bare arms do not need Linux; a HOLD, meaning steps 1-3 did not finish, holds it), and while rev8's
# firmware builds, which never touches the board. Usage: tier3-sv-batch.sh <TAL out> <out>. Ends by itself.
# SVB_RUNNER / SVB_DIR / SVB_WAIT replace the inner runner, the output directory and the board-free wait for the dry run.
main() {
  set -u
  local W=${1:?TAL out} O=${2:?out} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local F=/tmp/capstone/land-board/capstone/tests/rtl-smoke/satpstale-bare-2026-10-19 V=${SVB_DIR:-/tmp/capstone/treeB2-session/sv}
  local RT=/tmp/capstone/wt-b2/capstone/tests/rtl-smoke/sup-bare-2026-10-03-ladder/run_sup_bare.py
  local R=${SVB_RUNNER:-$RT} end=$(( $(date +%s) + 86400 )) n rc pass=0
  until grep -a -q "^CHAIN_TREEB_\|^STOP" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^CHAIN_TREEB_STEPS123_DONE" "$W" || { echo "SVBATCH HOLD: $(tail -1 "$W")" >> "$O"; return 1; }
  ( cd $F/images && sha256sum -c --quiet SHA256SUMS ) || { echo "SVBATCH STOP: the images fail SHA256SUMS" >> "$O"; return 2; }
  [ "$(python3 -c "import importlib.util as u; s=u.spec_from_file_location('c','$S/compare-sv.py'); c=u.module_from_spec(s); s.loader.exec_module(c); print(len(c.table('B')))")" = 7 ] \
    || { echo "SVBATCH STOP: the tree B table does not give 7 arms" >> "$O"; return 2; }
  ${SVB_WAIT:-bash /tmp/capstone/treeB2-session/board-free-wait.sh 21600} || { echo "SVBATCH STOP: the board stayed busy" >> "$O"; return 1; }
  export FPGA_BITSTREAM=caplifive_vp2c_8de3c48cf.bit
  export FPGA_BOARD=2 FPGA_URL="$(cat $HOME/.claude-kisp/secrets/fpga-console-url-board2)"
  mkdir -p $V; echo "SVBATCH START $(date +%T) on $FPGA_BITSTREAM" >> "$O"
  for n in sv-stale sv-stale-f sv-stale-b sv-stale-bf sv-stale-bn sv-fetch-c sv-fetch; do
    SUP_IMG=$F/images/$n.bin SUP_REC_ADDR=$(cat $F/images/$n.rec) SUP_OUT=$V/$n SUP_TIMEOUT=120 SUP_REAL_RUNNER=$R \
      timeout 1200 python3 /tmp/capstone/treeB2-session/infra-retry.py > $V/$n.log 2>&1
    python3 $S/compare-sv.py $n $V/$n/uart.txt --tree B > $V/$n.cmp 2>&1; rc=$?   # never filter between a check and its status
    echo "SVARM $n cmp rc=$rc :: $(tail -2 $V/$n.cmp | tr '\n' ' ') $(date +%T)" >> "$O"; [ $rc -eq 0 ] && pass=$((pass+1)); sleep ${SVB_GAP:-8}
  done
  echo "SVBATCH DONE $(date +%T) $pass of 7 as pre-registered for tree B" >> "$O"
}
main "$@"; exit $?
