#!/bin/bash
# Stubbed dry run of objb-bare.sh alone: the REAL objb-bare.sh and objb-bare-grade.py, the real image kit and its sums;
# the runner (supbare-stub.py), busy check and hub stubbed. Usage: run-bare-dryrun.sh <scenario> <scratch dir>.
main() {
  set -u
  local SC=${1:?scenario} T=${2:?scratch} D BN
  D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  rm -rf "$T"; mkdir -p "$T"
  case $SC in bs3*) BN=caplifive_vp2c_e7af48457;; *) BN=caplifive_vp2c_603528b8b;; esac
  OBB_SCEN=$SC OBB_RUN="python3 $D/supbare-stub.py" OBB_BUSY=true OBB_BOOK="bash $D/book-stub.sh" \
    bash "$D/../objb-bare.sh" 2 $BN "$T/bare"; local rc=$?
  echo "== $SC ($BN): objb-bare rc=$rc"
  sed 's/^/   /' "$T/bare/grade.txt" | head -16
}
main "$@"; exit $?
