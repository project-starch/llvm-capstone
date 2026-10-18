#!/bin/bash
# The postgres Tier 3 arms on the board, one boot each, through their own runners with --board-kit (c-repros run-arm.py,
# mmgr run-defects.py), on the t3-freeze-nb builds made by build-postgres.sh. Usage: tier3-pg-arm.sh <kit> <out root>
# <pg build dir> <tag>... (tags: pgc pgm0 pgm1). Stops after a boot that does not complete. Ends by itself.
main() {
  set -u
  local K=${1:?kit} R=${2:?out root} PG=${3:?pg build dir}; shift 3
  local G=/tmp/capstone/wt-gate/capstone/bug-corpora/postgres LB=/tmp/capstone/vp2-prep/llvm-bin-v6 t d s arm
  for t in "$@"; do
    d=$R/$t
    [ -e "$d" ] && { echo "T3PG $t SKIPPED: $d exists"; continue; }
    mkdir -p "$d"; echo "T3PG $t BEGIN $(date +%T)"
    case $t in
      pgc) timeout 10800 python3 $G/c-repros/shared/run-arm.py --arm virtual-malloc --board-kit $K --bindir $PG/pgc-bin \
             --llvm-bin $LB --out $d/out --raw $d/raw > $d/run.log 2>&1 ;;
      pgm0|pgm1) s=OFF arm=virtual-malloc; [ $t = pgm1 ] && { s=ON; arm=virtual-pg-pools; }
             timeout 10800 python3 $G/mmgr-repros/shared/run-defects.py $d/out --board-kit $K --arm $arm \
               --hosted-build $PG/mmgr-cases-$s --controls-hosted-build $PG/mmgr-controls-$s --llvm-bin $LB --raw $d/raw \
               > $d/run.log 2>&1 ;;
      *) echo "T3PG $t: unknown tag"; return 2 ;;
    esac
    echo "T3PG $t END rc=$? $(date +%T) :: $(grep -a '^--- ' $d/run.log | tail -1)"
    # completion is the batch's own end marker inside the batch rung (boardvm writes the transcript as raw/guest/board.txt)
    grep -a -q '^VIRTUAL_STAGED_DONE' $d/raw/guest/board.txt 2>/dev/null || { echo "T3PG STOP after $t: the boot did not complete"; return 1; }
  done
  echo "T3PG DONE $(date +%T)"
}
main "$@"; exit $?
