#!/bin/bash
# Stubbed dry run of the Plain chain: run-dryrun.sh <name> [pilot stub] [direct]
# Nothing reaches a board or the hub: CB_BUSY/SB_BUSY true, SB_BOOK book-stub.sh, SB_RUN sbrun-stub.sh (the REAL
# run-stage.py on BOARDVM_STUB_TRANSCRIPT). Each run gets its own stage copy (arming renames files in it), list and out.
# "direct" runs chain-board.sh itself over the whole list, without plain-chain.sh (the disarm defence).
main() {
  local name=${1:?name} pstub=${2:-} mode=${3:-wrapped}
  local B=/mnt/hdd/$USER/capstone/perf/T0b/board W=/mnt/hdd/$USER/capstone
  local R=$B/dryrun/$name
  rm -rf "$R"; mkdir -p "$R"
  python3 $B/make-plain-stages.py --out "$R/stages" > "$R/stages.out" || return 1
  sed "s|$B/stages|$R/stages|" $B/plain-b1.list > "$R/list"
  export PLAIN_DRY=$R CB_BUSY=true SB_BUSY=true CB_POLL_S=1 SB_BOOK="bash $B/dryrun/book-stub.sh" SB_RUN="bash $B/dryrun/sbrun-stub.sh"
  unset BOARDVM_STUB_TRANSCRIPT
  [ -n "$pstub" ] && export PLAIN_PILOT_STUB=$pstub || unset PLAIN_PILOT_STUB
  if [ "$mode" = direct ]; then
    taskset -c 16-31,48-63 bash $W/wt-spec/capstone/ports/spec-cpu2017/board/chain-board.sh 1 "$R/list" "$R/out"
  else
    taskset -c 16-31,48-63 bash $B/plain-chain.sh 1 "$R/list" "$R/out"
  fi
  echo "$name rc=$?"
}
main "$@"; exit
