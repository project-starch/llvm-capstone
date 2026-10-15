#!/bin/bash
# R-55 (regions) codegen gate: wrapper+monitor at their committed parents (orig: d7489e3 + 7361e1c) against the edited
# worktrees (fix), through capstone-c for QEMU and FPGA, plain and with the supervision defines. QEMU must be
# byte-identical; FPGA must compile; prints the FPGA functions that changed.
main() {
  set -u
  local W=${GW:-/tmp/capstone/wt-wrap-exact} MW=${GM:-/tmp/capstone/wt-mon-exact} WB=${GWB:-d7489e3} MB=${GMB:-7361e1c} CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
  local T=${GT:-/tmp/capstone/r55-table/gate}; rm -rf $T; mkdir -p $T
  for v in orig fix; do
    mkdir -p $T/$v
    if [ $v = orig ]; then git -C $W archive $WB | tar -x -C $T/$v; else (cd $W && git ls-files -z | xargs -0 tar -c) | tar -x -C $T/$v; fi
    rm -rf $T/$v/lib/sbi/capstone-sbi; mkdir -p $T/$v/lib/sbi/capstone-sbi
    if [ $v = orig ]; then git -C $MW archive $MB | tar -x -C $T/$v/lib/sbi/capstone-sbi; else (cd $MW && git ls-files -z | xargs -0 tar -c) | tar -x -C $T/$v/lib/sbi/capstone-sbi; fi
  done
  local rc=0
  for defs in "plain:" "sup:-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET"; do
    local name=${defs%%:*} D=${defs#*:}
    for v in orig fix; do for tgt in FPGA QEMU; do
      local d="-DCAPSTONE_TARGET_$tgt"; [ $tgt = QEMU ] && d="$d -DCAPSTONE_DEBUG_ENABLE"
      "$CC" --abi capstone $T/$v/lib/sbi/sbi_capstone_dom.c -- -I$T/$v/lib/sbi/capstone-sbi -D__riscv_xlen=64 $d $D \
        > $T/$v.$name.$tgt.S 2> $T/$v.$name.$tgt.err || { echo "capstone-c FAILED $v $name $tgt"; tail -5 $T/$v.$name.$tgt.err; rc=1; }
    done; done
    if cmp -s $T/orig.$name.QEMU.S $T/fix.$name.QEMU.S; then echo "$name QEMU: byte-identical ($(wc -l < $T/fix.$name.QEMU.S) lines)"; else echo "$name QEMU: DIFFERS"; rc=1; fi
    if cmp -s $T/orig.$name.FPGA.S $T/fix.$name.FPGA.S; then echo "$name FPGA: identical -- the change did not reach the output"; rc=1; else echo "$name FPGA: changed ($(wc -l < $T/orig.$name.FPGA.S) -> $(wc -l < $T/fix.$name.FPGA.S) lines)"; fi
  done
  echo "gate dir: $T"
  return $rc
}
main "$@"; exit $?
