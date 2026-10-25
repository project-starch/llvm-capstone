#!/bin/bash
# monitor-variant-gate.sh <old capstone-sbi commit> <new commit> <fresh out dir>: the monitor's generated code, old against
# new, under three configurations, so a monitor change states exactly which builds it moves.
#   fpga  the board's defines (vp2 Tier 1: SUPERVISED_CALL, SUPERVISOR_CSR_EVENTS, SUPERVISE_QUIET, VP2_CSMINT) in the
#         FPGA wrapper (caplifive-opensbi 1269c77);
#   qemu  the kit's QEMU defines (DEBUG_ENABLE, SUPERVISED_CALL) in OpenSBI cf344cf3;
#   qinj  the same plus CAPSTONE_S_TRAP_INJECT.
# For each: capstone-c over the wrapper's sbi_capstone_dom.c (the C monitor) and cpp over sbi_capstone.S (the trap
# entry), with the monitor archived into lib/sbi/capstone-sbi. Prints the differing line count of each pair; with -v
# the diffs. CONFIGS="qemu qinj" limits the set (a commit older than the FPGA wrapper cannot build `fpga`). Exit 0 when
# every pair was produced, 2 on no data (an empty output, a compiler failure), never a zero.
main() {
  set -u
  local OLD=${1:?old commit} NEW=${2:?new commit} O=${3:?fresh out dir} V=${4:-}
  local R=$HOME/dev/llvm-capstone/capstone SBI=/tmp/capstone/csbi-land WRAP=/tmp/capstone/wt-wrap-exact
  local CC=$R/capstone-c/target/debug/capstone-c XC=$R/caplifive-buildroot/build/host/bin/riscv64-buildroot-linux-gnu-
  [ -e "$O" ] && { echo "STOP: $O exists"; return 2; }
  local cfg rev t defs
  local CFGS=${CONFIGS:-fpga qemu qinj} TS=qemu
  case " $CFGS " in *" fpga "*) TS="fpga qemu";; esac
  for rev in old new; do
    local c; [ $rev = old ] && c=$OLD || c=$NEW
    for t in $TS; do
      mkdir -p "$O/$rev-$t/lib/sbi/capstone-sbi"
      if [ $t = fpga ]; then git -C $WRAP archive 1269c77 | tar -x -C "$O/$rev-$t" || return 2
      else git -C $R/caplifive-buildroot/components/opensbi archive cf344cf3884f | tar -x -C "$O/$rev-$t" || return 2; fi
      git -C $SBI archive "$c" | tar -x -C "$O/$rev-$t/lib/sbi/capstone-sbi" || return 2
    done
  done
  local worst=0
  for cfg in $CFGS; do
    case $cfg in
      fpga) t=fpga; defs="-DCAPSTONE_TARGET_FPGA -DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT";;
      qemu) t=qemu; defs="-DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE -DCAPSTONE_SUPERVISED_CALL";;
      qinj) t=qemu; defs="-DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE -DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_S_TRAP_INJECT";;
    esac
    for rev in old new; do
      local T="$O/$rev-$t"
      $CC --abi capstone "$T/lib/sbi/sbi_capstone_dom.c" -- -I"$T/lib/sbi/capstone-sbi" -D__riscv_xlen=64 $defs \
        > "$O/$rev-$cfg.c.S" 2> "$O/$rev-$cfg.c.err" || { echo "NO DATA: capstone-c failed on $rev-$cfg"; return 2; }
      ${XC}cpp -P -I"$T/lib/sbi/capstone-sbi" -I"$T/include" $defs "$T/lib/sbi/capstone-sbi/sbi_capstone.S" \
        > "$O/$rev-$cfg.entry.i" 2> "$O/$rev-$cfg.entry.err" || { echo "NO DATA: cpp failed on $rev-$cfg"; return 2; }
      [ -s "$O/$rev-$cfg.c.S" ] && [ -s "$O/$rev-$cfg.entry.i" ] || { echo "NO DATA: empty output for $rev-$cfg"; return 2; }
    done
    local dc de
    dc=$(diff "$O/old-$cfg.c.S" "$O/new-$cfg.c.S" | grep -c '^[<>]')
    de=$(diff "$O/old-$cfg.entry.i" "$O/new-$cfg.entry.i" | grep -c '^[<>]')
    echo "$cfg: monitor C $dc differing lines ($(wc -l < "$O/new-$cfg.c.S") lines), entry $de differing lines ($(wc -l < "$O/new-$cfg.entry.i") lines)"
    if [ -n "$V" ]; then diff "$O/old-$cfg.c.S" "$O/new-$cfg.c.S"; diff "$O/old-$cfg.entry.i" "$O/new-$cfg.entry.i"; fi
  done
  return 0
}
main "$@"; exit $?
