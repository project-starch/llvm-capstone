#!/bin/bash
# R-55 runtime-half codegen gate: <wrapper worktree> <monitor worktree> "<defs>". The wrapper's committed HEAD with the
# monitor's committed HEAD (orig) against the monitor's working tree (fix); FPGA and QEMU each.
main() {
  set -u
  local W=$1 MW=$2 DEFS=$3 CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
  local T; T=$(mktemp -d /tmp/capstone/r55-fix/gmon.XXXX)
  for v in orig fix; do
    mkdir -p $T/$v; git -C $W archive HEAD | tar -x -C $T/$v; rm -rf $T/$v/lib/sbi/capstone-sbi; mkdir -p $T/$v/lib/sbi/capstone-sbi
    git -C $MW archive HEAD | tar -x -C $T/$v/lib/sbi/capstone-sbi
  done
  cp $MW/sbi_capstone.c $T/fix/lib/sbi/capstone-sbi/sbi_capstone.c
  for v in orig fix; do for tgt in FPGA QEMU; do
    local d="-DCAPSTONE_TARGET_$tgt"; [ $tgt = QEMU ] && d="$d -DCAPSTONE_DEBUG_ENABLE"
    "$CC" --abi capstone $T/$v/lib/sbi/sbi_capstone_dom.c -- -I$T/$v/lib/sbi/capstone-sbi -D__riscv_xlen=64 $d $DEFS > $T/$v.$tgt.S 2> $T/$v.$tgt.err \
      || { echo "capstone-c FAILED $v $tgt"; tail -3 $T/$v.$tgt.err; return 1; }
  done; done
  cmp -s $T/orig.QEMU.S $T/fix.QEMU.S && echo "QEMU: byte-identical ($(wc -l < $T/fix.QEMU.S) lines)" || { echo "QEMU: DIFFERS"; return 1; }
  cmp -s $T/orig.FPGA.S $T/fix.FPGA.S && { echo "FPGA: identical -- the patch did not reach the output"; return 1; }
  python3 - $T/orig.FPGA.S $T/fix.FPGA.S <<'PY'
import sys, difflib, re
a, b = (open(f).read().split("\n") for f in sys.argv[1:3])
def owner(lines, i):
    for j in range(i, -1, -1):
        l = lines[j]
        if l and not l[0].isspace() and l.endswith(":") and not l.startswith("."):
            return re.sub(r"\.\d+$|\.ret$", "", l[:-1]).lstrip("_")
    return "?"
fns, added_scc = set(), 0
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if tag != "equal":
        fns.add(owner(b, max(j1 - 1, 0)))
        added_scc += sum(1 for l in b[j1:j2] if "scc(" in l) - sum(1 for l in a[i1:i2] if "scc(" in l)
print("FPGA changed functions:", sorted(fns), " net scc added:", added_scc)
PY
  echo "gate dir: $T"
}
main "$@"; exit $?
