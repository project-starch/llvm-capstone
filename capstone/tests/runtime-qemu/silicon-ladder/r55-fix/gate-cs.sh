#!/bin/bash
# R-55 codegen gate: <wrapper worktree> <monitor include dir> "<extra defs>". Compiles sbi_capstone_dom.c patched (the
# worktree's) and unpatched (git HEAD's) for FPGA and for QEMU; QEMU must be byte-identical, FPGA must differ.
main() {
  set -u
  local W=$1 INC=$2 DEFS=$3 CC=$HOME/dev/llvm-capstone/capstone/capstone-c/target/debug/capstone-c
  local T; T=$(mktemp -d /tmp/capstone/r55-fix/gate.XXXX)
  git -C "$W" show HEAD:lib/sbi/sbi_capstone_dom.c > "$T/orig.c"; cp "$W/lib/sbi/sbi_capstone_dom.c" "$T/fix.c"
  # the wrapper #includes "capstone-sbi/sbi_capstone.c": compile in place so that include resolves the same way
  for v in orig fix; do
    for tgt in FPGA QEMU; do
      local d="-DCAPSTONE_TARGET_$tgt"; [ $tgt = QEMU ] && d="$d -DCAPSTONE_DEBUG_ENABLE"
      cp "$T/$v.c" "$W/lib/sbi/sbi_capstone_dom.c"
      "$CC" --abi capstone "$W/lib/sbi/sbi_capstone_dom.c" -- -I"$INC" -D__riscv_xlen=64 $d $DEFS > "$T/$v.$tgt.S" 2> "$T/$v.$tgt.err" \
        || { echo "capstone-c FAILED $v $tgt"; cp "$T/fix.c" "$W/lib/sbi/sbi_capstone_dom.c"; return 1; }
    done
  done
  cp "$T/fix.c" "$W/lib/sbi/sbi_capstone_dom.c"
  cmp -s "$T/orig.QEMU.S" "$T/fix.QEMU.S" && echo "QEMU: byte-identical ($(wc -l < $T/fix.QEMU.S) lines)" || { echo "QEMU: DIFFERS"; return 1; }
  cmp -s "$T/orig.FPGA.S" "$T/fix.FPGA.S" && { echo "FPGA: identical -- the patch did not reach the output"; return 1; }
  echo "FPGA: differs; changed lines:"; diff "$T/orig.FPGA.S" "$T/fix.FPGA.S" | grep '^[<>]' | head -20
  python3 - "$T/orig.FPGA.S" "$T/fix.FPGA.S" <<'PY'
import sys, difflib
a, b = (open(f).read().split("\n") for f in sys.argv[1:3])
def owner(lines, i):
    for j in range(i, -1, -1):
        l = lines[j]
        if l and not l[0].isspace() and l.endswith(":") and not l.startswith("."):
            return l[:-1]
    return "?"
fns = set()
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
    if tag != "equal":
        fns.add(owner(b, max(j1 - 1, 0)))
print("changed functions:", sorted(fns))
PY
  echo "gate dir: $T"
}
main "$@"; exit $?
