#!/usr/bin/env bash
# The virtual image's two-segment layout (link.ld): text padded to the data segment's alignment, data
# starting exactly where text ends, for initialized globals of any alignment. The control links the
# same inputs with the text padding of one page only, which must fail once the alignment exceeds
# a page; without it a pass would not show that the check can fire.
#   CAPSTONE_LLVM_BIN=<llvm bin dir> bash test-link-layout.sh
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${CAPSTONE_LLVM_BIN:?LLVM bin directory with llvm-mc, ld.lld and llvm-readelf}
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
sed 's/ALIGN(MAX(4096, ALIGNOF(.data)))/ALIGN(4096)/' "$here/link.ld" > "$work/one-page.ld"
cmp -s "$here/link.ld" "$work/one-page.ld" && { echo "FAIL control script equals link.ld"; exit 1; }
for align in 64 4096 8192 16384 65536; do
    printf '.section .text.entry,"ax",@progbits\n.global _start\n_start: .long 0x13\n.section .gct,"a",@progbits\n.balign 16\n.section .data,"aw",@progbits\n.balign %s\n.long 1\n.section .bss,"aw",@nobits\n.zero 16\n' "$align" > "$work/in.s"
    "$bin/llvm-mc" -triple=capstone64 -filetype=obj "$work/in.s" -o "$work/in.o"
    "$bin/ld.lld" -T "$here/link.ld" -o "$work/out.elf" "$work/in.o"
    python3 - "$work/out.elf" "$bin/llvm-readelf" "$align" <<'PY'
import subprocess, sys
elf, readelf, align = sys.argv[1], sys.argv[2], int(sys.argv[3])
loads = [l.split() for l in subprocess.check_output([readelf, '-lW', elf], text=True).splitlines() if l.split()[:1] == ['LOAD']]
assert len(loads) == 2, loads
(t, d) = loads
t_vaddr, t_memsz, d_vaddr, d_off = int(t[2], 16), int(t[5], 16), int(d[2], 16), int(d[1], 16)
assert int(t[4], 16) == t_memsz, t  # text is whole file bytes: FileSiz == MemSiz
assert d_vaddr == t_vaddr + t_memsz, (hex(d_vaddr), hex(t_vaddr + t_memsz))
assert d_vaddr % max(4096, align) == 0 and d_off % 4096 == 0, (hex(d_vaddr), hex(d_off))
PY
    echo "ok   data alignment $align"
    if "$bin/ld.lld" -T "$work/one-page.ld" -o "$work/ctl.elf" "$work/in.o" 2> "$work/ctl.err"; then
        [ "$align" -le 4096 ] || { echo "FAIL control linked at alignment $align"; exit 1; }
    else
        [ "$align" -gt 4096 ] || { echo "FAIL control failed at alignment $align"; exit 1; }
        grep -q "data segment does not start where the text segment ends" "$work/ctl.err" || { echo "FAIL control failed for another reason"; cat "$work/ctl.err"; exit 1; }
    fi
done
echo "PASS"
