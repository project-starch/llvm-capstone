#!/usr/bin/env python3
"""Per module: 16-bit (RVC) instruction count, and for every CSRUNV word (opcode 0x5b, funct3 1, funct7 0x24) whether the
NEXT instruction is csrw 0x5c1, zero (csrrw x0, 0x5c1, x0 = 0x5c101073). Raw words, never mnemonics."""
import re, subprocess, sys
OBJ = sys.argv[1]
for ko in sys.argv[2:]:
    out = subprocess.run([OBJ, '-d', ko], capture_output=True, text=True).stdout
    words = []
    for l in out.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){2,4})', l)          # llvm-objdump: bytes
        g = re.match(r'\s*([0-9a-f]+):\t([0-9a-f]{8}|[0-9a-f]{4})\s', l)      # GNU objdump: one hex word
        if m: words.append(bytes.fromhex(m.group(2).replace(' ', '')))
        elif g: words.append(int(g.group(2), 16).to_bytes(len(g.group(2)) // 2, 'little'))
    if not words: sys.exit(f"NO DATA: {ko} parsed to 0 instructions")
    rvc = sum(len(b) == 2 for b in words)
    w32 = [int.from_bytes(b, 'little') if len(b) == 4 else None for b in words]
    runv = [i for i, w in enumerate(w32) if w is not None and w & 0x7f == 0x5b and (w >> 12) & 7 == 1 and (w >> 25) == 0x24]
    zeroed = sum(1 for i in runv if i + 1 < len(w32) and w32[i + 1] == 0x5c101073)
    allz = sum(1 for w in w32 if w == 0x5c101073)
    print(f"{ko.split('/')[-3]:16} insns {len(words)} RVC {rvc} CSRUNV {len(runv)} followed-by-csrw-0x5c1-zero {zeroed} csrw-0x5c1-zero-total {allz}")
