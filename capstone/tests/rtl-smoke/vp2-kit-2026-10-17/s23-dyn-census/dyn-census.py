#!/usr/bin/env python3
"""Per image: every instruction decoder.sv at capstone-ariane 8522d72fe routes to CAPSTONE_DYN, counted by RAW 32-bit word
and field decode (opcode 0x5b custom-2; funct3 1 by funct7, funct3 3 LDC, funct3 4 STC), never by mnemonic. Instruction
boundaries come from llvm-objdump over executable sections, so 16-bit RVC words are skipped, not misread. Prints
'N instructions parsed' per image; exits 2 if an image parses to no instructions (no data)."""
import re, subprocess, sys, collections
OBJ = sys.argv[1]; images = sys.argv[2:]
R = {0x00: 'REVOKE', 0x02: 'TIGHTEN', 0x03: 'DELIN', 0x04: 'LCC', 0x06: 'SPLIT', 0x08: 'MREV', 0x0b: 'DROP',
     0x20: 'CALL', 0x21: 'RETURN', 0x22: 'CSSUPERVISE', 0x24: 'CSRUNV', 0x51: 'CDERIVE', 0x52: 'CREVOKE',
     0x53: 'CSRETIRE', 0x54: 'CSNODE'}
# The rev-node channel each op SENDS, from its own handler in capstone_dyn_unit.anvil at 8522d72fe (send rev_node_ep.*_req,
# comments excluded): REVOKE query+rev (:67,:72), DROP drop (:41), CREVOKE query x2+kill (:319,:323,:330), CSRETIRE retire
# (:625); CDERIVE query+derive (:268,:281), MREV deepen+mrev (:132,:146,:153), SPLIT query+init (:198,:220), DELIN delin (:808).
# The node-invalidating ones are those four; CDERIVE is NOT one (an earlier version of this table said it was).
INVALIDATES = {'REVOKE': 'rev', 'DROP': 'drop', 'CREVOKE': 'kill', 'CSRETIRE': 'retire'}
rc = 0
for img in images:
    out = subprocess.run([OBJ, '-d', '--no-print-imm-hex', img], capture_output=True, text=True).stdout
    n = 0; n32 = 0; c = collections.Counter()
    for line in out.splitlines():
        m = re.match(r'\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} ){2,4})', line)
        if not m: continue
        b = bytes.fromhex(m.group(1).replace(' ', ''))
        n += 1
        if len(b) != 4: continue
        n32 += 1
        w = int.from_bytes(b, 'little')
        if w & 0x7f != 0x5b: continue
        f3 = (w >> 12) & 7; f7 = w >> 25
        if f3 == 1 and f7 in R: c[R[f7]] += 1
        elif f3 == 3: c['LDC'] += 1
        elif f3 == 4: c['STC'] += 1
    print(f"== {img.split('/')[-1]}: {n} instructions parsed ({n32} 32-bit)")
    for k in sorted(c, key=lambda k: -c[k]):
        print(f"   {k:12} {c[k]:>7}  {'node-invalidating via ' + INVALIDATES[k] if k in INVALIDATES else ''}")
    if n == 0: rc = 2
sys.exit(rc)
