#!/usr/bin/env python3
"""scan_illegal.py IMAGE...: count Capstone encodings in an ELF's executable sections, classified by the B5b-2
decoder (capstone-ariane 904cd9c6b, core/decoder.sv under CAPSTONE_SUBLET_FMT) for code running in protected U.

Raw encodings, a linear RVC-aware sweep of every SHF_EXECINSTR section (16-bit parcels with low bits != 11 are
compressed). No disassembler is involved, so an op the disassembler cannot name is still counted. Prints one line
per image with every class's count, the ILLEGAL total, and the words swept; exits 2 if any image has no executable
bytes (no data is an error, not a zero), 1 if any image has an ILLEGAL encoding, else 0.

The classes, by the decoder's own case arms:
  custom-2 0x5b funct3 0 SHRINKTO, 2 CINCOFFSETIMM, 3 LDC, 4 STC, 6 CBNZ: legal (LDC/STC need protected U)
  custom-2 funct3 5 CJALR, 7 CCSRRW: illegal
  custom-2 funct3 1, by funct7: 0x01 SHRINK, 0x04 LCC, 0x05 SCC, 0x0a MOVC, 0x0c CINCOFFSET, 0x51 CDERIVE legal;
    0x52 CREVOKE legal only with rd = rs2 = x0; 0x54 CRENEW legal only with rs2 = x0; 0x50 CSMINT illegal in U;
    0x00 REVOKE, 0x02 TIGHTEN, 0x03 DELIN, 0x06 SPLIT, 0x07 SEAL, 0x08 MREV, 0x09 INIT, 0x0b DROP, 0x0d CAPENTER,
    0x20 CALL, 0x21 RETURN, 0x22 CSSUPERVISE, 0x24 CSRUNV, 0x53 CSRETIRE and every other funct7: illegal
  custom-3 0x7b (the debug family): illegal; custom-0 0x0b (debug print, CLDRAW/CSTRAW, CPTAG): illegal in U;
  custom-1 0x2b: no decoder arm, illegal
  SYSTEM csr ops on the supervisor CSRs 0x7c3, 0x7c4, 0xfc0-0xfc4: illegal (B5b-2 step 8a)"""
import collections, struct, sys

F7 = {0x00: 'REVOKE', 0x01: 'SHRINK', 0x02: 'TIGHTEN', 0x03: 'DELIN', 0x04: 'LCC', 0x05: 'SCC', 0x06: 'SPLIT',
      0x07: 'SEAL', 0x08: 'MREV', 0x09: 'INIT', 0x0a: 'MOVC', 0x0b: 'DROP', 0x0c: 'CINCOFFSET', 0x0d: 'CAPENTER',
      0x20: 'CALL', 0x21: 'RETURN', 0x22: 'CSSUPERVISE', 0x24: 'CSRUNV', 0x50: 'CSMINT', 0x51: 'CDERIVE',
      0x52: 'CREVOKE', 0x53: 'CSRETIRE', 0x54: 'CRENEW'}
LEGAL_F7 = {'SHRINK', 'LCC', 'SCC', 'MOVC', 'CINCOFFSET', 'CDERIVE'}
F3 = {0: ('SHRINKTO', True), 2: ('CINCOFFSETIMM', True), 3: ('LDC', True), 4: ('STC', True), 5: ('CJALR', False),
      6: ('CBNZ', True), 7: ('CCSRRW', False)}
SUP_CSRS = {0x7c3, 0x7c4, 0xfc0, 0xfc1, 0xfc2, 0xfc3, 0xfc4}


def classify(w):
    """(name, legal) for a Capstone-relevant 32-bit word, or None"""
    op = w & 0x7f; f3 = (w >> 12) & 7; rd = (w >> 7) & 31; rs2 = (w >> 20) & 31; f7 = w >> 25
    if op == 0x5b:
        if f3 != 1:
            return F3[f3]
        name = F7.get(f7, 'custom2-f7-%#x' % f7)
        if name == 'CREVOKE':
            return (name, True) if rd == 0 and rs2 == 0 else ('CREVOKE-two-operand', False)
        if name == 'CRENEW':
            return (name, True) if rs2 == 0 else ('CRENEW-rs2', False)
        return name, name in LEGAL_F7
    if op == 0x7b:
        return 'custom3', False
    if op == 0x0b:
        return 'custom0', False
    if op == 0x2b:
        return 'custom1', False
    if op == 0x73 and f3 in (1, 2, 3, 5, 6, 7) and (w >> 20) in SUP_CSRS:
        return 'supervisor-csr-%#x' % (w >> 20), False
    return None


def exec_sections(data):
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        raise ValueError('not a little-endian ELF64')
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', data, 0x3a)
    for i in range(shnum):
        _, typ, flags, _, off, size = struct.unpack_from('<IIQQQQ', data, shoff + i * shentsize)
        if typ == 1 and flags & 4 and size:  # SHT_PROGBITS, SHF_EXECINSTR
            yield data[off:off + size]


def scan(path):
    data = open(path, 'rb').read()
    counts = collections.Counter(); words = 0; parcels16 = 0
    for sec in exec_sections(data):
        i = 0
        while i + 2 <= len(sec):
            lo, = struct.unpack_from('<H', sec, i)
            if lo & 3 != 3:
                parcels16 += 1; i += 2; continue
            if i + 4 > len(sec):
                break
            w, = struct.unpack_from('<I', sec, i); i += 4; words += 1
            c = classify(w)
            if c:
                counts[c] += 1
    return counts, words, parcels16


def main():
    status = 0
    for path in sys.argv[1:]:
        counts, words, parcels16 = scan(path)
        if words + parcels16 == 0:
            print(f'{path}: NO-DATA (no executable bytes)'); status = 2; continue
        illegal = sum(n for (name, legal), n in counts.items() if not legal)
        parts = ' '.join(f'{name}{"" if legal else "!"}={n}' for (name, legal), n in sorted(counts.items()))
        print(f'{path}: words32={words} parcels16={parcels16} ILLEGAL={illegal} DELIN={counts[("DELIN", False)]} {parts}')
        if illegal and status == 0:
            status = 1
    sys.exit(status)


if __name__ == '__main__':
    main()
