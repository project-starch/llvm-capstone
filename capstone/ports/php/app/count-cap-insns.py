#!/usr/bin/env python3
"""Count CDERIVE and CREVOKE in a disassembly, by ENCODING rather than by mnemonic.

WHY NOT grep FOR A MNEMONIC. The build's gate used to be `objdump -d | grep -c '\\brevoke\\b'`,
and it was load-bearing: the nested layer is a RUNTIME mode, so an image that cannot revoke at all would
make every temporal column read the same and all three arms would look like "nothing revoked".

That grep silently stopped meaning anything on 2026-10-26, when the nested layer moved off the deleted
Sublet loan onto capability.h's CDERIVE/CREVOKE. This llvm-objdump does not know either
instruction -- both print as `<unknown>` -- so the mnemonic is simply absent from a correct
image, and the gate failed closed. It failing closed is the only reason this was noticed.

So match the encoding instead. Both are R-type on opcode 0x5b with funct3 = 1, distinguished by
funct7: 0x51 CDERIVE (issue a child), 0x52 CREVOKE (end a child's subtree). Reading them off the
bytes cannot be invalidated by a disassembler that has not been taught the mnemonics.

Usage:  llvm-objdump -d IMAGE | count-cap-insns.py        -> "derive=N revoke=M"
        ... | count-cap-insns.py --require                -> exit 1 unless BOTH are present
"""
import re
import sys

LINE = re.compile(r"^\s+[0-9a-f]+:\s+((?:[0-9a-f]{2} ){4})")
CDERIVE, CREVOKE = 0x51, 0x52


def count(stream):
    derive = revoke = 0
    for line in stream:
        m = LINE.match(line)
        if not m:
            continue
        b = [int(x, 16) for x in m.group(1).split()]
        word = b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24
        if (word & 0x7F) != 0x5B or (word >> 12) & 7 != 1:
            continue
        funct7 = word >> 25
        if funct7 == CDERIVE:
            derive += 1
        elif funct7 == CREVOKE:
            revoke += 1
    return derive, revoke


def main():
    derive, revoke = count(sys.stdin)
    print(f"derive={derive} revoke={revoke}")
    if "--require" in sys.argv[1:]:
        # BOTH, not either: an image that can issue but not revoke is exactly the shape that
        # would report a clean sweep with no revocation having happened.
        if derive == 0 or revoke == 0:
            print("the image is missing a capability instruction the nested layer needs",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
