#!/usr/bin/env python3
"""Our copy of PoisonCap's test/speedtest1.c for the Capstone side, with one dead function removed.

    make-speedtest1-capstone.py THEIR_SPEEDTEST1_C OUT_C

Their file does not compile for Capstone: it defines `static inline int cgetpoison(char *a)`, whose
inline assembly uses the CHERI operand constraint "C", which the Capstone compiler rejects
("invalid input constraint 'C' in asm"), although nothing calls the function (its one use is in a
comment). Headers or compiler flags cannot remove a function body, so this removes exactly that
definition, and checks: it occurs once, as written; after removal `cgetpoison` appears only on
comment lines; nothing else in the file differs (the output is the input minus those lines). Their
CHERI includes and their start-up cheri_revoke call stay, served by shim/ (a no-op revoke on Capstone,
which has no sweeping revoker). PoisonCap's own side uses their file unchanged.
"""
import difflib
from pathlib import Path
import sys

DEAD = '''static inline int cgetpoison(char *a){
       int poison = 0;
       asm volatile("cgetpoison %0,%1" : "=r"(poison) : "C"(a));
       return poison;
}
'''


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    text = src.read_text()
    if text.count(DEAD) != 1:
        sys.exit(f'{src}: the cgetpoison definition occurs {text.count(DEAD)} times, expected 1')
    result = text.replace(DEAD, '')
    left = [l for l in result.splitlines() if 'cgetpoison' in l]
    if any(not l.lstrip().startswith('//') for l in left):
        sys.exit(f'cgetpoison still used outside comments: {left}')
    removed = [l for l in difflib.ndiff(text.splitlines(True), result.splitlines(True)) if l[:2] in ('- ', '+ ')]
    if any(l.startswith('+ ') for l in removed) or len(removed) != DEAD.count('\n'):
        sys.exit(f'unexpected difference: {removed}')
    out.write_text(result)
    print(f'{out}: {src.name} minus the {DEAD.count(chr(10))}-line cgetpoison definition; '
          f'remaining mentions, all comments: {len(left)}')


if __name__ == '__main__':
    main()
