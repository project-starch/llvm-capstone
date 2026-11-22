#!/usr/bin/env python3
"""active-lines.py <file.S> DEF... : the lines active under the given defines (#ifdef/#ifndef/#if defined(..)||../#elif/
#else/#endif), directives other than #define dropped; #define lines kept (their bodies may name registers)."""
import re, sys
f, defs = sys.argv[1], set(a[2:] if a.startswith('-D') else a for a in sys.argv[2:])
def ev(expr):
    e = re.sub(r'defined\s*\(\s*(\w+)\s*\)', lambda m: '1' if m.group(1) in defs else '0', expr)
    e = e.replace('||', ' or ').replace('&&', ' and ').replace('!', ' not ')
    assert re.fullmatch(r'[01 ()orandnt]*', e), expr
    return bool(eval(e))
stack = []  # (active_parent, taken, current)
active = True
out = []
for l in open(f):
    s = l.strip()
    m = re.match(r'#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(.*)', s)
    if m:
        d, rest = m.group(1), m.group(2).split('//')[0].strip()
        if d in ('ifdef', 'ifndef', 'if'):
            c = (rest in defs) if d == 'ifdef' else (rest not in defs) if d == 'ifndef' else ev(rest)
            stack.append((active, c)); active = active and c
        elif d == 'elif':
            par, taken = stack[-1]; c = (not taken) and ev(rest); stack[-1] = (par, taken or c); active = par and c
        elif d == 'else':
            par, taken = stack[-1]; stack[-1] = (par, True); active = par and not taken
        elif d == 'endif':
            active = stack.pop()[0]
        continue
    if active:
        out.append(l)
assert not stack, 'unbalanced'
sys.stdout.write(''.join(out))
