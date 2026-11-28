"""grade-linux3.py 3|4 UART_TXT : grade boots 3 and 4 of b5b2-linux-2026-11-25 (runtime/b5b2-linux 90f2f5bc5e83,
PREREG.md Amendment 2) from the run-scoped capture, first boot only. Result lines only (never the Linux version line).
Boot 3 (keep-tags): 'Capstone: register tags kept ...' after 'OpenSBI v'; MEDELEG 0x000000007f00b109; captag, sublet and
  '.dom loader registered' before 'Run /init'; then in order TRAPTEST up, 'regs status 0', 'regs-strip23 status 23',
  'enter status 200', 'loop alive', 'done'. The line 'STRIP CONTROL 23' is printed only when the strip line reads 23:
  boot 4's gate keys on it.
Boot 4 (flag off): no 'Capstone:' line; MEDELEG 0x000000000000b109; 'Run /init' reached; regs status NOT 0, counted as
  the contract mattering when it is 1..31, 100+c, a signal status (>128), or a kernel failure after /init; 'regs status
  0' is printed as REFUTES (the firmware's necessity on silicon).
Missing or empty capture: ERROR, exit 2. Exit 0 only when every prediction holds."""
import re, sys
if len(sys.argv) != 3 or sys.argv[1] not in ('3', '4'): sys.exit(__doc__)
B = sys.argv[1]
try: raw = open(sys.argv[2], errors='replace').read()
except OSError as e: print(f'ERROR no capture: {e}'); sys.exit(2)
if not raw.strip(): print(f'ERROR empty capture {sys.argv[2]}'); sys.exit(2)
L = [l.rstrip('\r') for l in raw.replace('\r\n', '\n').split('\n')]
ob = [i for i, l in enumerate(L) if 'OpenSBI v' in l]
if len(ob) > 1: print(f'note: {len(ob)} OpenSBI banners; graded up to line {ob[1]}'); L = L[:ob[1]]
def idx(s, start=0, kern=False):
    for i in range(start, len(L)):
        if s in L[i] and not (kern and L[i].startswith('TRAPTEST:')): return i
    return None
strip_ts = lambda l: re.sub(r'^\[ *[0-9.]+\] ', '', l).strip()
res = {}
print(f'boot {B}; capture: {len(L)} lines, {len(raw)} chars')
o = idx('OpenSBI v')
if o is None: print(f'VOID: no OpenSBI output after the load\nB5B2-LINUX-{B} VOID'); sys.exit(1)
cap = idx('Capstone:', o)
if B == '3':
    res['P1'] = cap is not None and 'register tags kept across M traps, causes 24-30 delegated' in L[cap]
    print(f"P1 {'AS PREDICTED' if res['P1'] else 'MISS'}: Capstone line {strip_ts(L[cap])[:90] if cap is not None else 'absent'}")
else:
    res['P1'] = cap is None
    print(f"P1 {'AS PREDICTED' if res['P1'] else 'MISS'}: Capstone line {'absent' if cap is None else 'PRESENT: ' + strip_ts(L[cap])[:90]}")
md = idx('Boot HART MEDELEG'); m = re.search(r':\s*(0x[0-9a-fA-F]+)', L[md]) if md is not None else None
mv = m.group(1).lower() if m else None
res['P2'] = mv == ('0x000000007f00b109' if B == '3' else '0x000000000000b109')
print(f"P2 {'AS PREDICTED' if res['P2'] else 'MISS'}: Boot HART MEDELEG = {mv}")
lx = idx('Linux version'); run = idx('Run /init as init process')
pre = run if run is not None else len(L)
bad = [i for i in range(lx, pre) if re.search(r'Oops|Unable to handle|Kernel panic|BUG:|scause|epc *:|badaddr', L[i])] if lx is not None else []
if B == '3':
    k = [idx(s, lx or 0, True) for s in ('captag: bulk=0', 'sublet: 63-bit Sublet format, exact bounds 0', 'sublet: .dom loader registered')]
    res['P3'] = lx is not None and run is not None and all(i is not None and i < run for i in k) and not bad
    print(f"P3 {'AS PREDICTED' if res['P3'] else 'MISS'}: Linux banner {'present' if lx is not None else 'absent'}; Run /init "
          f"{'at line ' + str(run + 1) if run is not None else 'absent'}; kernel lines before it {[i is not None and run is not None and i < run for i in k]}")
else:
    res['P3'] = lx is not None and run is not None and not bad
    print(f"P3 {'AS PREDICTED' if res['P3'] else 'MISS'}: Linux banner {'present' if lx is not None else 'absent'}; Run /init "
          f"{'at line ' + str(run + 1) if run is not None else 'absent'}")
for i in bad[:30]: print(f'  before /init: {strip_ts(L[i])[:150]}')
if run is None:
    print(f"\nB5B2-LINUX-{B} " + ' '.join(f"{k}={'ok' if v else 'MISS'}" for k, v in res.items()) + ' P4=NO-READING'); sys.exit(1)
after = L[run + 1:]
for l in after:
    if l.startswith('TRAPTEST:') and not l.startswith('TRAPTEST: dmesg') or re.search(r'test loader|Oops|Unable to handle|Kernel panic|BUG:|segfault|reboot:|epc *:|scause', l):
        print(f'  {strip_ts(l)[:150]}')
st = {}
for l in after:
    m = re.match(r'TRAPTEST: (regs|regs-strip23|enter) status (\d+)$', l.strip())
    if m and m.group(1) not in st: st[m.group(1)] = int(m.group(2))
tt = [l.strip() for l in after if l.startswith('TRAPTEST:') and not l.startswith('TRAPTEST: dmesg')]
kfail = [l for l in after if re.search(r'Oops|Unable to handle|Kernel panic|BUG:', l) and 'Attempted to kill init' not in l]
if not tt:
    print('P4 NO READING: no TRAPTEST line after /init' + (f'; {len(kfail)} kernel failure lines' if kfail else ''))
    res['P4'] = B == '4' and bool(kfail)
    if B == '4' and kfail: print('P4 (boot 4): a kernel failure after /init counts as the contract mattering')
elif B == '3':
    want = ['TRAPTEST: regs status 0', 'TRAPTEST: regs-strip23 status 23', 'TRAPTEST: enter status 200', 'TRAPTEST: loop alive', 'TRAPTEST: done']
    res['P4'] = len(tt) == 6 and tt[0].startswith('TRAPTEST: up') and tt[1:] == want
    print(f"P4 {'AS PREDICTED' if res['P4'] else 'MISS'}: statuses {st}")
    if st.get('regs-strip23') == 23: print('STRIP CONTROL 23')
else:
    r = st.get('regs')
    if r is None: res['P4'] = bool(kfail); print(f"P4 {'AS PREDICTED (a kernel failure after /init)' if kfail else 'NO READING'}: no regs status line")
    elif r == 0: res['P4'] = False; print('P4 REFUTES: regs status 0 under the flag-off firmware')
    else:
        kind = ('lost register x%d' % r) if 1 <= r <= 31 else ('hook fault, cause %d' % (r - 100)) if 100 <= r < 128 else ('signal %d' % (r - 128)) if r > 128 else 'setup failure' if 90 <= r <= 95 else 'other'
        res['P4'] = kind not in ('setup failure', 'other')
        print(f"P4 {'AS PREDICTED' if res['P4'] else 'MISS'}: regs status {r} ({kind}); statuses {st}")
print(f"\nB5B2-LINUX-{B} " + ' '.join(f"{k}={'ok' if v else 'MISS'}" for k, v in res.items()))
sys.exit(0 if all(res.values()) and len(res) == 4 else 1)
