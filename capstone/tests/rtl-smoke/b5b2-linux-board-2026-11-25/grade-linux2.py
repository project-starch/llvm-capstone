"""grade-linux2.py UART_TXT : grade boot 2 of b5b2-linux-2026-11-25 (runtime/b5b2-linux 05ac6828a092, PREREG.md P1-P4
as amended: P3 adds the kernel line 'sublet: .dom loader registered' before /init). Derived from grade-linux.py
from the run-scoped capture. Prints result lines only (never the Linux version line, which names the build account).
 P1 'OpenSBI v' then 'Capstone: register tags kept across M traps, causes 24-30 delegated';
 P2 'Boot HART MEDELEG' = 0x000000007f00b109;
 P3 'Run /init as init process', with kernel lines 'sublet: 63-bit Sublet format, exact bounds 0' and 'captag: bulk=0'
    before it (TRAPTEST dmesg echoes are not counted);
 P4 in order: 'TRAPTEST: up', 'TRAPTEST: regs status 91', 'TRAPTEST: enter status 1', 'TRAPTEST: loop alive',
    'TRAPTEST: done' (a known failure, matched against the QEMU twin; not a pass).
Void rules (PREREG): no OpenSBI output -> VOID for P2-P4; a banner but no Linux banner -> payload/FDT/kernel; an oops,
panic or trap before /init -> its lines printed, P4 VOID. An empty or missing capture is an ERROR (exit 2)."""
import re, sys
if len(sys.argv) != 2: sys.exit(__doc__)
try: raw = open(sys.argv[1], errors='replace').read()
except OSError as e: print(f'ERROR no capture: {e}'); sys.exit(2)
L = [l.rstrip('\r') for l in raw.replace('\r\n', '\n').split('\n')]
if not raw.strip(): print(f'ERROR empty capture {sys.argv[1]}'); sys.exit(2)
ob0 = [i for i, l in enumerate(L) if 'OpenSBI v' in l]
if len(ob0) > 1: print(f'note: {len(ob0)} OpenSBI banners; graded up to line {ob0[1]} (the first boot only)'); L = L[:ob0[1]]
def idx(s, start=0, pred=None):
    for i in range(start, len(L)):
        if s in L[i] and (pred is None or pred(L[i])): return i
    return None
kern = lambda l: not l.startswith('TRAPTEST:')
res = {}
ob = idx('OpenSBI v'); cap = idx('Capstone: register tags kept across M traps, causes 24-30 delegated')
lx = idx('Linux version'); run = idx('Run /init as init process')
pre = run if run is not None else len(L)
bad = [i for i in range(lx or 0, pre) if re.search(r'Oops|Unable to handle|Kernel panic|BUG:|scause|epc *:|badaddr', L[i])] if lx is not None else []
print(f'capture: {len(L)} lines, {len(raw)} chars')
if ob is None:
    print('VOID for P2-P4: no OpenSBI output after the load (the load or the firmware)')
    print('P1 MISS'); sys.exit(1)
print(f'line {ob + 1}: {L[ob].strip()[:60]}')
res['P1'] = cap is not None and cap > ob
print(f"P1 {'AS PREDICTED' if res['P1'] else 'MISS'}: Capstone keep-tags line {'at line ' + str(cap + 1) if cap is not None else 'absent'}")
md = idx('Boot HART MEDELEG')
mv = re.search(r':\s*(0x[0-9a-fA-F]+)', L[md]).group(1) if md is not None and re.search(r':\s*(0x[0-9a-fA-F]+)', L[md]) else None
res['P2'] = mv is not None and mv.lower() == '0x000000007f00b109'
print(f"P2 {'AS PREDICTED' if res['P2'] else 'MISS'}: Boot HART MEDELEG = {mv}")
for k in ('Boot HART MIDELEG', 'Boot HART ISA', 'Platform Name'):
    i = idx(k)
    if i is not None: print(f'  {L[i].strip()[:100]}')
if lx is None:
    print('P3 MISS: an OpenSBI banner but no Linux banner (the payload, FDT or kernel); P4 VOID'); sys.exit(1)
print(f'line {lx + 1}: Linux banner present')
sb = idx('sublet: 63-bit Sublet format, exact bounds 0', lx, kern); ct = idx('captag: bulk=0', lx, kern)
rg = idx('sublet: .dom loader registered', lx, kern)
res['P3'] = run is not None and sb is not None and sb < run and ct is not None and ct < run and rg is not None and rg < run
print(f"P3 {'AS PREDICTED' if res['P3'] else 'MISS'}: Run /init {'at line ' + str(run + 1) if run is not None else 'absent'}; "
      f"sublet line {'before it' if sb is not None and run is not None and sb < run else 'absent before it'}; "
      f"captag line {'before it' if ct is not None and run is not None and ct < run else 'absent before it'}; "
      f"loader-registered line {'before it' if rg is not None and run is not None and rg < run else 'absent before it'}")
for i in (sb, ct, rg):
    if i is not None: print(f'  {re.sub(r"^\[ *[0-9.]+\] ", "", L[i]).strip()[:120]}')
if bad:
    print(f'P4 VOID: {len(bad)} oops/trap lines before /init:')
    for i in bad[:40]: print(f'  {L[i].strip()[:160]}')
    sys.exit(1)
tt = [l.strip() for l in L if l.startswith('TRAPTEST:') and not l.startswith('TRAPTEST: dmesg')]
if not any(l.startswith('TRAPTEST:') for l in L):
    nc = idx('unable to open an initial console')
    print('P4 NO READING: no TRAPTEST line at all' + (f"; the kernel printed 'Warning: unable to open an initial console' at line {nc + 1}, so /init ran with no stdout" if nc is not None else ''))
    res['P4'] = False
    print('\nB5B2-LINUX ' + ' '.join(f"{k}={'ok' if v else 'MISS'}" for k, v in res.items() if k != 'P4') + ' P4=NO-READING')
    sys.exit(1)
want = ['TRAPTEST: up', 'TRAPTEST: regs status 91', 'TRAPTEST: enter status 1', 'TRAPTEST: loop alive', 'TRAPTEST: done']
got = [next((t for t in tt if t.startswith(w.split(',')[0]) and (w != 'TRAPTEST: up' or t.startswith('TRAPTEST: up'))), None) for w in want]
seq = [t for t in tt]
res['P4'] = len(tt) >= 5 and [tt[0].startswith('TRAPTEST: up')] + tt[1:5] == [True] + want[1:]
print(f"P4 {'AS ON THE TWIN' if res['P4'] else 'DIFFERS FROM THE TWIN'} (a known failure, not a pass):")
for t in seq: print(f'  {t[:120]}')
pz = idx('Kernel panic', run or 0)
if pz is not None: print(f'after done (the end of the run, not a reading): {re.sub(r"^\[ *[0-9.]+\] ", "", L[pz]).strip()[:100]}')
print('\nB5B2-LINUX ' + ' '.join(f"{k}={'ok' if v else 'MISS'}" for k, v in res.items()))
sys.exit(0 if all(res.get(k) for k in ('P1', 'P2', 'P3', 'P4')) else 1)
