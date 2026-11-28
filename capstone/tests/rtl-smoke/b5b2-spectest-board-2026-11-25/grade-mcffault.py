"""grade-mcffault.py UART_TXT ARM [SEAL9] : mcf's reading on the a5c45a46d seal against b5b2-spectest Amendments 7 and 8
(runtime/b5b2-linux 2e4b4a85768a, 313276f8892a). The mcf fault line (the mcf pid's "sublet: fault") is mapped by mapfaults.py
to the instruction at its pc; then one class:
  CAUSE 25 AT AN LDC     supports R-80 + R-77 as the pair (Amendment 8's prediction);
  NAMED FAULT            any other cause or instruction: named, not the prediction;
  COMPLETED              the mcf stage ended (QSTEP-END) with no mcf fault: consistent with R-80, not evidence for it;
  WATCHDOG               killed at its bound with the kernel alive: no reading;
  SILENT WEDGE           the capture ends inside mcf with no fault and no stage end: REFUTES R-80 as mcf's mechanism.
Also prints the exposure (mcf-exposure.py). No mcf entry is an ERROR (exit 2).
2026-11-26 (with scan-c25ldc.py, on the runtime lane's report): every line is matched ANYWHERE in a line, since the slow console
splices printk and stage lines into program output, and its "TRAPTEST: dmesg" echo is not at a line start either; the mcf block of
mapfaults.py's output is the one headed by the mcf pid AND the fault's pc, ending at the next header. Before, a spliced mcf fault
line was not seen and the stage read COMPLETED."""
import os, re, subprocess, sys
X = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) not in (3, 4): print('ERROR', __doc__); sys.exit(2)
p, arm = sys.argv[1:3]; SEAL9 = sys.argv[3] if len(sys.argv) > 3 else 'a5c45a46d'   # b42c8c644: Amendment 9 makes a 25 at an LDC a finding
t = open(p, errors='replace').read().replace('\r', '')
m = re.search(r'\[ *([0-9.]+)\] sublet: mcf_r_base\.nati\[(\d+)\] enters', t)
if not m: print(f'ERROR no mcf entry in {p}'); sys.exit(2)
pid = m.group(2); rest = t[m.end():]
ex = subprocess.run([sys.executable, '-I', f'{X}/mcf-exposure.py', f'run:-:-:{p}'], capture_output=True, text=True).stdout.strip().split('\n')[0]
print('exposure: ' + re.sub(r'^run\s+board - -\s+', '', ex))
fl = re.search(rf'\[ *[0-9.]+\] (sublet: fault cause=(\d+) pc=(0x[0-9a-f]+) [^\n]*pid={pid}\b[^\n]*)', rest)
if fl:
    mp = subprocess.run([sys.executable, '-I', f'{X}/mapfaults.py', p, arm], capture_output=True, text=True).stdout
    hdr = [(h.start(), h.group(1), int(h.group(2), 16)) for h in re.finditer(r'^fault cause=\d+ pid=(\d+) comm=\S+ pc=(0x[0-9a-f]+)', mp, re.M)]
    hs = [k for k, (_, hp, hpc) in enumerate(hdr) if hp == pid and hpc == int(fl.group(3), 16)]
    blk = mp[hdr[hs[0]][0]:(hdr[hs[0] + 1][0] if hs[0] + 1 < len(hdr) else len(mp))] if hs else ''
    ins = re.search(r'^\s+>\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*(\S+)(.*)$', blk, re.M)
    print(fl.group(1)[:200]); print(blk.strip()[:900] if blk else 'mapfaults: no block for the mcf pid')
    if not ins: v = f'NAMED FAULT: cause {fl.group(2)} at pc {fl.group(3)}, instruction not mapped'
    elif fl.group(2) == '25' and ins.group(2) == 'ldc' and SEAL9 == 'b42c8c644': v = f'CAUSE 25 AT AN LDC ON b42c8c644 (vaddr 0x{ins.group(1)}{ins.group(3).rstrip()}): a FINDING against the PRE fix or a new mechanism (Amendment 9); NOT classified as R-77'
    elif fl.group(2) == '25' and ins.group(2) == 'ldc': v = f'CAUSE 25 AT AN LDC (vaddr 0x{ins.group(1)}{ins.group(3).rstrip()}): supports R-80 + R-77 (Amendment 8)'
    else: v = f'NAMED FAULT: cause {fl.group(2)} at vaddr 0x{ins.group(1)} {ins.group(2)}{ins.group(3).rstrip()} (not the Amendment 8 prediction)'
elif re.search(r'TRAPTEST: watchdog killed [^\n]*mcf_r', rest): v = 'WATCHDOG: killed at its bound with the kernel alive (no reading)'
elif re.search(r'@@QSTEP-END \S*mcf_r rc=\d+', rest): v = 'COMPLETED: no mcf fault; consistent with R-80, not evidence for it (Amendment 8)'
else: v = 'SILENT WEDGE: the capture ends inside mcf with no fault: REFUTES R-80 as mcf\'s mechanism'
print(f'\nMCF {v}')
