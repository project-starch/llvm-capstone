"""scan-inband.py CAPTURE... : for each board capture, every line inside an @@SPEC-FILE .. @@SPEC-FILE-END block that is not pure
base64, and every line inside an @@SPEC-HASH-BEGIN .. @@SPEC-HASH-END block that is not "<64 hex>  <name>", is printed with
its line number: a kernel printk, a TRAPTEST line or anything else the console interleaved there (the hazard runtime raised
2026-11-26 from nested-corpus, nested-workloads 0e8089933116). Prints per capture "INBAND <path> blocks F/H foreign N", then
"SCAN-INBAND captures C, with a foreign line K". A missing or empty capture is an ERROR (exit 2). Exit 1 iff any foreign line."""
import re, sys
B64 = re.compile(r'[A-Za-z0-9+/]+={0,2}'); HL = re.compile(r'[0-9a-f]{64}  \S.*')
bad_caps = 0
for p in sys.argv[1:]:
    try: t = open(p, errors='replace').read().replace('\r', '')
    except OSError as e: print(f'ERROR {p}: {e}'); sys.exit(2)
    if not t.strip(): print(f'ERROR {p}: empty'); sys.exit(2)
    mode = None; nf = nh = 0; foreign = []
    for i, l in enumerate(t.split('\n'), 1):
        if mode == 'F':
            if l == '@@SPEC-FILE-END': mode = None; continue
            if l and not B64.fullmatch(l): foreign.append((i, 'FILE', l))   # empty lines are harmless (whitespace is dropped)
        elif mode == 'H':
            if l.startswith('@@SPEC-HASH-END '): mode = None; continue
            if l and not HL.fullmatch(l): foreign.append((i, 'HASH', l))
        elif l.startswith('@@SPEC-FILE '): mode = 'F'; nf += 1
        elif l.startswith('@@SPEC-HASH-BEGIN '): mode = 'H'; nh += 1
    print(f'INBAND {p} blocks {nf}/{nh} foreign {len(foreign)}')
    for i, k, l in foreign[:20]: print(f'    {k} line {i}: {l[:160]}')
    bad_caps += bool(foreign)
print(f'\nSCAN-INBAND captures {len(sys.argv) - 1}, with a foreign line {bad_caps}')
sys.exit(1 if bad_caps else 0)
