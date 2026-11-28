"""grade-spec-chain-v2.py UART_TXT OUTDIR STAGES : grade-spec-chain.py (4fa95eaf) plus INBAND, below; its header follows.
grade-spec-chain.py UART_TXT OUTDIR STAGES : grade one boot of b5b2-spectest-2026-11-25 Amendment 1's chain (runtime/b5b2-linux
5aaa7bbb2c6e). Derived from grade-spec5.py: each stage <ARM>-<bench> is read against board-stages/<ARM>/<bench>
(SPEC_STAGE_ROOT overrides the root, for dry runs only), and STAGES (comma-separated, from SHA256SUMS.chain) is the
expected list: an expected stage that never began is NO READING, and a stage not in the list is an ERROR.
 C   'TRAPTEST: control smoke.dom rc=7', with 'ok' printed before it (smoke prints 'ok' with no newline);
 S*  each '@@QSTEP-BEGIN <stage>' .. '@@QSTEP-END <stage> rc=N' span is written to OUTDIR/<stage>/step.txt (CR stripped) and
     read by the P1 tooling, wt-spec ports/spec-cpu2017/board/run-stage.py read, against ARMDIR/<bench>/plan.json (ARMDIR
     default <mnt-hdd>/capstone/c4/b5b2spec/board-stages/B5T) and the board SPEC tree. A stage passes only with
     QSTEP rc=0, the read's exit 0 and its READ line saying 'specdiff k/k PASS' with rc!=0: 0;
     a 'TRAPTEST: watchdog killed' inside the span is NO READING for that stage;
 T   each command's seconds (up1 - up0), as run-stage.py prints them: a first reading, no band.
A failed C voids S. No capture, or no stage, is an ERROR (exit 2).
 X264 (Amendment 13, runtime/b5b2-linux e7b3046cf5e8; x264 only, every other stage unchanged): 525.x264_r's test stage runs
     no imagevalidate, so specdiff reads NO-OUTPUT by construction (EXPECTED, not a failure). It passes with QSTEP rc=0, every
     arm's gen and run commands present and rc 0, every validation verdict NO-OUTPUT, and the seven data outputs (frame_0/50/
     100/150/155.yuv, BuckBunny_New.264, BuckBunny.yuv) equal to X264_REF by sha256 (the twin's key B5TL1-525.x264_r at
     e7b3046cf5e8, "data_hash_verdict EQUAL"; the reference reads the same on the Full and spatial-only twins).
 INBAND (v2; the console-interleave hazard the runtime lane raised 2026-11-26 from nested-corpus, nested-workloads 0e8089933116):
     on the slow UART a kernel printk ("[ <t>] ...") or a TRAPTEST line can land inside an @@SPEC-FILE base64 block or an
     @@SPEC-HASH block, as a whole line or mid-line. run-stage.py's reader then raises on the base64/gzip decode (the stage
     read MISS that protection did not cause), and a mid-line splice in a hash line silently truncates that file's hash.
     Inside those blocks only, every "[ <t>] ...\n" and "TRAPTEST: ...\n" is removed with its newline, so a split line
     rejoins; the stage's step.txt is the result, step.raw.txt its slice as captured, and "INBAND <stage>: k interleaved
     line(s) removed" is printed (nothing is printed when k = 0, so a clean capture grades byte-identically). A line still
     not base64 inside a file block (or not "<64 hex>  <name>" inside a hash block) makes the stage NO READING (unreadable
     in-band block), never MISS. gzip's CRC32 checks every decoded file."""
import json, os, re, subprocess, sys
X264_REF = {  # runtime/b5b2-linux e7b3046cf5e8 twin-sp-b5tl1.json, key B5TL1-525.x264_r (Amendment 13)
    'frame_0.yuv': '3214803d8be74416f3de93f30363f4c6b0703d0415ff259505189fd8f9c5d31d',
    'frame_50.yuv': 'bdf3b9ea5f28d7ec6055120196217205a1cdd51f38a4a36060b0347c84e91b21',
    'frame_100.yuv': '7fc792cf69a27155462dec8e8778b9ec4406cdee9ff0457f40da4b42dfe13150',
    'frame_150.yuv': 'c485b156e2db47cdc4ee9e06b50261e42069930f62998e2b6d320984460857ec',
    'frame_155.yuv': 'f1352e842177ffea4580565949c4e38741c57e0c1901877aa43263911a49be95',
    'BuckBunny_New.264': '0e453c440e686acc95e660653bf014731fffeadfed40a53700ec7afdf73e50c0',
    'BuckBunny.yuv': 'c3a3d3e272a82bb6dd35486e55f1f05c9446308b2556e8dfb60ed1c084234459'}
IB_LINE = re.compile(r'(?:\[ *\d+\.\d+\] |TRAPTEST: )[^\n]*\n')
B64 = re.compile(r'[A-Za-z0-9+/]+={0,2}'); HL = re.compile(r'[0-9a-f]{64}  \S.*')
def inband_clean(body):   # (cleaned body, lines removed, residual foreign lines)
    out, k, bad, i = [], 0, [], 0
    for m in re.finditer(r'^(@@SPEC-FILE [^\n]*\n|@@SPEC-HASH-BEGIN (\S+)\n)(.*?)^(@@SPEC-FILE-END$|@@SPEC-HASH-END \2$)', body, re.M | re.S):
        out.append(body[i:m.start(3)]); blk, n = IB_LINE.subn('', m.group(3)); k += n; out.append(blk); i = m.end(3)
        pat = B64 if m.group(1).startswith('@@SPEC-FILE ') else HL
        bad += [l for l in blk.split('\n') if l and not pat.fullmatch(l)]
    out.append(body[i:]); return ''.join(out), k, bad
def x264_verdict(d, qrc, rd):   # Amendment 13: (ok, why, extra lines)
    try: r = json.load(open(d / 'result.json'))
    except (OSError, ValueError) as e: return False, f'X264 no result.json ({e})', []
    arms = r.get('arms') or {}; why = []; lines = []
    if qrc != 0: why.append(f'QSTEP rc={qrc}')
    if not re.search(r'(\d+)/\1 commands ended, rc!=0: 0,', rd): why.append('not every command ended rc 0')
    if not arms: why.append('no arm in result.json')
    for an, a in sorted(arms.items()):
        cs = {c['tag']: c['rc'] for c in a.get('commands', [])}
        if set(cs) != {'gen', 'run'} or any(v != 0 for v in cs.values()): why.append(f'{an}: commands {cs}, want gen and run rc 0')
        vs = [v['verdict'] for v in a.get('validation', [])]
        if not all(v.startswith('NO-OUTPUT') for v in vs): why.append(f'{an}: a specdiff verdict is not NO-OUTPUT: {vs}')
        h = a.get('hashes', {}); bad = [k for k in X264_REF if h.get(k) != X264_REF[k]]
        lines.append(f'X264 {an}: data outputs {len(X264_REF) - len(bad)}/{len(X264_REF)} equal the reference'
                     + (f'; differ or absent: {[(k, (h.get(k) or "ABSENT")[:16]) for k in bad]}' if bad else '')
                     + f'; specdiff {len(vs)}x NO-OUTPUT (expected)')
        if bad: why.append(f'{an}: {len(bad)} data output(s) differ from the reference')
    return not why, 'X264 (Amendment 13): ' + ('gen+run rc 0, 7/7 data outputs EQUAL' if not why else '; '.join(why)), lines
from pathlib import Path
if len(sys.argv) != 4: sys.exit(__doc__)
H = os.environ['CAPSTONE_HDD']  # the HDD root (the run copy had it as a literal)
RS = f'{H}/wt-spec/capstone/ports/spec-cpu2017/board/run-stage.py'; TREE = f'{H}/spec/tree-board/SPEC17'
ROOT = Path(os.environ.get('SPEC_STAGE_ROOT') or f'{H}/c4/b5b2spec/board-stages'); EXP = [x for x in sys.argv[3].split(',') if x]
try: raw = open(sys.argv[1], errors='replace').read()
except OSError as e: print(f'ERROR no capture: {e}'); sys.exit(2)
t = raw.replace('\r\n', '\n').replace('\r', '')
if not t.strip(): print('ERROR empty capture'); sys.exit(2)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ob = [m.start() for m in re.finditer(r'OpenSBI v', t)]
if len(ob) > 1: print(f'note: {len(ob)} OpenSBI banners; graded up to the second'); t = t[:ob[1]]
print(f'capture: {len(t)} chars; OpenSBI banner {"present" if ob else "ABSENT"}; Run /init {"present" if "Run /init as init process" in t else "ABSENT"}')
for k in ('Capstone: register tags kept', 'Boot HART MEDELEG', 'sublet: .dom loader registered', 'TRAPTEST: id ', 'TRAPTEST: up,'):
    m = re.search(rf'^.*{re.escape(k)}.*$', t, re.M)
    print(f"  {re.sub(r'^\[ *[0-9.]+\] ', '', m.group(0)).strip()[:120] if m else 'ABSENT: ' + k}")
c = re.search(r'(ok)?TRAPTEST: control smoke\.dom rc=(\d+)', t)
C = bool(c and c.group(1) and c.group(2) == '7')
print(f"C {'AS PREDICTED' if C else 'FAILED'}: {c.group(0) if c else 'no control line'}")
spans = list(re.finditer(r'^@@QSTEP-BEGIN (\S+)$', t, re.M))
if not spans: print('S: no stage began'); print(f"\nB5B2-SPECCHAIN C={'ok' if C else 'FAILED'} S=none"); sys.exit(2 if C else 1)
res = []
for i, m in enumerate(spans):
    n = m.group(1); stop = spans[i + 1].start() if i + 1 < len(spans) else len(t)
    e = re.search(rf'^@@QSTEP-END {re.escape(n)} rc=(\d+)$', t[m.start():stop], re.M)
    body = t[m.start():m.start() + e.end() if e else stop]
    d = out / n; d.mkdir(exist_ok=True)
    cb, k, bad = inband_clean(body)
    if k or bad: (d / 'step.raw.txt').write_text(body); print(f'INBAND {n}: {k} interleaved line(s) removed' + (f'; {len(bad)} foreign line(s) remain: {bad[0][:100]!r}' if bad else ''))
    body = cb; (d / 'step.txt').write_text(body)
    if n not in EXP: print(f'ERROR stage {n} is not in the expected list {EXP}'); sys.exit(2)
    arm, bench = n.split('-', 1); wd = 'TRAPTEST: watchdog killed' in body
    if wd:
        km = re.search(r'TRAPTEST: watchdog killed .* after (\d+)s', body)
        res.append((n, 'NO READING', f"watchdog kill at {int(km.group(1)):,} s" if km else 'watchdog kill (no bound on its line)', [])); continue
    if not e: res.append((n, 'NO READING', 'no QSTEP-END (the capture ended inside the stage)', [])); continue
    if bad: res.append((n, 'NO READING', f'unreadable in-band block: {len(bad)} foreign line(s) after removing {k} interleaved', [])); continue
    p = subprocess.run(['python3', RS, 'read', '--stage', str(ROOT / arm / bench), '--tree', TREE, '--text', str(d / 'step.txt'), '--out', str(d)],
                       capture_output=True, text=True)
    (d / 'read.txt').write_text(p.stdout + p.stderr)
    rl = [l for l in p.stdout.splitlines() if l.startswith('READ ') or l.lstrip().startswith(('run', 'specdiff'))]
    rd = next((l for l in rl if l.startswith('READ ')), '')
    mm = re.search(r'specdiff (\d+)/(\d+) PASS', rd)
    ok = int(e.group(1)) == 0 and p.returncode == 0 and 'rc!=0: 0' in rd and mm and mm.group(1) == mm.group(2) and int(mm.group(2)) > 0
    if bench == '525.x264_r':   # Amendment 13 replaces the specdiff criterion for x264 only
        ok, xw, xl = x264_verdict(d, int(e.group(1)), rd); rl = rl + xl
        res.append((n, 'PASS' if ok else 'MISS', f'QSTEP rc={e.group(1)}; read rc={p.returncode}; {xw}', rl)); continue
    res.append((n, 'PASS' if ok else 'MISS', f'QSTEP rc={e.group(1)}; read rc={p.returncode}', rl))
for n in EXP:
    if n not in [r[0] for r in res]: res.append((n, 'NO READING', 'the stage never began', []))
for n, v, why, rl in res:
    print(f"S {n} {'VOID (C failed): ' + v if not C else v}: {why}")
    for l in rl: print(f'    {l.strip()[:200]}')
for l in re.findall(r'^TRAPTEST: (?:watchdog|dmesg|stage).*$', t, re.M): print(f'  {l[:160]}')
done = 'TRAPTEST: done' in t
print(f"\nB5B2-SPECCHAIN C={'ok' if C else 'FAILED'} " + ' '.join(f"{n}={'VOID' if not C else v}" for n, v, _, _ in res) + f" done={'yes' if done else 'NO'}")
sys.exit(0 if C and res and all(v == 'PASS' for _, v, _, _ in res) else 1)
