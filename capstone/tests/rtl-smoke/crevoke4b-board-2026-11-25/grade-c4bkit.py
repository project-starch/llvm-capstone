"""grade-c4bkit.py KITDIR RUNDIR MODE SEALSUFFIX : grade one boot of the crevoke4b bare kit (llvm-capstone
land/rtl-crevoke4b 77f0e191b97f, crevoke4b-bare-2026-11-25/PREREG.md "Grading").
MODE base : the instrument control on 904cd9c6b (no flash); every image graded against the 904cd9c6b column.
MODE cand : the run on the 88db40027 seal; AS PREDICTED = every E position equals the 88db40027 column; NOT FLASHED =
            every E position equals the 904cd9c6b column where the two columns differ; FAIL otherwise. VOID ARM (cand only):
            an interrupt arm (crevint-mt, -sh, -lat, every crevsv) whose position 9 reads 0.
Both modes:
 - VOID BOOT if any readback does not name SEALSUFFIX, a RUN BEGIN line's sha16 or rec differs, or sublet-life's E
   positions are not exact (in the mode's column);
 - VOID IMAGE if its SR differs from the SR of both columns; NO READING without an SR line; the key-check sentinel
   0x6b657962616420 in any SV is a KEY-CHECK FAILURE;
 - U positions are skipped; M and R positions are printed, never graded, except the crevint-lat band: each positive
   lat[1..7] (positions 14-20, signed 64-bit) times cycles/ticks (positions 21/22) is at most 15,000; VOID if none.
Missing kit files are an ERROR (exit 2). Exit 0 only when the boot is not VOID and every image reads AS PREDICTED."""
import json, os, re, sys
def err(m): print('ERROR', m); sys.exit(2)
if len(sys.argv) != 5 or sys.argv[3] not in ('base', 'cand'): err(__doc__)
k, rd, mode, seal = sys.argv[1:]
SENT = 0x6b657962616420
INTR = {'b5b2-crevint-mt', 'b5b2-crevint-sh', 'b5b2-crevint-lat', 'b5b2-crevsv-none', 'b5b2-crevsv-srev', 'b5b2-crevsv-straw',
        'b5b2-crevsv-derive', 'b5b2-crevsv-kill'}
try: ev = json.load(open(os.path.join(k, 'expected-vectors.json')))
except OSError as e: err(e)
cl = open(os.path.join(rd, 'chain.log')).read()
void = []
rb = re.findall(r'^RK \S+ \S+ READBACK (resident|after flash|after loads) = (\S+) rc=[03]', cl, re.M)
if not any(w == 'after loads' for w, _ in rb) or not any(w != 'after loads' for w, _ in rb): void.append(f'readbacks missing ({[w for w, _ in rb]})')
if any(not n.endswith(seal) for _, n in rb): void.append(f'a readback does not name {seal}')
tags = sorted(d for d in os.listdir(rd) if re.fullmatch(r'\d\d-[\w-]+', d))
if not tags: err('no load directories')
s64 = lambda x: x - (1 << 64) if x >> 63 else x
res = []
for t in tags:
    name = t[3:]; e = ev.get(name) or err(f'{t}: not a kit image')
    bm = re.search(rf'^RK \S+ \S+ RUN {re.escape(t)} BEGIN image {re.escape(name)} sha ([0-9a-f]{{16}}) rec ([0-9a-f]+) ', cl, re.M)
    if not bm or bm.group(1) != e['sha256'][:16] or int(bm.group(2), 16) != int(e['board_rec'], 16): void.append(f'{t}: RUN BEGIN missing or sha/rec differ')
    p = os.path.join(rd, t, 'uart.txt')
    if not os.path.isfile(p): res.append((t, 'NO READING', 'no uart.txt')); continue
    x = open(p, errors='replace').read()
    srl = [int(y, 16) for y in re.findall(r'SR ([0-9A-F]{16})', x)]; sv = [int(y, 16) for y in re.findall(r'SV ([0-9A-F]{16})', x)]
    if SENT in sv: res.append((t, 'KEY-CHECK FAILURE', f'the sentinel at positions {[j + 1 for j, y in enumerate(sv) if y == SENT]}')); continue
    if not srl: res.append((t, 'NO READING', 'no SR line')); continue
    vc = [int(y, 16) for y in e['vector']]; vb = [int(y, 16) for y in e['vector_904cd9c6b']]
    cc, cb = e['classes'], e['classes_904cd9c6b']
    if srl != [len(vc)] and srl != [len(vb)]: res.append((t, 'VOID IMAGE', f'SR {srl} is neither {len(vc)} nor {len(vb)}')); continue
    def exact(vec, cls):
        if len(sv) != len(vec): return False, [f'length {len(sv)} != {len(vec)}']
        d = [j + 1 for j in range(len(vec)) if cls[j] == 'E' and sv[j] != vec[j]]
        return not d, d
    rec = ' '.join(f'p{j + 1}={sv[j]:#x}' for j in range(len(sv)) if (cc[j] if j < len(cc) else 'E') in 'MR')
    okc, dc = exact(vc, cc); okb, db = exact(vb, cb)
    if mode == 'base':
        v = 'AS PREDICTED (base)' if okb else 'MISS (base)'; why = '' if okb else f'E positions differ from the 904cd9c6b column at {db}'
    else:
        if okc: v, why = 'AS PREDICTED', ''
        else:
            diffpos = [j for j in range(min(len(vc), len(vb))) if cc[j] == 'E' and vc[j] != vb[j]]
            nf = len(sv) == len(vb) and diffpos and all(sv[j] == vb[j] for j in diffpos)
            v, why = ('NOT FLASHED', f'reads the 904cd9c6b column on the differing E positions {[j + 1 for j in diffpos]}') if nf else ('FAIL', f'E positions differ from the 88db40027 column at {dc}')
        if name in INTR and len(sv) >= 9 and sv[8] == 0 and v != 'NOT FLASHED': v, why = 'VOID ARM', f'position 9 (timer traps at the site) reads 0; was {v}'
    if name == 'b5b2-crevint-lat' and len(sv) >= 22:
        lat = [s64(sv[j]) for j in range(12, 20)]; cyc, tk = sv[20], sv[21]
        if tk == 0: band = 'VOID (ticks 0)'
        else:
            pos = [(i + 1, l * cyc / tk) for i, l in enumerate(lat[1:]) if l > 0]
            band = ('VOID (no positive latency after the first)' if not pos else
                    f"{'PASS' if all(c <= 15000 for _, c in pos) else 'FAIL'}: lat {lat} ticks; cycles/ticks {cyc}/{tk}; positive after the first in cycles {[(i, round(c)) for i, c in pos]} (bound 15,000; first {lat[0]} recorded)")
        why += f'; latency band {band}'
        if mode == 'cand' and not band.startswith('PASS') and v == 'AS PREDICTED': v = 'VOID ARM' if band.startswith('VOID') else 'FAIL'
    res.append((t, v, (why + ('; ' if why and rec else '') + (f'recorded {rec}' if rec else '')).strip('; ')))
for t, v, d in res: print(f'{t:24s} {v:22s} {d[:300]}')
ctl = next((v for t, v, _ in res if t.endswith('sublet-life')), None)
if ctl not in ('AS PREDICTED', 'AS PREDICTED (base)'): void.append(f'the sublet-life control reads {ctl}')
ok = sum(v.startswith('AS PREDICTED') for t, v, _ in res[1:]); n = len(res) - 1
print(f"\nC4BKIT {mode} " + ('VOID: ' + '; '.join(void) if void else f"{ok}/{n} AS PREDICTED; " + ', '.join(f"{c} {[t for t, v, _ in res if v == c]}" for c in sorted({v for _, v, _ in res[1:]}) if not c.startswith('AS PREDICTED'))))
sys.exit(1 if void or ok != n else 0)
