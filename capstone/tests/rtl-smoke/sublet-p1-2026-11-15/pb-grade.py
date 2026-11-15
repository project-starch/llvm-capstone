#!/usr/bin/env python3
"""Grade one Phase 1 pair-B row against the pre-registration (land/rtl-sublet-p1 2796b81618d8 + edd4971aab89).
usage: pb-grade.py <mcf|gcc|perl> run_flat run_paged ev23 ev29 ev30 ev31 [ev32 ev33]
  every count from the row's own runs; ev* from the PAGED run.
exit 0 = checks 1-3 PASS, 1 = a check FAILS, 4 = NO DATA (missing or non-numeric input).
Every reading carries "key 5 NOT MET" (Addendum 149)."""
import sys
OPS = {'mcf': (60282, 60238), 'gcc': (1088904, 1036504), 'perl': (921, 141)}
# Brackets as pre-registered, from G's Addendum 143 rows (bv-b1): (ev23_G, run_G)
G = {'mcf': (180614513, 144015279285), 'gcc': (6409096, 41536277321), 'perl': (30296, 6755758710)}
K, HOT, MIG_HI = 2, 7, 40
def die(m): print('NO DATA:', m); sys.exit(4)
a = sys.argv[1:]
if len(a) < 7 or a[0] not in OPS: die(__doc__.splitlines()[1])
try: v = [int(x.replace(',', '').replace('_', ''), 0) for x in a[1:]]
except ValueError as e: die(str(e))
row = a[0]; run_f, run_p, e23, e29, e30, e31 = v[:6]; extra = v[6:]
if min(run_f, run_p) <= 0: die('run_cycles must be > 0')
cd, cr = OPS[row]
def R(e): return e + 6*cd + 8*cr, e + 8*cd + 9*cr
def A(e): return e + 9*cd + 11*cr, e + 10*cd + 11*cr
d = run_p - run_f; noise = 0.002 * run_f; fails = []
r_lo, r_hi = R(e23); b_lo, b_hi = 0.9*r_lo, 1.1*r_hi
c1 = b_lo <= e31 <= b_hi
print(f'[key 5 NOT MET] pair B {row}: Delta run {d:+,} ({100*d/run_f:+.3f} % of flat {run_f:,})')
print(f'check 1 ev31 {e31:,} in band {b_lo:,.0f}..{b_hi:,.0f} (ev23 {e23:,} from this run): {"PASS" if c1 else "FAIL"}')
if not c1: fails.append(1)
model = 2*e31 + e30 + e29; tol = 0.25*model + noise
c2 = abs(d - model) <= tol
print(f'check 2 model 2*ev31+ev30+ev29 = {model:,}; |Delta - model| = {abs(d-model):,} <= {tol:,.0f}: {"PASS" if c2 else "FAIL"}')
if not c2: fails.append(2)
ge23, grun = G[row]; ga_lo, ga_hi = A(ge23); gr_lo, gr_hi = R(ge23)
lo = K*ga_lo - noise; hi = K*ga_hi + (HOT + 3*MIG_HI)*gr_hi + noise
c3 = lo <= d <= hi
print(f'check 3 Delta in [m=0, m=1 MIG 40] +/- 0.2 % = {lo:,.0f}..{hi:,.0f} (G ev23 as pre-registered): {"PASS" if c3 else "FAIL"}')
if not c3: fails.append(3)
a_lo, a_hi = A(e23); r2lo, r2hi = R(e23)
print(f'  info: same bracket on this run\'s ev23: {K*a_lo - noise:,.0f}..{K*a_hi + (HOT+3*MIG_HI)*r2hi + noise:,.0f} (not graded)')
print(f'reported: m = ev29/ev31 = {e29/e31 if e31 else float("nan"):.4f}; cycles/walk = ev30/ev29 = '
      f'{e30/e29 if e29 else float("nan"):.2f}', end='')
if extra: print(f'; ev32 = {extra[0]:,} (expected 0)' + (f'; ev33 = {extra[1]:,} (grows)' if len(extra) > 1 else ''))
else: print('; ev32/ev33 not given')
if row == 'perl' and abs(d) > noise: print(f'  NULL row: |Delta| {abs(d):,} > 0.2 % ({noise:,.0f}) -- outside run noise')
print('RESULT', 'PASS' if not fails else 'FAIL checks ' + ','.join(map(str, fails)))
sys.exit(1 if fails else 0)
