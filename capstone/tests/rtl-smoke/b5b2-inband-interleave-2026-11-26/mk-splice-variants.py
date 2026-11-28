"""mk-splice-variants.py CLEAN_CAPTURE OUTDIR : the in-band interleave controls, from one clean board capture of B5TL1-502.gcc_r
(board 3's B5TL1-g3, uart sha16 given in the README). Writes base.txt (CR stripped) and eight variants:
  A whole TRAPTEST tick line inside the .s file's base64 block      B whole kernel line there
  C kernel line spliced mid-line (after 30 chars) into that block    D tick line spliced mid-line there
  E whole kernel line inside the SPEC-HASH block                     F kernel line spliced mid-line into a hash line
  G one base64 character replaced by '$' (not interleave: must read NO READING under v2)
  H a tick and a kernel line spliced into one base64 line"""
import sys
src, D = sys.argv[1:]; L = open(src, errors='replace').read().replace('\r', '').split('\n')
open(f'{D}/base.txt', 'w').write('\n'.join(L))
tick = 'TRAPTEST: tick 1234.56'; kern = '[ 2243.335305] sublet: fault cause=25 pc=0x3ff7c01f00 address=0x3ff7f15e00 pid=503 comm=cc1_r ra=0x3ff7c023a4'
i = L.index('@@SPEC-FILE capstone-lifetimes t1.opts-O3_-finline-limit_50000.s') + 3; h = L.index('@@SPEC-HASH-BEGIN capstone-lifetimes') + 2
def w(n, LL): open(f'{D}/{n}.txt', 'w').write('\n'.join(LL))
x = L[i]; y = L[h]
w('A-tick-line', L[:i] + [tick] + L[i:]); w('B-kernel-line', L[:i] + [kern] + L[i:])
w('C-kernel-midline', L[:i] + [x[:30] + kern, x[30:]] + L[i + 1:]); w('D-tick-midline', L[:i] + [x[:30] + tick, x[30:]] + L[i + 1:])
w('E-kernel-in-hash', L[:h] + [kern] + L[h:]); w('F-kernel-midline-hash', L[:h] + [y[:20] + kern, y[20:]] + L[h + 1:])
w('G-garbage', L[:i] + [x[:30] + '$' + x[31:]] + L[i + 1:])
w('H-two-splices', L[:i] + [x[:10] + tick, x[10:40] + '[  7.123456] random: crng init done', x[40:]] + L[i + 1:])
