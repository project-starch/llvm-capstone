#!/usr/bin/env python3
"""make-bench-100.py PHP-5.1.0-TARBALL OUTDIR: bench.php at about 1/100 of its work, for P1 under QEMU and silicon.

From Zend/bench.php of the PHP 5.1.0 release (both checked by SHA-256), as the Fig. 10 tool
(5060011844e reuse-hooks/php/make-bench.py, the 1/10 setting) but with every count cut about 10x further.
Each replacement must match exactly once. Writes:
  bench100.php        the workload (bench.php's own form: results discarded by ob_end_clean, timings printed)
  bench100-check.php  the same with end_test's ob_end_clean() -> ob_end_flush(), so every test's result prints;
                      a correctness check, not the timed workload.
  bench100g.php, bench100g-check.php   as above plus GRID: the fixed sizes cut too (mandel's and mandel2's grids,
                      ary3's array, matrix's SIZE, sieve's range), which loop counts alone cannot reach: natively
                      bench100 is 1/54 of bench.php's time, mostly those fixed costs."""
import hashlib, sys, tarfile
TARBALL_SHA256 = '3eab94beeac8d1d4f64a79190b78989324025d068d90501e0dbea0698a5d2f41'
BENCH_SHA256 = 'cf2587610279004af239024a2d57af0aaf1f4d66b573b3c59f7b5d8fe7a71053'
R = []
def loop(old, new, tail): R.append((f'for ($i = 0; $i < {old}; $i++) \n    {tail}', f'for ($i = 0; $i < {new}; $i++) \n    {tail}'))
loop(1000000, 10000, '$a++;')                                        # simple
R.append(('$thisisalongname < 1000000;', '$thisisalongname < 10000;'))  # simplecall
loop(1000000, 10000, 'strlen')                                       # simplecall
loop(1000000, 10000, 'hallo(')                                       # simpleucall
loop(1000000, 10000, 'hallo2(')                                      # simpleudcall
R.append(('$color=1000;', '$color=10;'))                             # mandel: iterations per point
R.append(('$k<5000; $k++)', '$k<50; $k++)'))                         # mandel2: iterations per point
R.append(('for ($k=0; $k<1000; $k++) {', 'for ($k=0; $k<10; $k++) {'))  # ary3: passes
def call(f, old, new): R.append((f'{f}({old});\n$t = end_test($t, "{f}({old})");', f'{f}({new});\n$t = end_test($t, "{f}({new})");'))
call('ackermann', 7, 4)     # Ack(3,n) ~ 4^n: 1/64
call('ary', 50000, 500)
call('ary2', 50000, 500)
call('fibo', 30, 20)        # ~phi^n: 1/123
call('hash', 50000, 500)
call('hash2', 500, 50)      # n^2: 1/100
call('heapsort', 20000, 200)
call('matrix', 20, 1)       # products of 30x30: 1/20, the floor
call('nestedloop', 12, 6)   # n^6: 1/64
call('sieve', 30, 1)        # passes: 1/30, the floor
call('strcat', 200000, 2000)
GRID = [
    ('$h1=150;', '$h1=50;'),                                         # mandel: 51 x 151 points -> 51 x 51
    ('$x++ < 75;', '$x++ < 25;'),                                    # mandel2: 30 x 75 points -> 30 x 25
    ('ary3(2000);\n$t = end_test($t, "ary3(2000)");', 'ary3(500);\n$t = end_test($t, "ary3(500)");'),
    ('$SIZE = 30;', '$SIZE = 20;'),                                  # matrix: 30x30 -> 20x20 products
    ('range (0,8192);', 'range (0,2048);'), ('$i<8193;', '$i<2049;'), ('$k <= 8192;', '$k <= 2048;'),  # sieve
]
CHECK = ('  ob_end_clean();\n  $total += $end-$start;', '  ob_end_flush();\n  $total += $end-$start;')
def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    tarball, out = sys.argv[1], sys.argv[2]
    if sha(open(tarball, 'rb').read()) != TARBALL_SHA256: raise SystemExit(f'{tarball}: not the pinned php-5.1.0.tar.gz')
    with tarfile.open(tarball) as t: bench = t.extractfile('php-5.1.0/Zend/bench.php').read()
    if sha(bench) != BENCH_SHA256: raise SystemExit('Zend/bench.php: not the pinned file')
    text = bench.decode()
    for old, new in R + GRID + [CHECK]:
        if text.count(old) != 1: raise SystemExit(f'expected exactly one {old!r}, found {text.count(old)}')
    for old, new in R: text = text.replace(old, new)
    grid = text
    for old, new in GRID: grid = grid.replace(old, new)
    for name, body in (('bench100.php', text), ('bench100-check.php', text.replace(*CHECK)),
                       ('bench100g.php', grid), ('bench100g-check.php', grid.replace(*CHECK))):
        open(f'{out}/{name}', 'w').write(body); print(f'{out}/{name} {sha(body.encode())}')
if __name__ == '__main__': main()
