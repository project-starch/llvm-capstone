#!/usr/bin/env python3
"""The PHP reuse study's workload (docs/plans/2026-11-15-php-reuse-distance.md) from PHP's own benchmark.

    make-bench.py PHP-5.1.0-TARBALL OUT.php

Reads Zend/bench.php out of the PHP 5.1.0 release tarball (both checked by SHA-256) and writes the
pinned one-tenth setting: every loop count and argument cut to about a tenth of bench.php's work
under QEMU (the full file did not finish in 140 minutes on the trap route), each
replacement matched exactly once. The result must have the SHA-256 the plan names.
"""
import hashlib
import sys
import tarfile

TARBALL_SHA256 = '3eab94beeac8d1d4f64a79190b78989324025d068d90501e0dbea0698a5d2f41'
BENCH_SHA256 = 'cf2587610279004af239024a2d57af0aaf1f4d66b573b3c59f7b5d8fe7a71053'
OUT_SHA256 = '6b9fc8f9ba15cb7a0972408b8e69724787085e5a50fc59898b49d0c52b934606'

# (old, new), each old text present exactly once.
REPLACEMENTS = [
    ('for ($i = 0; $i < 1000000; $i++) \n    $a++;', 'for ($i = 0; $i < 100000; $i++) \n    $a++;'),
    ('$thisisalongname < 1000000;', '$thisisalongname < 100000;'),
    ('for ($i = 0; $i < 1000000; $i++) \n    strlen', 'for ($i = 0; $i < 100000; $i++) \n    strlen'),
    ('for ($i = 0; $i < 1000000; $i++) \n    hallo(', 'for ($i = 0; $i < 100000; $i++) \n    hallo('),
    ('for ($i = 0; $i < 1000000; $i++) \n    hallo2(', 'for ($i = 0; $i < 100000; $i++) \n    hallo2('),
    ('$color=1000;', '$color=100;'),
    ('$k<5000; $k++)', '$k<500; $k++)'),
    ('for ($k=0; $k<1000; $k++) {', 'for ($k=0; $k<100; $k++) {'),
    ('ackermann(7);\n$t = end_test($t, "ackermann(7)");', 'ackermann(5);\n$t = end_test($t, "ackermann(5)");'),
    ('ary(50000);\n$t = end_test($t, "ary(50000)");', 'ary(5000);\n$t = end_test($t, "ary(5000)");'),
    ('ary2(50000);\n$t = end_test($t, "ary2(50000)");', 'ary2(5000);\n$t = end_test($t, "ary2(5000)");'),
    ('fibo(30);\n$t = end_test($t, "fibo(30)");', 'fibo(25);\n$t = end_test($t, "fibo(25)");'),
    ('hash(50000);\n$t = end_test($t, "hash(50000)");', 'hash(5000);\n$t = end_test($t, "hash(5000)");'),
    ('hash2(500);\n$t = end_test($t, "hash2(500)");', 'hash2(158);\n$t = end_test($t, "hash2(158)");'),
    ('heapsort(20000);\n$t = end_test($t, "heapsort(20000)");', 'heapsort(2000);\n$t = end_test($t, "heapsort(2000)");'),
    ('matrix(20);\n$t = end_test($t, "matrix(20)");', 'matrix(2);\n$t = end_test($t, "matrix(2)");'),
    ('nestedloop(12);\n$t = end_test($t, "nestedloop(12)");', 'nestedloop(8);\n$t = end_test($t, "nestedloop(8)");'),
    ('sieve(30);\n$t = end_test($t, "sieve(30)");', 'sieve(3);\n$t = end_test($t, "sieve(3)");'),
    ('strcat(200000);\n$t = end_test($t, "strcat(200000)");', 'strcat(20000);\n$t = end_test($t, "strcat(20000)");'),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    tarball, out = sys.argv[1], sys.argv[2]
    with open(tarball, 'rb') as f:
        if sha256(f.read()) != TARBALL_SHA256:
            raise SystemExit(f'{tarball}: not the pinned php-5.1.0.tar.gz')
    with tarfile.open(tarball) as t:
        bench = t.extractfile('php-5.1.0/Zend/bench.php').read()
    if sha256(bench) != BENCH_SHA256:
        raise SystemExit('Zend/bench.php: not the pinned file')
    text = bench.decode()
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(f'expected exactly one {old!r}')
        text = text.replace(old, new)
    data = text.encode()
    if sha256(data) != OUT_SHA256:
        raise SystemExit(f'the result has SHA-256 {sha256(data)}, not the pinned {OUT_SHA256}')
    with open(out, 'wb') as f:
        f.write(data)
    print(f'{out} {OUT_SHA256}')


if __name__ == '__main__':
    main()
