#!/usr/bin/env python3
"""Adapt php-src's Zend/bench.php and Zend/micro_bench.php (PHP 8.3.17) to PHP 5.0.0.

usage: adapt-scripts.py SRC_DIR OUT_DIR [MICRO_N]
  SRC_DIR  a directory holding the two scripts as released (checked by SHA-256 below)
  OUT_DIR  receives bench.php and micro_bench.php, adapted, and bench-reduced.php: bench.php with
           every test's work cut to about a fifth (REDUCED below), so the Capstone arm fits a
           one-hour profile (the full bench.php is ~68e9 attempts in the plain arm, and the
           Capstone arm under QEMU runs an order of magnitude fewer attempts per second)
  MICRO_N  micro_bench.php's iteration count N (default 5000000, the script's own); a smaller N is
           a REDUCED workload and is reported as one (each test's loop runs N times)

Every change is listed here and nothing else differs (each replacement must match exactly once):
  both   gethrtime(): hrtime() is PHP 7.3; PHP 5.0.0's microtime(true) returns the same quantity
         (seconds, as a float), so the body becomes `return microtime(true);`
  micro  `$x = $this::TEST;` -> `$x = self::TEST;`  ($this::CONST is PHP 5.3; the test still reads
         the class constant from inside a method)
  micro  `const N = 5000000;` -> `define('N', 5000000);`  (top-level const is PHP 5.3; MICRO_N
         replaces the 5000000)
  micro  issetor() and issetor2() and their two timed calls removed: they measure the short
         ternary `?:`, which PHP 5.0.0 does not have (5.3), so no rewrite measures the same thing

REDUCED (bench-reduced.php only, on top of the bench.php change; each test keeps its code, only its
size argument or loop bound changes, and the printed label names the new argument):
  simple, simplecall, simpleucall, simpleudcall   loop bound 1000000 -> 200000 (5 loops)
  mandel   the per-point iteration cap $color=1000 -> 200;  mandel2  the cap $k<5000 -> $k<1000
  ackermann(7) -> (6)   ary(50000) -> (10000)   ary2(50000) -> (10000)   ary3(2000) -> (400)
  fibo(30) -> (27)      hash1(50000) -> (10000) hash2(500) -> (224)      heapsort(20000) -> (4000)
  matrix(20) -> (4)     nestedloop(12) -> (9)   sieve(30) -> (6)         strcat(200000) -> (40000)

PER-TEST (OUT_DIR/pertest/NN-name.php, one script per bench.php test): bench.php's own functions and
timing helpers, unchanged but for the bounds below, then a driver that times ONE test. Each test's
size is set so that it is about 0.5-1.2e9 attempts in the plain arm (calibrated on the plain arm's
full bench.php, where the guest clock under -icount shift=0 reads one second per 1e9 instructions):
  simple, simplecall, simpleucall, simpleudcall   loop bound 1000000 -> 250000 (5 loops)
  mandel   $color=1000 -> 50;   mandel2  $k<5000 -> $k<250
  ackermann(6) ary(100000) ary2(150000) ary3(300) fibo(25) hash1(100000) hash2(500)
  heapsort(6000) matrix(6) nestedloop(9) sieve(10) strcat(500000)
"""
import hashlib, os, sys

SHA256 = {"bench.php": "c9bf9e8a225ac2d388225bec21714b2ddf93e9b1b541c1c2fe23e666039893fb",
          "micro_bench.php": "272c470c08c5b0677517575fa3114eb988d3915cdb65b9d830a29ff170c8ae1c"}
HRTIME = ("  $hrtime = hrtime();\n  return (($hrtime[0]*1000000000 + $hrtime[1]) / 1000000000);",
          "  return microtime(true);")
ISSETOR = """function issetor($n) {
    $val = array(0,1,2,3,4,5,6,7,8,9);
    for ($i = 0; $i < $n; ++$i) {
        $x = $val ?: null;
    }
}

function issetor2($n) {
    $f = false; $j = 0;
    for ($i = 0; $i < $n; ++$i) {
        $x = $f ?: $j + 1;
    }
}

"""
ISSETOR_CALLS = """issetor(N);
$t = end_test($t, '$x = $a ?: null', $overhead);
issetor2(N);
$t = end_test($t, '$x = $f ?: tmp', $overhead);
"""
REDUCED_ARGS = [("ackermann", "7", "6"), ("ary", "50000", "10000"), ("ary2", "50000", "10000"),
                ("ary3", "2000", "400"), ("fibo", "30", "27"), ("hash1", "50000", "10000"),
                ("hash2", "500", "224"), ("heapsort", "20000", "4000"), ("matrix", "20", "4"),
                ("nestedloop", "12", "9"), ("sieve", "30", "6"), ("strcat", "200000", "40000")]
REDUCED = [("for ($i = 0; $i < 1000000; $i++)", "for ($i = 0; $i < 200000; $i++)", 4),
           ("$thisisalongname < 1000000;", "$thisisalongname < 200000;", 1),
           ("      $color=1000;", "      $color=200;", 1),
           ("$z=$t, $k<5000; $k++)", "$z=$t, $k<1000; $k++)", 1)] + [
    (f'{f}({a});\n$t = end_test($t, "{f}({a})");', f'{f}({b});\n$t = end_test($t, "{f}({b})");', 1)
    for f, a, b in REDUCED_ARGS]
PER_TEST_BOUNDS = [("for ($i = 0; $i < 1000000; $i++)", "for ($i = 0; $i < 250000; $i++)", 4),
                   ("$thisisalongname < 1000000;", "$thisisalongname < 250000;", 1),
                   ("      $color=1000;", "      $color=50;", 1),
                   ("$z=$t, $k<5000; $k++)", "$z=$t, $k<250; $k++)", 1)]
PER_TEST = [("simple", None), ("simplecall", None), ("simpleucall", None), ("simpleudcall", None),
            ("mandel", None), ("mandel2", None), ("ackermann", "6"), ("ary", "100000"), ("ary2", "150000"),
            ("ary3", "300"), ("fibo", "25"), ("hash1", "100000"), ("hash2", "500"), ("heapsort", "6000"),
            ("matrix", "6"), ("nestedloop", "9"), ("sieve", "10"), ("strcat", "500000")]
DRIVER = "$t0 = $t = start_test();\n"
CHANGES = {
    "bench.php": [HRTIME],
    "micro_bench.php": [HRTIME, ("$x = $this::TEST;", "$x = self::TEST;"),
                        ("const N = 5000000;", "define('N', 5000000);"),
                        (ISSETOR, ""), (ISSETOR_CALLS, "")],
}


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    src, out = sys.argv[1:3]
    n = int(sys.argv[3]) if len(sys.argv) == 4 else 5000000
    if n <= 0:
        sys.exit("MICRO_N must be positive")
    CHANGES["micro_bench.php"][2] = ("const N = 5000000;", f"define('N', {n});")
    os.makedirs(out, exist_ok=True)
    for name, changes in CHANGES.items():
        data = open(os.path.join(src, name), "rb").read()
        got = hashlib.sha256(data).hexdigest()
        if got != SHA256[name]:
            sys.exit(f"{name}: sha256 {got}, expected {SHA256[name]} (php-src 8.3.17)")
        text = data.decode()
        for old, new in changes:
            if text.count(old) != 1:
                sys.exit(f"{name}: {old.splitlines()[0]!r} matches {text.count(old)} times, expected 1")
            text = text.replace(old, new)
        write(out, name, text, len(changes))
        bench_text = text
        if name == "bench.php":
            for old, new, count in REDUCED:
                if text.count(old) != count:
                    sys.exit(f"bench.php: {old!r} matches {text.count(old)} times, expected {count}")
                text = text.replace(old, new)
            write(out, "bench-reduced.php", text, len(changes) + len(REDUCED))
            per_test(out, bench_text)


def per_test(out, text):
    """bench.php's definitions (everything before its driver), the per-test bounds, one test."""
    if text.count(DRIVER) != 1:
        sys.exit("bench.php: the driver's first line is not where it was")
    defs = text[:text.index(DRIVER)]
    for old, new, count in PER_TEST_BOUNDS:
        if defs.count(old) != count:
            sys.exit(f"bench.php: {old!r} matches {defs.count(old)} times, expected {count}")
        defs = defs.replace(old, new)
    os.makedirs(os.path.join(out, "pertest"), exist_ok=True)
    for i, (f, arg) in enumerate(PER_TEST):
        if f"function {f}(" not in defs and f"function {f} (" not in defs:
            sys.exit(f"bench.php: no function {f}")
        call = f"{f}({arg or ''})"
        label = call if arg else f
        body = defs + DRIVER + f'{call};\n$t = end_test($t, "{label}");\ntotal();\n?>\n'
        write(out, os.path.join("pertest", f"{i:02d}-{f}.php"), body, len(PER_TEST_BOUNDS) + 1)


def write(out, name, text, n):
    open(os.path.join(out, name), "w").write(text)
    print(f"{name}: {n} changes, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


main()
