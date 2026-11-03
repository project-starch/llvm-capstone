#!/usr/bin/env python3
"""Adapt php-src's Zend/bench.php and Zend/micro_bench.php (PHP 8.3.17) to PHP 5.0.0.

usage: adapt-scripts.py SRC_DIR OUT_DIR [MICRO_N]
  SRC_DIR  a directory holding the two scripts as released (checked by SHA-256 below)
  OUT_DIR  receives bench.php and micro_bench.php, adapted
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
        open(os.path.join(out, name), "w").write(text)
        print(f"{name}: {len(changes)} changes, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


main()
