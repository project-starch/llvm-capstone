#!/usr/bin/env python3
"""Scale mruby's own benchmarks to emulator-sized inputs and record native references.

    prepare-bench.py <mruby benchmark dir> <native mruby> <native speedtest1> <out dir>

The code of each benchmark is unchanged; only its input size is reduced, so a run
under capstone-qemu takes tens of seconds instead of tens of minutes. Two
benchmarks print a result they otherwise discard, so the output can be checked.
<out dir>/native/<name>.out is the native output every guest run is compared with
byte for byte; for speedtest1 it is the STUDY-ORACLE lines of `--size 1`.
"""
import pathlib
import subprocess
import sys
import tempfile

SCALE = {
    # name: (source, [(old, new), ...])
    "ao_render": ("bm_ao_render.rb", [("Integer(ARGV[0] || 64)", "Integer(ARGV[0] || 16)")]),
    # FizzBuzz over 1..30 instead of 1..100: the Church numeral HUNDRED becomes THIRTY
    "lc_fizzbuzz": ("bm_app_lc_fizzbuzz.rb", [("p[" * 100 + "x" + "]" * 100, "p[" * 30 + "x" + "]" * 30),
                                               ("# puts answer", "puts answer")]),
    "so_lists": ("bm_so_lists.rb", [("NUM = 300", "NUM = 30")]),
    "fib": ("bm_fib.rb", [("fib(37)", "fib(30)")]),
    "so_mandelbrot": ("bm_so_mandelbrot.rb", [("size = 600", "size = 150")]),
    "mandel_term": ("bm_mandel_term.rb", []),
}


def main():
    src, mruby, speedtest1, out = (pathlib.Path(a) for a in sys.argv[1:5])
    (out / "native").mkdir(parents=True, exist_ok=True)
    for name, (source, edits) in SCALE.items():
        text = (src / source).read_text()
        for old, new in edits:
            if text.count(old) != 1:
                sys.exit(f"{source}: expected exactly one {old[:40]!r}")
            text = text.replace(old, new)
        if name == "so_lists":
            text = text.rstrip() + "\nputs result\n"
        (out / f"{name}.rb").write_text(text)
        res = subprocess.run([str(mruby), str(out / f"{name}.rb")], capture_output=True, check=True)
        if not res.stdout:
            sys.exit(f"{name}: native run printed nothing")
        (out / "native" / f"{name}.out").write_bytes(res.stdout)
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run([str(speedtest1), "--size", "1", "st.db"], cwd=tmp,
                             capture_output=True, text=True, check=True)
    oracle = [l for l in res.stdout.splitlines() if l.startswith("STUDY-ORACLE")]
    if len(oracle) < 30:
        sys.exit(f"speedtest1: only {len(oracle)} oracle lines")
    (out / "native" / "speedtest1.out").write_text("\n".join(oracle) + "\n")
    print(f"prepared {len(SCALE)} benchmarks and speedtest1 ({len(oracle)} oracle lines) in {out}")


if __name__ == "__main__":
    main()
