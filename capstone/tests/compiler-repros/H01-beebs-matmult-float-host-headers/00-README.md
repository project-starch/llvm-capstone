# H-01 — BEEBS `matmult-float` pulled the HOST glibc headers into a cross-compile · FIXED

**This is a BUILD-CONFIGURATION defect in one benchmark, not a compiler regression and
not a silicon defect.** It is filed here because `tests/compiler-repros/` is the closest
existing home; the fix belongs to whoever owns the BEEBS harness, not to the Capstone
backend. Sibling material a reader may have arrived looking for: C-19 in this same tree
is a genuine backend regression, and `tests/fpga-repros/` is for suspected RTL defects.

## Verdict

`run-beebs-matmult-float.sh` reaches `/usr/include/math.h` — the build machine's glibc —
and dies on `/usr/include/bits/floatn.h:97`:

    typedef __float128 _Float128;
    error: __float128 is not supported on this target

**It is not Capstone-specific.** The same line is rejected for `riscv64-unknown-elf`, so
this is a cross-compile reaching host headers, not a gap in the Capstone target:

    capstone64-unknown-elf   error: __float128 is not supported on this target
    riscv64-unknown-elf      error: __float128 is not supported on this target

## Why it surfaced only now

The benchmark has almost certainly never run. It is number **33 of 82** in the sweep
order, and every recorded BEEBS run stopped earlier than that: the three on file reached
**3, 5 and 24** benchmarks before one flaked boot ended the sweep. It was added on
2026-06-23, so it has been dead for two months without anyone being able to see it.

The run that found it is the first sweep that did not abort — 78 PASS, 3 FLAKE, and this
one FAIL out of 82. That is the point of the change that enabled it: a suite that stops
early looks fast and says nothing.

## Reproduce

    bash src/repro.sh

Deterministic; it does not depend on a boot and never reaches QEMU. Two commands: the
benchmark as the nightly runs it, then the one-line reduction against both targets.

## The fix, in the same commit as this folder

`build-beebs-matmult-float-capstone.sh` already strips `<stdio.h>` and `<stdlib.h>` from
the upstream source because neither exists freestanding. `<math.h>` belonged in that list
and was missed. It now goes too, with freestanding prototypes for the two functions the
source names:

    float fabsf(float);
    float frexpf(float, int *);

Prototypes rather than implementations, because the only callers sit in `values_match`,
which is dead once `verify_benchmark` is replaced and which `--gc-sections` drops at
link — the comment on that link line already said so.

**And the benchmark passes.** This folder was written while that was still unknown; the
first run after the fix returns `__BEEBS_MATMULT_FLOAT_PASSED__` and exit 0. So the two
months it spent unreachable cost nothing but the not knowing.

## What it does NOT establish

That the other 81 benchmarks are healthy. 78 passed and 3 flaked in the sweep that found
this one; the three that flaked have no verdict yet.
