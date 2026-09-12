# MicroPython in a Capstone domain

The interpreter, its GC heap and the port, compiled as one translation unit for
`capstone64` and linked into a domain image. `REPRODUCING.md` is the recipe for
every measurement; the scripts say at their heads what they do.

    bash fetch-micropython.sh                    # the pin, plus patches/
    bash build-micropython-silicon.sh            # the image, two-pass link
    bash census-capstone.sh                      # what a capability build costs
    bash repro-selftest.sh                       # the verdict logic, no board

## Four things a reader would otherwise get wrong

**Build with an assertions-enabled clang.** Without assertions,
`APInt::getSExtValue()` returns the low 64 bits of a capability-width constant
silently rather than aborting, so a NoAsserts compiler miscompiles where this
one stops. Treat any domain built with one as void.

**A file that compiles is not a file that is correct.** The C front end accepts
every tag manipulation MicroPython does, so `(mp_int_t)obj & 7` in a file that
compiles cleanly still lowered to *something*, and whether that something
preserves a capability tag is a runtime question no compile can answer. Two
such sites are patched here, `patches/0002` and `patches/0003`. The census
measures the compilation axis and nothing else.

**`-nostdlibinc` is load-bearing and not tidiness.** Without it clang still
searches `/usr/include` for a bare-metal triple, `#include <string.h>` resolves
to the host glibc header, and nothing in `adapted/include/` is read at all.
That was true for the first four rounds of the census and was caught only by
deleting `adapted/string.h` and watching the result not change.

**Everything external comes from somewhere named.** There is no `malloc`:
allocation is MicroPython's own GC over a static array. `setjmp` and `longjmp`
are the capability-aware pair in
`tests/runtime-qemu/silicon-ladder/nlrjmp_kernel.h`. The string and memory
functions are `benchmarks/beebs/adapted/beebs_freestanding_string.c`, whose
copies preserve tags. `__gpfree_globals_base` comes from the linker script.

## Why the corpora are here and not under `bug-corpora/`

`temporal-corpus/` and `spatial-corpus/` are generated: a CSV of rows, a script
that turns each into a self-contained case, and a verifier. Their `STATUS.md`
files are written by those scripts and say so. They live beside the port
because a case is a Python script plus an expectation, and running one means
building this image with that script baked in.
