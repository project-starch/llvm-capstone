# MicroPython for a freestanding Capstone domain

Bring-up material for running the MicroPython interpreter in a pure-capability domain. The plan,
the measurements and the open questions live in
`agent-handoff/plans/micropython-domain-compilation.md`; this directory holds the things that have
to be reproducible.

Layout mirrors `benchmarks/sqlite/`: upstream source is fetched to `$CAPSTONE_TMP_ROOT`, our
changes live here as patches, and the freestanding environment is in `adapted/`.

| Path | What it is |
|---|---|
| `fetch-micropython.sh` | clone at a pinned commit, apply `patches/` |
| `patches/` | portability fixes against upstream, each with its rationale above the diff |
| `adapted/include/` | freestanding headers the core needs and `-ffreestanding` does not provide |
| `port/` | the Capstone port: `mpconfigport.h`, the hal headers, and `mpy_domain.c` |
| `census-capstone.sh` | compile every `py/*.c` for the silicon ABI and report what fails |

## Getting to a census

```bash
bash fetch-micropython.sh
# once, with a stock toolchain, to generate the qstr/module headers:
make -C $CAPSTONE_TMP_ROOT/micropython/ports/minimal CROSS_COMPILE=riscv64-linux-
bash census-capstone.sh
```

## Where this stands

The whole core plus the port compiles as ONE translation unit for `capstone64`, which is what the
gp-captable ABI requires (`getGpCaptableIndex` numbers globals per module). Measured on that object:

| | |
|---|---:|
| `.text` at `-O0` | 321 KiB |
| carves in `.capstone_gp_initdesc` | 232 |
| `__capstone_cap_init` | 11,900 B |
| `.bss` | 97 KiB (the 96 KiB GC heap) |
| undefined symbols | 11 |

The eleven are `__gpfree_globals_base` (the linker script provides it), `setjmp`/`longjmp` (the
capability-aware pair in `tests/runtime-qemu/silicon-ladder/nlrjmp_kernel.h`, which passes in a
domain under QEMU), and the eight mem/str functions that already exist in
`benchmarks/beebs/adapted/beebs_freestanding_string.c`. There is no `malloc`: allocation is
MicroPython's own GC over a static array.

**It has never been linked or run.** Linking needs the globals offset sized to `.text` the way
`build-sqlite-silicon.sh` does it, and running needs the descriptor delivered into `dom_data`.

## Two things that will mislead you

**Build with an assertions-enabled clang.** Without assertions, `APInt::getSExtValue()` returns the
low 64 bits of a capability-width constant silently rather than aborting, so a NoAsserts compiler
miscompiles where this one stops. Treat any domain built with one as void.

**A file that compiles is not a file that is correct.** The C front end accepts every tag
manipulation MicroPython does, so `(mp_int_t)obj & 7` in a file that compiles cleanly still lowered
to *something*, and whether that something preserves a capability tag is a runtime question no
compile can answer. Two such sites are patched here (`patches/0002`, `patches/0003`); the census
measures the compilation axis and nothing else.

## Why `adapted/include/` is thin

The whole core needs ten external symbols: `memcpy memmove memset memcmp strlen strcmp strncmp
strchr` plus `read`/`write`. All eight string/memory functions already exist in
`benchmarks/beebs/adapted/beebs_freestanding_string.c`, and `read`/`write` are the existing HostCall
path. The headers here declare that surface and stub the rest.

`-nostdlibinc` is load-bearing and not tidiness: without it clang still searches `/usr/include` for
a bare-metal triple, `#include <string.h>` resolves to the **host glibc header**, and nothing in
this directory is read at all. That was true for the first four rounds of the census and was caught
only by deleting `adapted/string.h` and watching the result not change.
