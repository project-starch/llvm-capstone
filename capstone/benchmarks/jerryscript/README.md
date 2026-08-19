# JerryScript as a Capstone domain

Second allocator port, and the reason for it is comparison rather than coverage.
MicroPython's finding rests on one runtime; a second one with the same allocator
shape and a different bug corpus is what turns it from an observation into a
result.

## Why JerryScript specifically

`jerry-core/jmem/jmem-heap.c` allocates from

    struct jmem_heap_t { ...; uint8_t area[]; };

one contiguous byte array carved in software by a first-fit free list. That is
structurally the same as MicroPython's `mpy_heap`, which is what makes the two
comparable.

It is also **sharper**. `typedef uint32_t ecma_value_t` -- every JavaScript value is
a 32-bit word, and every object reference is either a 16-bit offset into the heap or
a truncated address, reconstructed by

    uint_ptr <<= JMEM_ALIGNMENT_LOG;
    uint_ptr += (uintptr_t) &JERRY_HEAP_CONTEXT (first);

JerryScript therefore **cannot hold a per-object capability**; there is nowhere to
put one. In MicroPython the gap is a consequence of how the collector is written. Here
it is constitutive.

Selected over mruby, duktape and quickjs on the numbers in the sweep that chose it:
97 CVEs, 238 use-after-free issues, and an allocator that is one region rather than
many malloc'd pages. Lua was excluded because it does not file bugs on GitHub, so the
count said something about the search and nothing about Lua.

## State: it compiles AND links. It has not run.

    ./probe-compile.sh             ->  compiled 200    failed 0    (of 200)
    ./build-jerryscript-silicon.sh ->  .text 1,370,372   VERDICT: fits

Linking took three rounds, each one enumerated by the linker rather than guessed:
eleven soft-float builtins (JS numbers are doubles and this domain has no FP ABI),
then twenty libm functions, then `log2`. The soft-float set is the shared BEEBS one;
eighteen of the libm functions come from MicroPython's `lib/libm_dbl` -- **the same
math the MicroPython port measured with**, which is a real coupling to that checkout
and a deliberate one, since two runtimes being compared should not differ in their
transcendentals. `fabs`, `cbrt` and `log2` are implemented in
`port/capstone_libc_extra.c` and each was checked against the host libm before use.
`log2` splits off the exponent rather than dividing by ln2, because the obvious form
returns 2.9999999999999996 for `Math.log2(8)`.

### The next blocker, found by reading the budget rather than by a fault

    - cap table           48   (3 globals)
    - storage            144

`jerry_global_heap` is 524,288 bytes in `.bss` and it is NOT among those three. Under
`-capstone-gp-captable` a global that is not carved is not reachable as data, so the
heap would fault on first use.

The cause is the build shape, and it was confirmed with a control rather than
inferred: a single translation unit declaring two globals emits a descriptor with
`02000000` -- two carves. The descriptor is per translation unit, and this build
compiles 200 separate objects, of which one survives the link. MicroPython does not
hit this because it **amalgamates** `py/` and its port into one translation unit, and
gets 232 carves.

So the next step is an amalgamation, the same shape as
`../micropython/build-micropython-silicon.sh`. That also brings the collision problem
that port has already solved twice -- two `lib/` headers defining the same typedef, or
two modules defining the same static -- for which its `EXTMOD_STATICS` renaming
mechanism is the precedent.

### Still not done beyond that

- **The capability half is untouched.** `jmem_decompress_pointer` and
  `ecma_get_pointer_from_ecma_value` round-trip through `uintptr_t`, which drops the
  tag; every dereference afterwards is a cause-24 fault. All 51 call sites funnel
  through those two functions, which is far better than MicroPython's scattered
  arithmetic, but the patches do not exist yet.
- **Heap size is capped at 512 KB** and this must be decided before anything else.
  `ecma_value_t` is 32 bits and the domain heap sits above 4 GB, so the direct
  pointer path is unusable and the 16-bit compressed path is forced;
  `65535 << JMEM_ALIGNMENT_LOG` is 524280 bytes. MicroPython runs in 384 KB, so this
  is not a problem -- but finding it in phase two rather than now would be.

## Layout

Mirrors `../micropython/` on purpose.

| | |
|---|---|
| `fetch-jerryscript.sh` | pin + apply patches; refuses to continue on a stale patch |
| `probe-compile.sh` | compile every `.c` with the domain's flags, classify failures BY KIND |
| `patches/` | numbered, each with prose saying why and what was measured |
| `adapted/include/` | freestanding headers |
| `port/` | glue with bodies |

`adapted/include/` overlaps the MicroPython port's, and that duplication is
deliberate for now: extracting a shared shim before a third port exists would be
guessing at what is common. Seven headers were copied over unchanged, and
`inttypes.h`, `ctype.h`, `cbrt`, `log2`, `rand` and `RAND_MAX` were added here.

## What the probe was worth

Ten minutes, before any porting work, and it produced the whole shape:

| | files |
|---|---|
| compiled at once | 66 |
| blocked by `inttypes.h` | 104 |
| blocked by `ctype.h` | 6 |
| blocked by a missing libc function | 2 |
| **a compiler crash** | **1** |

The last one is C-20, `__builtin_ctz`, filed at
`../../tests/compiler-repros/C20-cttz-i32-crashes-legalizer/` and worked around by
`patches/0001` behind `JERRY_NO_BUILTIN_CTZ` so the workaround can be deleted rather
than found. Everything else was two headers.

The MicroPython port learned this lesson the expensive way: a compiler assertion was
recorded as blocking two corpus rows for a day, and the probe that refuted it would
have cost a minute. `probe-compile.sh` takes a source root as an argument and is not
JerryScript-specific.
