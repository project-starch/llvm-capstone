# `gh-148395` / CVE-2026-6100 -- a decompressor's input pointer left dangling after a MemoryError

A use-after-free in `zlib._ZlibDecompressor`, reached in the whole interpreter.

**Upstream fix:** `c3cf71c336` on the 3.13 branch (back-port of `8fc66aef6d` on main), after our
`v3.13.7` tag: it adds `self->zst.next_in = NULL;` at `decompress()`'s error label in
`Modules/zlibmodule.c`. **Consumer:** the freed block's pymalloc pool. **CVE:** CVE-2026-6100.

## The trigger

Upstream's own regression test body, `test_decompress_memoryerror_no_dangling_input`, added for this
CVE in `gh-154713` (commit `5bcd67189b`, `Lib/test/test_zlib.py`). It forces an allocation to fail
inside `decompress()` with CPython's `set_nomemory`, then calls `decompress()` again. The body is
copied unchanged; the injector is imported as whichever of `_testcapi` (the host oracle) or the
image's `_capstone_nomemory` is present.

The injector is harness scaffolding, not the program under test and not the defect. It is CPython's
own pyfailmalloc hook (`set_nomemory`/`remove_mem_hooks`), copied verbatim from
`Modules/_testcapi/mem.c` of the pin into a one-function module
(`ports/cpython/app/toolchain/capstone-nomemory.c`) because the capstone build omits the 32-file
`_testcapi`. The defect is `zlibmodule.c`, the pinned release's own code.

## Why this one had to be ported

The port had no zlib: its link check deliberately refuses `-lz`. zlib is plain C that cross-compiles
for the target unchanged, so `prepare-cpython-capstone.sh` now cross-builds it to a static archive and
hands it to configure by `-I`/`-L`; the final capability link takes `libz.a` as an archive
(`ports/common/application/build.py`, `link-cpython-capstone.py`). The link check still proves a bare
`-lz` fails without that `-L`, so nothing but the cross-built archive can satisfy it.

## Liveness at the pin, measured, with a differential

On the pinned 3.13.7 ASan build (pymalloc off) with `PYTHONMALLOC=malloc` and
`ASAN_OPTIONS=detect_leaks=0`: `AddressSanitizer: memcpy-param-overlap`, the overlapping-source
`memcpy` at `decompress` (`Modules/zlibmodule.c:1585`), reached from
`zlib_ZlibDecompressor_decompress_impl` -- the stale input pointer the next `decompress()` reuses. A
second build of the same tree with the upstream fix applied runs the trigger to its end with no
report. That fixed/buggy differential is the proof, in place of a C-model sibling.

## Which allocator owns the object

**nested.** `allocator_layer` = `L1-pymalloc-pool`, `allocator_consumed` = `yes`. The input buffer is
`PyMem_Malloc` memory, a pymalloc block; under `PYTHONMALLOC=pymalloc` the report does not survive.

## Negative control

`negative_control.py` runs the same forty decompressor objects and the same two `decompress()` calls
with no `set_nomemory`: every allocation succeeds, no error path is taken, and the second call reads a
live buffer. A fault there does not depend on the defect.
