# WAMR as a third nested-allocator candidate: a feasibility read, not a port

Assessed after JerryScript was re-tested and still fails. Nothing here is built or
run; every number below is either read out of the upstream tree at f73410e
(2026-08-25) or queried from an API, and the two are marked apart.

## Why a WebAssembly runtime at all

The corpus finding so far is about NESTED allocators: a runtime carves one heap at
startup and sub-allocates in software, so every block inherits the whole-heap
capability and the runtime's own free is bookkeeping that never reaches the
hardware. MicroPython shows it at one 384 KiB array; mruby shows it at GC-page
granularity (corpus row 14, already measured).

A WebAssembly runtime would add something neither has: the guest's linear memory
is one contiguous array BY SPECIFICATION, and every guest pointer is a 32-bit
offset into it. The finding would stop being about implementation choices and
become a statement about a deployed standard.

## WAMR brings its own allocator, wasm3 does not

Read from the tree:

  core/shared/mem-alloc/{mem_alloc.c, ems/}   ems_alloc.c, ems_gc.c, ems_hmu.c,
                                              ems_kfc.c -- its own allocator WITH
                                              its own GC
  mem_allocator_create(void *mem, uint32_t size)   carves from one contiguous buffer

That is MicroPython's shape (`py/gc.c` over `mpy_heap`) in a different language
runtime, which is exactly what a second data point needs.

wasm3 by contrast has only `m3_Malloc_Impl` / `m3_Free_Impl`, wrappers over
malloc. Its linear memory is still one array, but that is the standard, not its
allocator. wasm3 is therefore a HYBRID and not a second MicroPython.

## The property JerryScript lacked, and WAMR has

    typedef void *gc_object_t;                     ems_gc.h:53
    gc_alloc_vo(void *heap, gc_size_t size)        returns a pointer

Objects are real pointers, not compressed offsets. That is the difference that
decided JerryScript: its `ecma_value_t` is a `uint32_t`, so there is nowhere to
put a capability and 93 sites in 60 functions reconstruct pointers arithmetically.

Counted in WAMR's interpreter core, the same pattern:

    core/iwasm/interpreter    70 uintptr_t casts, 2 back to a pointer
    core/iwasm/common        138 uintptr_t casts, 12 back to a pointer
    core/shared/mem-alloc     20 uintptr_t casts, 2 back to a pointer

Sixteen sites against JerryScript's ninety-three. `STORE_PTR`, which narrows a
pointer into a wasm value slot, lives only in the
`WASM_CPU_SUPPORTS_UNALIGNED_ADDR_ACCESS == 0` branch and has 16 uses -- a
workaround path, not the object model.

## Porting cost, measured from the tree

There is no bare-metal platform, but the RTOS ports are the template and they are
small:

    nuttx      1 .c,  478 lines
    riot       3 .c,  693 lines   (platform, thread, time -- the last two are
    rt-thread  4 .c, 1363 lines    stubs in a single-threaded domain)
    zephyr     6 .c, 3774 lines

An interpreter-only build is about 80k lines across
core/iwasm/{interpreter,common} and core/shared/{mem-alloc,utils,platform/common},
comparable to MicroPython's core, which compiled to 321 KiB of .text at -O0.

Confirmed rather than assumed: compiling the allocator against the LINUX platform
layer fails immediately, because `platform_internal.h` pulls `/usr/include/time.h`
and the host typedefs conflict with a freestanding cross target. A Capstone
platform layer is the first piece of work and there is no way around it.

## The corpus, queried rather than recalled

Same method as the MicroPython corpus: NVD REST plus GitHub search, never typed
from memory.

    WAMR      1 temporal CVE (CVE-2023-52284, CWE-415)   10 use-after-free issues,
                                                          16 double-free issues
    wasm3     1 temporal CVE (CVE-2024-27530, CWE-416)    4 / 1
    wasmtime  4 temporal of 44 CVEs -- Rust with a JIT, unusable freestanding

**These are search hits, not classifications.** The MicroPython corpus audited its
own class column on 2026-08-18 and found a THIRD of it wrong, because it had been
assigned from issue titles rather than fix commits. The 26 above are an upper
bound on candidates; a corpus needs the same fix-commit reading.

## What would have to happen next, in order

1. A Capstone platform layer, using nuttx or riot as the template. ~500 lines,
   thread and time as stubs.
2. A census like `micropython/census-capstone.sh`: compile every core source for
   the silicon ABI and count what fails. That, not the line count, says whether
   the image fits -- SQLite at 3.3 MB and JerryScript at 2.9 MB both hit the
   single-region ceiling this week.
3. Only then a corpus, read from fix commits.

Steps 1 and 2 are what decide it, and step 2 is cheap once step 1 exists.
