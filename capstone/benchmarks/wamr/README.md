# WAMR as a Capstone domain

Third candidate for the nested-allocator corpus, after MicroPython (one 384 KiB
array) and mruby (GC pages). The assessment that led here is
`agent-handoff/history/26-08-2026_15-30-00_wamr-as-a-third-nested-allocator-candidate.md`.

Layout mirrors `benchmarks/micropython/`: upstream is fetched to
`$CAPSTONE_TMP_ROOT` at a pinned SHA, and everything of ours lives here.

| Path | What it is |
|---|---|
| `fetch-wamr.sh` | clone at a pinned commit |
| `port/` | the Capstone platform layer: `platform_internal.h` + `capstone_platform.c` |
| `census-capstone.sh` | compile every core source for capstone64 and bucket the failures BY CAUSE |

## Why WAMR and not wasm3

WAMR ships its own allocator with its own GC (`core/shared/mem-alloc/ems/`) behind
`mem_allocator_create(void *mem, uint32_t size)`, which carves from one contiguous
buffer. That is MicroPython's shape in a different runtime, which is what a second
data point needs. wasm3 only wraps malloc.

And `gc_object_t` is a `void *`: objects are real POINTERS, which is the property
JerryScript lacked and died on. Sixteen sites in WAMR's core reconstruct a pointer
from `uintptr_t`, against JerryScript's ninety-three.

## Where this stands

```
WAMR census at f73410e
  compiled 15, failed 14
failures by cause:
    5  freestanding libc: snprintf
    2  freestanding libc: isnan
    1  other: unexpected BH_MALLOC
    1  other: must use 'struct' tag to refer to type 'WASMModuleInstance'
    1  freestanding libc: wasm_runtime_malloc
    1  freestanding libc: vsnprintf
    1  freestanding libc: labs
    1  freestanding libc: bsearch
    1  freestanding libc: abort
```

**Not one capability-related failure.** No `Cannot select`, no i128, no pointer or
tag diagnostic. Every one is either a libc declaration `-ffreestanding` does not
provide, or a config combination. That is the answer the census existed to give,
and it is the difference between weeks of work and months.

The census asserts its baseline rather than printing it, the way the musl survey
does, and the gate is negative-tested: with the baseline set one higher it exits 1
and says so.

## The platform layer, and what is deliberate in it

The contract is 24 `os_*` entry points. nuttx implements it in 478 lines and riot
in 693; this is smaller because a domain is single-threaded and has no syscalls.

`os_mmap` carves from a static arena. A domain has no mmap, and WAMR's own
embedded configuration does the same thing. The nesting that follows is not an
artefact of the port -- it IS what the corpus measures, so the port must not
accidentally remove it. `os_malloc` shares that one arena for the same reason:
keeping the count of allocators in the image at one.

Mutexes are no-ops and there is one thread id. Not "unimplemented": complete, for
a machine with one core and no scheduler.

`os_thread_get_stack_boundary` returns NULL, which the vmcore reads as "unknown"
and skips its stack guard. A made-up boundary would be worse than none.

`os_time_get_boot_us` returns zero, marked with its ceiling: anything timed
through it reads zero, so a profile is void rather than wrong-by-a-little. Wire
the cycle counter before quoting any timing number.

## Next

1. A freestanding libc shim in `adapted/include/`, the way micropython and sqlite
   have one. Nine distinct functions from the census above.
2. Re-run the census and raise `BASELINE_OK`.
3. Then, and only then, a link and a size measurement. SQLite at 3.3 MB and
   JerryScript at 2.9 MB both hit the single-region ceiling this week, so the
   image size decides feasibility more than the line count does.
