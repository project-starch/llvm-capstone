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
  compiled 27, failed 0
```

The whole interpreter core compiles for capstone64: `core/iwasm/{interpreter,common}`,
`core/shared/{mem-alloc,utils}`, in one coherent configuration (classic interpreter,
no AOT, no WASI, no threads, global heap pool).

Getting there took a freestanding libc shim, a platform layer, and exactly ONE
upstream patch. The route is worth recording because it is the answer to "is this
worth finishing":

| | compiled | what moved |
|---|---:|---|
| platform layer only | 15 / 29 | every failure a missing libc declaration |
| headers pulled through the platform layer | 22 | WAMR relies on it to include stdio/math, as the RTOS ports do |
| `static_assert` | 23 | |
| one coherent config | 24 / 27 | classic and fast interpreter exclude each other; compiling both was the "no member named 'operand'" failures |
| `BH_MALLOC`, `strtof`, `strtok_r`, `strtoull` | 26 | |
| patch 0001 | **27 / 27** | |

**Not one capability failure at any step.** No `Cannot select`, no i128, no tag
diagnostic. The single upstream patch is a function pointer routed through
`uintptr_t` in a static initialiser, which is the same shape as SQLite's
`SQLITE_INT_TO_PTR` and MicroPython's `atexit.c`, and it is three characters per
site.

`-nostdinc` is load-bearing in the census. Without it the driver still searches
`/usr/include`, the host `stdio.h` wins over `adapted/include/`, and the census
reports libc failures that are really include-order failures. It did exactly that
on the first run with the shim in place.

The census asserts its baseline rather than printing it, the way the musl survey
does, and the gate is negative-tested: with the baseline one higher it exits 1 and
names the regression.

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

## Size, which is what decided the other candidates

Every object built and totalled:

    .text  259 901      .data  3 156      .bss  1 048 804

254 KiB of code. The `.bss` is almost entirely `CAPSTONE_WAMR_ARENA_BYTES`, a
number this port chooses, so the image size is not upstream's problem to solve.

For scale, from this week: SQLite at 3.3 MB and JerryScript at 2.9 MB both
exceeded the single-region ceiling of 1 376 256 bytes and needed a declared
two-region budget. MicroPython is 321 KiB of `.text`. WAMR is the smallest
candidate assessed, and it fits a single region even before declaring anything.

## Where the bring-up stands

```
stage  0  return at once                     OK
stage 10  os_malloc from the port's arena    OK
stage 11  static mutex + counter (gp carve)  OK
stage 12  wasm_runtime_memory_init           OK
stage 13  set_default_running_mode           OK
stage 14  wasm_native_init                   OK
stage  1  wasm_runtime_full_init             OK
stage 20  get_package_type                   OK
stage 21  a load that MUST fail: size 3      OK
stage 22  wasm_runtime_load                  OK   the module parses
stage  3  wasm_runtime_instantiate           OK
stage  4  wasm_runtime_call_wasm             capability fault
```

The runtime initialises, loads a module and instantiates it. What is left is the
call itself.

### The instrument that made this tractable

A trap reports a RUNTIME pc, and nothing prints the load base, so a fault used to
say "cause 24, somewhere". Guessing the base from the region's 512 KiB alignment
put the offset outside .text, which is how we know guessing does not work.

So the base is MEASURED, and from inside the same image: the first call to the
domain returns the runtime address of `domain_main`, whose ELF address is in the
same file; the second call does the work that faults. The entry glue does not
re-run initialisers between calls, so a static counter survives. One boot, and any
pc maps to a line.

That instrument is what turned "cause 24" into `lw a4, 0x34(s1)` inside
`alloc_hmu_ex`, and 0x34 is where `size` sits in a PACKED node with 16-byte
pointers. The offset was the diagnosis.

### Patch 0002 is right and it is not yet enough

The generated code proves the layout changed: `lw a4, 0x34(s1)` became
`ldc a2, 0x10(s1)` and `ldc s1, 0x20(s1)`, capability loads at the natural offsets
of an unpacked node. Static assertions confirm the node is 80 bytes, `left` is at
16, and the embedded root buffer is 16-aligned.

The fault then moved EARLIER, into `gc_init_with_pool`, at `movc a0, s1`
immediately after an indirect call. Stage 12 passed before the patch and does not
now, so this is a regression the patch surfaced or caused, and it is the next
thing to settle. Recorded as a fault site rather than a theory: no explanation is
offered here because none has been tested.

### What the ladder has already bought

Stage 12 faulted before patch 0002 and returns after it, which is the whole value
of a ladder over a single end-to-end run: it named the function, and the function
named the line.

`gc_init_with_pool` aligned its pool by masking through `uintptr_t`:

    char *buf_aligned = (char *)(((uintptr_t)buf + 7) & (uintptr_t)~7);

On this target that is not rounding, it is a discard. Four sites in the allocator
do it. The fix computes the same address and moves the ORIGINAL pointer by the
difference, so the value and the alignment are unchanged and the provenance
survives; on an ordinary target it compiles to the same code.

**That is the same defect and the same fix as MicroPython's
`0003-gc-align-down-without-losing-the-pointer`, in a different allocator.** Two
runtimes, independently written, one pattern -- which is the kind of thing the
corpus exists to say.

### Three things had to be right before that verdict was worth anything

**One translation unit.** `getGpCaptableIndex` numbers globals per module, so 28
separate objects each start at zero and collide. Built separately the image linked
and reported `cap table 1 (1 global)` for a runtime with its own heap: it built
and could not work. `tools/gen-amalgam.py` concatenates the sources and renames
the fifteen file-local statics that collide, and the build gates on exactly one TU
owning globals.

**The gp-captable ABI flags.** `start-gp-captable-interp.S` loops over
`.capstone_gp_initdesc`; without `-mllvm -capstone-gp-captable` that table is
empty and the glue hands the runtime globals that were never carved.

**A declared budget**, even though 208 KiB fits a single region. An image that
fits anyway is exactly when a missing declaration goes unnoticed.

## Next

Compiling is not running, and the gap is domain glue rather than porting:

1. A domain entry -- the equivalent of `micropython/port/mpy_domain.c`: a
   `domain_main(unsigned *res, unsigned func)` that initialises the runtime over
   the arena, instantiates a module and returns a marker.
2. A link: these objects plus `beebs_freestanding_string.o` for the string
   routines and `lua_libc.o` for snprintf/vsnprintf, under the gp-captable
   linker script.
3. A `.wasm` module to execute, baked in as a byte array so the domain needs no
   filesystem.
4. `domreq.S` with a declared budget. Not optional -- both SQLite and JerryScript
   silently produced unloadable images this week without it, and at 254 KiB this
   one would fit anyway, which is exactly when the mistake goes unnoticed.

Only after all four does a corpus make sense, and it should be read from fix
commits rather than issue titles.
