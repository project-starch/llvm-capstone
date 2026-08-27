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
stage  5  lookup + create_exec_env           OK
stage  4  wasm_runtime_call_wasm             OK   returns what the module computes
```

The runtime initialises, loads a module, instantiates it and **runs it**: stage 4
returns `0x5741002A`, and 0x2A is the 42 that `i32.const 7; i32.const 35; i32.add`
computes. Two things had to be right at once, and neither alone is enough -- see
the 2x2 below. This is one module with no imports; the residual section below says
what that qualification is doing there.

### Where stage 4 stops, and the two fixes the hypothesis produced

The frame hypothesis was tested, not assumed, with a static-assert probe over the
real struct layouts. Two of the three assertions passed and the third named the
defect:

```
offsetof(WASMInterpFrame, prev_frame) == 0        passes
offsetof(WASMInterpFrame, lp) % 16 == 0           passes
offsetof(WASMExecEnv, wasm_stack_u) % 16 == 0     FAILS -- it is 8
```

And upstream says why in the field's own name:

```c
union { uint64 __make_it_8_byte_aligned_; uint8 bottom[1]; } wasm_stack_u;
```

Eight is the pointer's alignment on the targets WAMR was written for. A frame is
placed at `bottom` and its first field is a pointer, so the first
`frame->prev_frame = ...` stores across an unaligned boundary -- which was exactly
the `stc s5, 0x0(s2)` the trap named.

The second half is `wasm_interp_interp_frame_size`, which rounds to 4. Frames come
from bumping `wasm_stack.top` by that size, so a 4-aligned size unaligns the next
frame however well the base is placed.

Both are patch 0003, and both moved the fault forward: 0x16484 to 0x1BE90 to
0x1BF24. What sat at 0x1BF24 is resolved below; the "untagged frame pointer" read
of it was wrong, and the pc was again not naming the instruction that faulted.

### What 0x1BF24 actually was: the label stack, and the dispatch table

Two independent defects, and the 2x2 is what separates them. All four arms are
stage 4 under QEMU, built from ONE tree with one variable changing, via the
`WAMR_LABEL_STACK_PAD` and `WAMR_LABELS_AS_VALUES` knobs:

| | computed goto | switch dispatch |
|---|---|---|
| **no pad** | cause 4, `addr = 0x1022d1e08` | cause 4, `addr = 0x1022d1e28` |
| **pad** | cause 1 at an unrelocated pc | **42** |

Both no-pad arms give the SAME fault whatever the dispatch, which is what attributes
it to the pad rather than to the knob next to it. Both addresses are congruent to 8
mod 16, and 8 is exactly what the layout predicts for this module: `param = 0`,
`local = 0`, `max_stack = 2`, so `sp_boundary = lp + 8` with `lp` 16-aligned.

Each knob has its own positive control in the artifact: `handle_table` is a symbol in
both `labels=1` images and absent from both `labels=0` ones, and the pad's
`neg`/`sub`/`andi 0x3`/`add` sequence appears only in the padded pair. The two
dispatch modes differ by one captable global (`handle_table` itself); the padded and
unpadded arms have identical global counts.

**RETRACTED (2026-08-27).** An earlier version of this table gave cause 24 for the
top-left cell. That number came from a log written the previous day, from a binary
built before patches 0002 and 0003. The image of that name on disk gives cause 4, and
the corrected table is the stronger one: without the pad, dispatch makes no
difference at all. The mistake was reading a log by filename instead of checking that
it belonged to the artifact under test.

*The alignment.* `wasm_interp_classic.c` builds the label stack by casting the
operand stack's end:

```c
frame_csp = frame->csp_bottom = (WASMBranchBlock *)frame->sp_boundary;
```

`sp_boundary` is a `uint32 *`, so it is 4-byte granular. A `WASMBranchBlock` is
made of pointers, which here are 16 bytes and need 16-byte alignment, so the first
capability store into the label stack is misaligned. Patch 0004 pads
`max_stack_cell_num` until `param + local + max_stack` is a whole number of
pointer-sized cells. The padding is counted in `all_cell_num`, which is what sizes
the frame, so nothing overflows. One site casts that pointer, so there is one fix.

Reading the pc would NOT have found this, and the reason generalises to every
capability fault this emulator reports. `_helper_access_with_cap` is `static` and is
not inlined, so `GETPC()` inside it returns an address outside the code-gen buffer,
`cpu_restore_state` takes its early exit, and `env->pc` is never restored: the
printed pc is the translation block's ENTRY, which in all three runs was the return
address of a `jalr`. The real faulting instruction was +4 in two of them and +12 in
the third, so "the next instruction" is not a rule either.

What identified it was the cause code, which narrows by WIDTH: `op_helper.c` raises
cause 4 from the capability path only inside `if (size == 16)`, for loads and stores
alike, so the two-byte `lhu` at the reported pc cannot be what faulted and the `stc`
after it can. `badaddr` is stale on this path and must not be used. The
`[CAPSTONE] Unaligned cap access (addr = ...)` line is the only one carrying the real
address. This is written up in `agent-handoff/ref/HOW-TO-RUN-ON-QEMU.md`.

*The dispatch table.* `core/config.h` turns `WASM_ENABLE_LABELS_AS_VALUES` on for
any GCC or Clang, which compiles the interpreter to a computed goto over a static
table of `&&label` addresses. Those addresses are not relocated for the domain's
load slide, so the first table-driven dispatch jumps to a link-time address:
cause 1 at pc 0x190ac, a value with no slide applied. Building with
`-DWASM_ENABLE_LABELS_AS_VALUES=0` selects the portable `switch`, which has no
address table. `WAMR_LABELS_AS_VALUES=1` restores the original for A/B work.

*The control.* A domain that reports a number proves nothing until the number is
shown to follow its input, so `gen-test-module.py` takes `--a` and `--b`. Changing
the module to 11 + 88 returned **-29**, which looked like a failure and was not:
`i32.const` takes SIGNED LEB128, the generator was writing the operand as a raw
byte, and one byte 0x58 means -40 rather than 88. 11 + -40 = -29 -- the interpreter
had computed exactly what the module said, including the mistake. The encoder is
fixed and the summands now round-trip. That miss is worth more than a clean pass
would have been: nothing that returns a constant can produce -29 from those bytes.

### The residual, which is why this says "runs this module"

The dispatch table was the largest instance of an un-relocated pointer, not the only
one. Scanning the working image's data segment for 8-byte words landing inside
`.text`:

```
.data 0x41b10  ->  0x23aa4 (invokeNative)   5 times
relocation sections in the image: 0
```

Five untagged words holding `invokeNative`'s LINK-TIME address, in an image with no
relocations. They never fire here because a 39-byte module with no imports never
reaches `wasm_interp_call_func_native`. So the honest claim is that WAMR executes
THIS module, and the first module with an import is expected to hit the same hazard
somewhere else. That is the next thing to test, and it is a cheap test.

Same root as patch 0002 either way: a constant that encodes the pointer's width
without saying so. There it was `UINTPTR_MAX == UINT64_MAX`; here the literals
8 and 4.

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
