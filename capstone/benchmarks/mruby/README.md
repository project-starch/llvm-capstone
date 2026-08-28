# mruby as a Capstone domain

The fourth nested-allocator subject, and the one chosen for a specific reason:
**it is the densest source of bugs standard CHERI cannot see.**

## Why mruby

Every object is carved from `RVALUE objects[MRB_HEAP_PAGE_SIZE]` inside a GC page,
and the free list is threaded through the objects themselves:

```c
p->as.free.next = page->freelist;   /* incremental_sweep_phase */
page->freelist  = p;
```

No `malloc` and no `free` happens per object. A use-after-free on an RVALUE
therefore yields a pointer that is **tagged, in bounds, and never returned to the
system allocator**, so purecap raises nothing and revocation has nothing to revoke.
Its tracker carries 90 heap-buffer-overflow and 82 use-after-free issues, and 36 of
them are usable specimens. The catalogue is
`agent-handoff/ref/blindspot-cases/mruby.md`.

**A sanitizer cannot see this class either**, for exactly the same reason: ASAN
observes only `malloc` and `free`. So the oracle is a **wrong answer**, not a crash
report. `cases/a1-6339.rb` returns 1 or 2 rather than printing.

## Two allocators, deliberately

The contrast between them *is* the measurement:

| | what it is | bounds |
|---|---|---|
| outer | `cap_heap.c` from the rv8 corpus, on umm_malloc | **narrows every result to the request** -- CHERI-equivalent |
| inner | mruby's GC, handed one region via `mrb_gc_add_region` | objects inside it are never narrowed |

So an overflow past a malloc'd buffer faults, and the same overflow inside a GC page
does not. **Do not "fix" the second one.** `src/gc.c:1508` reads
`if (dead_slot && !page->region)`, so region pages are never freed; that is what
makes the heap one capability and keeps revocation out of the picture.

`MRUBY_REGION` must be large enough that mruby never falls back to `malloc` for a
page. That fallback is silent and would change what is being measured, which is why
stage 3 reports the page count rather than just OK.

## The four flags that make mruby survive a capability target

Established by `xlang/cheri/mruby-port` for CheriBSD purecap; the same set applies
here, and three of them fail silently rather than loudly.

| flag | what it prevents |
|---|---|
| `-DMRB_NO_BOXING` | `mrbconf.h:62-65` defaults to `MRB_WORD_BOXING`, which packs a pointer into an integer word and truncates it. A static size assertion catches this one. |
| `-DMRB_USE_METHOD_T_STRUCT` | `proc.h` otherwise packs a C function pointer as `(uintptr_t)fn << 2 \| flag`, clearing the tag; the call then traps. |
| `-DPOOL_ALIGNMENT=16` | `src/pool.c` picks 8, and the parser's AST cons cells hold capabilities. |
| `MRB_STR_EMBED_LEN_BITS` 5 -> 6 | source edit: the embedded-string length field is too narrow once a pointer is 16 bytes. |

Plus `mrb_alignas(8)` -> `mrb_alignas(sizeof(void*))` at four sites (`src/proc.c`,
`src/class.c` twice, `mrbgems/mruby-catch`), which the compiler does report.

## Layout

| path | what |
|---|---|
| `mruby_build_config_capstone.rb` | host config: generates presyms, mrblib and the amalgamation with the `default-no-stdio` gembox |
| `tools/gen-amalgam.py` | one translation unit: allocator, `mruby.c`, port -- **in that order** |
| `tools/gen-specimen.sh` | a `.rb` specimen -> `port/md_specimen.h` via the host `mrbc` |
| `port/mruby_domain.c` | the domain entry and the stage ladder |
| `port/capstone_mruby_libc.h` | force-included: the libc names our freestanding headers lack |
| `cases/` | the specimens |

**The amalgamation order is load-bearing.** `mruby.c` contains
`#define malloc(s) mrb_basic_alloc_func(NULL, (s))`, so anything defining its own
`malloc` must precede it or that macro rewrites the definition into a call.

## The ladder

```
MD_STAGE 0   return at once                     entry, cap-init, return channel
         1   + the outer allocator              malloc/realloc/free, narrowed
         2   + mrb_open_core                    a VM on the outer allocator alone
         3   + mrb_gc_add_region                the heap becomes ONE region
         4   + run embedded bytecode            returns what Ruby computed
```

Every stage returns a marker tagged `0x6D52` ("mR"), so a run always yields a result
rather than a wedge. Build one `.dom` per stage and run them in ONE boot, ascending,
control first.

## Status

**The ladder reaches mruby's own VM.** One image, six calls:

| call | what | result |
|---|---|---|
| 0 | anchor, `&domain_main` | returns the load base |
| 1 | entry, cap-init, return channel | **OK** |
| 2 | the outer allocator, narrowed | **OK** |
| 3 | `mrb_open_core` | **cause 7**, a bounds fault on a store |
| 4 | `mrb_gc_add_region` | not reached |
| 5 | run bytecode | not reached |

**RETRACTED, and the retraction is the useful part.** This file previously said the
store landed "one element past a buffer of `nregs` elements", and that our exact
narrowing was what made a purecap-invisible overrun visible. Both were wrong, and
both came from reading three instructions instead of the whole block.

Read as a block, the loop is IN BOUNDS with respect to `nregs`:

```
37cc8:  ldc        a0, 0x30(s7)     ; a0 = c->ci          (offsetof ci = 0x30)
37ccc:  ldc        a1, 0x30(a0)     ; a1 = ci->stack      (offsetof stack = 0x30)
37cdc:  cincoffset a0, a1, s4       ; cursor = stack + stack_keep*32 + 16
37ce0:  cincoffset a1, a1, s8       ; limit  = stack + nregs*32 + 16
37ce4:  sd         zero, -0x10(a0)  ; element k at k*32
37ce8:  sw         zero, 0x0(a0)    ; and at k*32 + 16
37cec:  cincoffsetimm a0, a0, 0x20
37cf0:  bne        a0, a1, 0x37ce4
```

`a0` is a cursor and `a1` the limit, so the last write is to element `nregs-1`. The
offsets are the compiler's, not a guess: `-Xclang -fdump-record-layouts` gives
`mrb_context.ci` = 0x30, `mrb_callinfo.stack` = 0x30, `sizeof(mrb_value)` = 32, and
`llvm-nm` puts 0x37cc8 inside `mrb_vm_run` (0x37ac8-0x37d64). The faulting
instruction is 0x37ce4 and not the reported pc, which is the translation block's
entry; the monitor's own `rs1 = x10, imm = -16, size = 8` identifies it exactly.

What is actually wrong is the CAPABILITY, not the count:

```
Cap mem access OOB: rs1 = x10, cursor = 102172670, imm = -16, addr = 102172660,
                    size = 8, bounds = (10216e460, 10216e4b0)
```

`ci->stack` carries **80 bytes** of bounds and the store is 0x4200 past its base.
That is not an off-by-one; it is a pointer that never had room for the frame.

**The narrowing hypothesis is refuted, by an arm that differs in exactly one
thing.** `MRUBY_NO_NARROW=1` makes `cap_narrow` a no-op, so every capability
carries the whole 2 MiB arena, and the fault reproduces identically -- same
function, same instruction, same 80-byte bounds. The knob is real and not a
no-op: the narrowed image contains two `shrink` instructions and the wide one
contains none. So whatever produces an 80-byte capability, it is not our
allocator, which has no way to make one.

`envadjust` is the obvious suspect and is also NOT it, for the same reason -- with
wide bounds a stale capability would still cover everything. The patch for it is
real and is held, unapplied, in `patches/held/` with the argument written out; it
is a genuine capability-portability defect that a moving `realloc` hides, but it is
not this fault.

**The instrument reports, and the first frame is HEALTHY.** `port/md_probe.c` plus
patch 0003 hand `mrb_vm_run` its stack geometry back through the ladder. The first
frame mruby ever clears:

| | |
|---|---|
| `ci->stack` length | **4096** = `STACK_INIT_SIZE` 128 x `sizeof(mrb_value)` 32 |
| cursor - base | 0 |
| `nregs` / `stack_keep` | 4 / 0 |
| `stbase` length | 4096, and `ci->stack` starts exactly there |
| `ci->stack` base - heap base | 143360, so it IS inside the arena |
| tag | 1 |

So `stack_init` does the right thing and the corruption is later. The fault's own
capability is 80 bytes with the store 512 past its base, and `sizeof(struct RProc)`
is exactly 80 -- the image holds 29 of them (`mrblib_proc`, `neq_proc`, the gem
procs). `mrb_callinfo` puts `proc` at 0x10 and `stack` at 0x30 with a 96-byte
stride, so a `ci` reading 32 bytes low turns `ci->stack` into the real `ci->proc`.
That is a hypothesis, not a result; what is measured is the 80 bytes and the
healthy first frame.

The builtins the probe reads were checked against the backend rather than assumed:
`CapstoneISelDAGToDAG.cpp` selects LCC field 0 for the tag, 2 for the cursor, 3 for
the base and 4 for the end.

**Where it stands: an irreducible contradiction, measured from every reachable
side.** The probe measures the frame `mrb_vm_run` is about to clear as healthy, and
the next four instructions fault on an 80-byte capability. Each line below is a
measurement with a control, not an inference:

| | |
|---|---|
| frame 1 is healthy | `ci->stack` 4096 bytes, cursor at base, `nregs` 4, `stack_keep` 0, `stbase` the same 4096 at the same address, 143360 bytes inside the arena |
| the frame is `mrb_vm_run`'s | the probe reports its call site: 1, not `exec_irep` |
| escaping BEFORE frame 1's clear | no fault; the full 60-rung ladder completes |
| letting frame 1's clear run | fault, before frame 2 is reached |
| disabling BOTH clears | no fault; it hangs instead, on a stack mruby believes it cleared |
| the predicate can fire | ladder rung 4, a fake context carrying a deliberately tiny capability |
| the knobs are really compiled in | `md_knobs` is read back through the ladder, and out of the image itself |
| reading `ci->stack` twice inside the probe | identical, `md_reread_differs` = 0 |
| the probe does not touch the heap | the `malloc` it used to do for the arena base is gone; the domain hands it in |
| the probe preserves the caller | it saves and restores s0-s3 and touches nothing else callee-saved; `mrb_vm_run` holds the context in s7 |
| the domain stack does not overlap the heap | probe stack address is 183059 bytes BELOW the arena, growing away |
| the LCC field indices are right | `CapstoneISelDAGToDAG.cpp`: 0 tag, 2 cursor, 3 base, 4 end |

The probe now takes the context and does the same two loads the clear does --
`c->ci` at slot 3, then `ci->stack` at slot 3 -- so the two are reading the same
words through the same register. **What is left is the four instructions between
`md_probe_stack`'s `ret` and the store, and that is no longer an mruby question.**

Two earlier readings are retracted along the way, and both were retracted by a
knob rather than by an argument. "Disabling the clears does not help" was measured
on a build where `MD_PROBE_SKIP_CLEAR`'s body had been deleted by a rewrite,
leaving only its `#define`; the flag was accepted and did nothing. "The fault is
not in a `vm.c` clear" followed from that and falls with it. `md_knobs` reports the
compiled flags back through the ladder now, because a knob that did not take is
not visible from outside the image.

**The four instructions are a KNOWN SHAPE on this project, and doubling every
`ldc` moves the fault.** The reload is

```
ldc a1, 0x30(s7)     ; rd = a1
ldc a1, 0x30(a1)     ; rs1 = a1, the previous rd
```

which is, byte for byte, the pair `llvm/lib/Target/Capstone/CapstoneLdcRetry.cpp`
was written for: "two ADJACENT `ldc`s where the second's rs1 is the first's rd",
the shape shared by four S-07 wedges in four unrelated functions in four builds.
S-07 itself is a silicon defect in the load path and does not exist under QEMU, so
this is a shape match, not a mechanism match.

Built with `-mllvm -capstone-double-ldc` (MRUBY_DOUBLE_LDC=1), which re-issues every
`ldc` and takes the second result -- and unlike the type-query retry puts nothing
between the pair, so it does not serialise the overlap under test. The knob is
verified in the image rather than assumed: 22410 `ldc` become 38632.

The stack-clear fault is GONE in that build. `mrb_vm_run` runs past it and dies
later in the same function, on a CAPABILITY store (`size = 16`, `imm = 160`) rather
than the 8-byte store of the clear.

**That is a lead, not a verdict, and the pass's own header says why:** an instrument
rich enough to change this shape also perturbs register allocation and scheduling,
so a fault that moves is consistent both with "the first read was delivering
something wrong" and with "the code is simply different now". What it does
establish is that the contradiction sits on a shape this project has already found
trouble in twice, which is where to look next.

**The clear is past, and the next blocker is characterised.**
`MD_PROBE_DO_CLEAR` has the probe perform the clear itself, in C, over the same
capability and the same addresses, and return `stack_keep` so mruby's inlined loop
is skipped. Two results:

* **No fault.** The probe writes the very elements mruby's loop faults on, through
  the capability the probe measured as 4096 bytes, and nothing traps. Same data,
  same addresses, same semantic operation, different instruction sequence -- so the
  data is sound and the emitted loop is not.
* **`mrb_open_core` then HANGS.** Thirty minutes at that rung with no progress, so
  there is a second problem behind the first.

The hang is **not** the VM spinning through frames. Built with the escape set to
fire after 2000 `mrb_vm_run` frames, it never fires in fifteen minutes, so mruby is
stuck in one place rather than looping through the call machinery. That narrows the
next instrument: it has to observe inside a single stuck computation, not count
frames.

**Three probe versions were wrong before one was rightBefore that fault: **stage 0 returned `0x6D520001`.** The 1.4 MB image loads, the domain is created and
entered, `__capstone_cap_init` materialises the capability globals, and the marker
reaches the host. Stages 1 to 4 are the next step; no case has been scored.

Getting there took five build iterations and turned up two real compiler defects,
which is the part worth carrying to the next subject:

| | what stopped it | how it was closed |
|---|---|---|
| 1 | 20 compile errors | the libc header, `mrb_alignas`, `MRB_STR_EMBED_LEN_BITS` |
| 2 | `mruby.c` `#define`s `malloc` | the allocator moved ahead of it in the amalgamation |
| 3 | **segfault in the register allocator** | `SplitKit.cpp` null check where two register classes are disjoint -- the ordinary case here whenever a capability class meets an integer class |
| 4 | **assertion in the legalizer** | mruby's bignum is `unsigned __int128`, and i128 here IS the capability width; recorded, gem dropped |
| 5 | 20 undefined symbols | `mruby-math` dropped, setjmp from the micropython port, `memchr`/`strchr`/`abort`/`trunc`/`round`/`fmod` written |

`trunc`, `round` and `fmod` are built on beebs' `floor` and `ceil` rather than from
scratch, because those already handle the infinities and the NaNs. `fmod` carries a
`ponytail:` note naming its ceiling: it is the textbook identity, so beyond |x/y| of
2^53 it is not the exact IEEE remainder. Fine here, where no specimen computes a
float; not fine for a numeric benchmark.

The census that preceded all of this is in
`agent-handoff/history/28-08-2026_00-30-00_mruby-is-portable-jerryscript-is-not.md`.
It predicted eleven errors from a syntax pass and was a lower bound, as it said it
was: it could not see the link, and it could not see the compiler.
