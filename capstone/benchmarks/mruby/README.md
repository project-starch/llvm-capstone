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

**The stage-3 fault may be a result rather than a port defect, and that has to be
settled before it is called either.** It is in `mrb_vm_run`'s `stack_clear` after
`stack_extend`:

```
37cec:  addi       s4, s4, 0x10       ; s4 = nregs*32 + 16
37cf0:  cincoffset a0, a1, s4
37cf8:  sd         zero, -0x10(a0)    ; address = a1 + nregs*32
```

`slli s4, s4, 0x5` upstream confirms `sizeof(mrb_value) == 32` here, against 24 on
stock 64-bit. The store lands one element past a buffer of `nregs` elements.

Why this target sees it and CheriBSD does not is the interesting part: `cap_heap`
narrows every allocation to the **exact** request, while a purecap `malloc` rounds
bounds up for representability. A one-element overrun is therefore invisible there
and visible here. That is the narrowing discipline this file argues for, doing its
job -- but it could equally be our own sizing, so the next step is to read
`stack_extend` against the emitted code rather than assume either.

Before that fault: **stage 0 returned `0x6D520001`.** The 1.4 MB image loads, the domain is created and
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
