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

Bring-up in progress. The census that preceded it is in
`agent-handoff/history/28-08-2026_00-30-00_mruby-is-portable-jerryscript-is-not.md`;
nothing here has been measured yet, and no case has been scored.
