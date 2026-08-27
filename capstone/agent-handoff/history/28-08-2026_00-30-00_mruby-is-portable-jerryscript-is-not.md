# mruby will port; JerryScript will not, and the reason is the same one

Assessed 2026-08-28 while looking for a source with MANY bugs that standard CHERI
cannot catch. The criterion is in
`ref/nested-allocator-bug-inventory.md`: the bad access must stay inside the region
the nested allocator owns, because CHERI narrows a capability at `malloc()` and a
nested allocator carves objects out of one pool with no `csetbounds`.

## Why mruby is the density source

Structure read in source, not assumed:

```c
typedef struct mrb_heap_page {
  RVALUE *freelist;
  ...
  RVALUE objects[MRB_HEAP_PAGE_SIZE];
} mrb_heap_page;

typedef struct mrb_heap_region {
  uint8_t *base;               /* "from contiguous region, not malloc" */
```

Every object is carved from `objects[]` inside a page, and allocation is
`p = freelist; freelist = p->as.free.next` -- **the free list runs through the
objects themselves.** So a use-after-free write corrupts the allocator's own
metadata, and both the object and the metadata stay inside the page. That is the
target class, by construction, for every heap bug in the tracker.

Tracker counts, taken with a control query that returns zero: **90 heap-buffer-overflow
and 82 use-after-free issues**, and mruby is an OSS-Fuzz project, so the reports
carry scripts.

## The port cost, measured

`rake amalgam` produces ONE `mruby.c` of 112,132 lines plus a separate
`mruby_compiler.c`. That matters twice: the gp-captable ABI requires a single
translation unit, and the compiler can be left out entirely if scripts are turned
into bytecode on the host, which leaves only the VM and the GC in the image -- which
is all we want to measure anyway.

Compiling that amalgamation for `capstone64` gives **11 errors**:

| what | sites | fix |
|---|---|---|
| `MRB_STR_EMBED_LEN_BITS 5` -- embedded-string length in 5 bits, and a 16-byte pointer makes the space 59 bytes | 1 | `#define` to 6 |
| `mrb_alignas(8)` on `const struct RProc` | 4 | `sizeof(void *)` |
| `ERANGE`, `HUGE_VAL`, `EXIT_FAILURE` | 4 | libc shim |
| `gc.c:508` `base = (uint8_t*)offset;` after aligning, in the contiguous-region path | 1 | `base += offset - (uintptr_t)base` |
| `symbol.c:73,88` low-bit tagging: `(const char*)((uintptr_t)ptr \| FLAG)` | 3 | a separate flag rather than the bit |

Plus `-DMRB_NO_BOXING`: word boxing is the default and compresses pointers into
integers. A static size assertion is what caught that, which is the good case.

The first two and the last two are the same defect this project has now fixed three
times: **a constant encoding the pointer's width without saying so.**

## JerryScript, for contrast, and it is a result rather than a complaint

`struct jmem_heap_t { jmem_heap_free_t first; uint8_t area[]; }` -- one contiguous
area, everything carved from it, so it is the *sharpest* CHERI-blind source of the
three. It does not run here and cannot cheaply be made to:

```c
const uintptr_t heap_start = (uintptr_t) &JERRY_HEAP_CONTEXT (first);
uint_ptr <<= JMEM_ALIGNMENT_LOG; uint_ptr += heap_start;
return (void *) uint_ptr;         /* untagged -> cause 24 */
```

93 sites across 60 functions, and `uintptr_t` cannot be made capability-wide on this
target (clang's `TargetInfo::IntType` has no 128-bit member). See
`19-08-2026_20-15-00_jerryscript-carve-and-uintptr.md`.

**The design that makes an allocator invisible to CHERI is the design Capstone's tag
model refuses outright.** Compressed pointers hide an allocator from bounds checks
precisely because nothing is a pointer any more. That belongs in the paper; it does
not belong in the measurement, because it will not run.

mruby sits in the useful middle: pooled like JerryScript, real pointers like
MicroPython.

## Comparison

| engine | pool | pointer model | port cost | heap issues |
|---|---|---|---|---|
| MicroPython | one area, allocation-table bitmap | real pointers (`MICROPY_OBJ_REPR_A`) | 1 site, **runs today** | 13 + 9 |
| **mruby** | `RVALUE objects[]` per page, free list through the objects | real pointers (`boxing_no.h`) | **~9 sites, 5 patches** | **90 + 82** |
| JerryScript | one `uint8_t area[]` | compressed offsets | 93 sites, **blocked** | 58 + 30 |

## Not yet done

The 11 errors are from `-fsyntax-only`. Nothing has been linked, no domain has been
built, and nothing has run. The count is a lower bound on the work: link-time and
runtime defects are not visible to a syntax pass, and WAMR's own bring-up took four
patches that no census predicted.
