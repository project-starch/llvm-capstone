# Nested-allocator blind spots: the taxonomy, and how to search with it

*The classification scheme for the corpus. `sqlite-bugs.csv` is its first filled
cell; `mruby.md` is the second. The mechanism it rests on, measured, is in
`sqlite-arenas.md`. This file is the search plan, not a result.*

## The one idea

**Whether memory-safety hardware catches a bug is decided not by the bug's class
but by which allocator owned the object.**

Bounds come into existence at allocation and revocation acts on what was passed
to `free`. Software that runs its own allocator on top of the system one moves
its objects out of that instrument's reach, and mature C software does this as a
matter of course, for speed. So the bug classes most often reported — use after
free, uninitialised read — land disproportionately in the region the hardware
cannot see.

**What is NOT ours to claim.** That pool allocators defeat quarantine-based
temporal safety is a *documented* limitation. Presenting it as a discovery would
not survive review. The contribution is quantitative: **what fraction of the
real, reported defect history of mature software falls in that region**, and the
demonstration that the same defect flips verdict when only the allocator changes.

## Inclusion rule

A case is in scope when both hold:

1. **Allocator.** The affected object's memory comes from an allocator other
   than the system allocator.
2. **Recycling.** That memory is reused without a `free`/`malloc` pair at the
   object's granularity — so nothing marks the moment the object died.

and the failure mode is one of:

- **temporal-reuse** — access after the object was returned to the arena
- **spatial-intra-arena** — write crossing from one sub-object into a neighbour
  inside the same underlying allocation
- **uninitialised-recycled** — read of arena memory still holding the previous
  occupant's bytes (on CHERI these bytes can be a *still-tagged* capability, so
  "junk has no tag, the load traps" does not hold)

## Control classes — mandatory, not leftovers

These are excluded from the result set and **required in every measurement run**,
because without them a null result is unfalsifiable:

| Control | Why it must be there |
|---|---|
| Out-of-bounds **past** the arena block | bounds checking must fire; if it does not, the harness is broken |
| **Double free** through the real allocator | revocation must fire |
| Same bug class on a **real malloc** object | the matched pair that isolates the allocator as the variable |
| **NULL deref** | traps everywhere; shows the vehicle reports faults at all |

The mruby measurement needed three of its four controls to catch a setup error
before it produced a usable verdict; every verdict before the vehicle control was
added had been a miss.

## Facet 1 — arena owner

| Value | Meaning | Status |
|---|---|---|
| `sqlite-arena` | lookaside, pcache1 bulk, memsys5, btree scratch | **filled** — 396 rows, 101 in scope |
| `host-runtime` | the language runtime the binding lives in | **started** — mruby measured; CPython/Go identified, not collected |
| `app-custom` | an application's own pool, incl. `SQLITE_CONFIG_MALLOC` replacements | empty |
| `system-malloc` | ordinary heap | **control group** |

## Facet 2 — arena discipline

How the arena recycles decides what a stale pointer sees, so it is not a detail:

| Discipline | Mechanism | What a stale pointer gets | Examples |
|---|---|---|---|
| `freelist` | fixed slots on a per-size free list | a **different live object** | SQLite lookaside, CPython obmalloc, SLUB |
| `region` | bump allocation, released wholesale | its own **stale but untouched** data until teardown | nginx `ngx_pool_t`, APR `apr_pool_t` |
| `gc-sweep` | collector reclaims slots | a different object, at a time the program does not choose | Ruby/mruby pages, Lua, V8 |
| `size-class` | bins per size class over large chunks | a different object of the **same size class** | Zend MM, jemalloc-style |

`region` is the extreme case: there is no per-object free at all, so a
use-after-free inside one request is invisible for the request's whole lifetime.

## Facet 3 — boundary role

Reuses the directions already in `benchmarks/sqlite/cve-repros/api-classification.csv`
(82 functions with their lifetime obligations):

| Value | Meaning | Count in that file |
|---|---|---|
| `internal` | object never crosses an API boundary | — |
| `E->H borrowed` | library lends memory; caller keeps it too long | 21 |
| `H->E owned` | caller hands memory over; frees it too early | 18 |
| `CB context` | caller registers a pointer, library calls back later | 15 |
| `E<->H handle` | shared handle lifetime | 19 |

**Read this facet backwards to generate candidates.** Each obligation is a rule;
each rule can be broken; a broken rule is a bug shape. That turns the search from
hunting for bugs into enumerating contracts and asking who violated each — and it
also shows where nobody has looked yet.

## Facet 4 — verdict

`blind` / `covered` / `arena-decides`, already carried as `cheri_expectation`.
Every row must also carry **how** the verdict was reached (`verification`):
measured > object traced to its allocator > inferred from the file > unverified.
No row in the corpus is measured yet.

## The survey axis: start at allocators, not at bugs

For each project: name the arena, state how to tell whether an object is in it,
then intersect its public defect history with that test.

| Project | Arena | Discipline | In-arena test | Defect history |
|---|---|---|---|---|
| **SQLite** | lookaside, pcache1, memsys5, btree scratch | freelist | file calls `sqlite3Db*` not `sqlite3_malloc` | done: 396 rows |
| **OpenSSL** | its own caching buffer freelist | freelist | `OPENSSL_malloc` + freelist path | CVE history. **Heartbleed is the canonical case — see below** |
| **CPython** | obmalloc arenas/pools, plus per-type freelists | freelist | allocation < 512 bytes; or a type with its own freelist | GH issues, CVEs |
| **Ruby / mruby** | GC heap pages, slots on a free list | gc-sweep | object is an `RVALUE` slot | mruby tracker — `ary-delete` already measured |
| **PHP** | Zend MM chunks → pages → bins | size-class | small allocation | php bug tracker, CVEs |
| **nginx** | `ngx_pool_t` per request | region | `ngx_palloc` | nginx advisories |
| **Apache / APR** | `apr_pool_t` | region | `apr_palloc` | httpd CVEs |
| **Lua** | own GC over `realloc` | gc-sweep | `GCObject` | Lua bug list |
| **V8 / SpiderMonkey** | generational GC heaps | gc-sweep | JS heap object | Chrome/Mozilla security bugs |
| **GLib** | GSlice (slab-like; disabled by default since 2.76) | size-class | `g_slice_alloc` | GNOME bugs |
| **Linux kernel** | SLUB/SLAB per-cache freelists | freelist | `kmem_cache_alloc` | CVEs, syzkaller |

The intersection is the work; both halves already exist publicly.

## The precedent this idea already has: Heartbleed

Worth stating early in any write-up, because it is the strongest evidence that
the idea matters and it is not ours to claim as new.

OpenSSL kept an application-level caching freelist so that repeated
same-size allocations would not go back to the system allocator. Two documented
consequences, and both are exactly the mechanism in this taxonomy:

- **Detection was blocked.** Because freed buffers were not returned to the
  underlying allocator, tools that find use-after-free and double-free could not
  see them. The bug "would have been utterly trivial to detect when introduced"
  had it been tested against an ordinary malloc that actually frees.
- **Severity was amplified.** The overread returned OpenSSL's own recycled
  buffers, which is why it leaked interesting data rather than zeroes, and
  OpenBSD's founder stated that Heartbleed "would not have worked at all" had the
  allocator sanitised memory.

So a nested allocator has already, once, turned a bounds bug into an
industry-scale incident *and* hidden it from the instruments meant to catch it.
Our contribution is not to point that out again; it is to measure how much of a
mature codebase's defect history sits in the same position.

Sources: dwheeler.com/essays/heartbleed.html; flak.tedunangst.com/post/analysis-of-openssl-freelist-reuse;
wiki.openssl.org/index.php/Memory_management_internals

## Where the corpus stands

| Cell (owner × discipline) | Rows | Measured |
|---|---|---|
| sqlite-arena × freelist | 101 in scope, 14 with CVE + named function | 0 |
| host-runtime × gc-sweep | 1 (mruby ary-delete) | **1** |
| host-runtime × freelist | identified only (CPython, Go) | 0 |
| everything else | empty | 0 |

**One measured case in the whole corpus.** That is the honest state, and it is
the reason the next step is a measurement and not more collection.

## Sharpest available result

CVE-2024-0232, the JSON parser use-after-free, changes verdict with **payload
size alone**: `dbReallocFinish` recycles a lookaside block into the freelist but
calls real `realloc` on a heap block (`src/malloc.c:715`). One proof of concept,
two payload sizes, opposite outcomes predicted in advance. An existence proof
that the allocator and not the defect decides is worth more than another hundred
classified rows.
