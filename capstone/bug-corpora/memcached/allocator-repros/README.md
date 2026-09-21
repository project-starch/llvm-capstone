# memcached allocator defect corpus

Consumer-side defects in code that allocates from memcached 1.6.45's
per-thread object cache, `cache.c`. A freed object does not go back to
`malloc`: `do_cache_free` (`cache.c:135`) pushes it on the cache's `STAILQ`,
writing the list link into its first bytes, and `do_cache_alloc`
(`cache.c:80`) pops the same object, uncleared, for the next request. `free()`
is called only over a limit the three per-thread instances -- connection-queue
items, read buffers, pending IOs -- do not have.

    00_7af02b0c87_rbuf_copied_after_cache_free/   a multiget's read buffer copied after it went back
    01_0ad4de66ae_io_walk_reads_freed_link/       a list walk reads the next-link out of a returned IO

| shape | cases |
|---|---|
| stale object pointer / cache.c reuse / read through the dead pointer | 0 |
| stale object pointer / cache.c reuse / list link read through the dead pointer | 1 |

Two cases, one shape in two readings: of the freed object's payload, and of
the freed object's own list link -- the field the walk needs next is the one
the allocator's push and the next owner overwrite.

## Why there is no slabs case

The port carries both of memcached's allocators, and its slab arm -- chunks
on a class's `slots` list, never `free()`d -- is exercised by the port's
example and smoke run in both modes. It has no corpus case, and this is why.

On 2026-09-21 upstream's history was searched (GitHub commit search over the
repository for *use-after-free*, *use after free*, *dangling*, *freed item*,
*double free*, *after free*, *refcount leak item*, *item_remove crash*,
*segfault item*; issue search for the same) for a consumer-side defect in
which an **item pointer outlives `slabs_free` and meets the chunk's next
tenant**. Every candidate turned out to be something else:

| commit | what it is | why not |
|---|---|---|
| `8caa4146a5` (2019) *close delete + incr item survival bug* | DELETE fetched, unlocked, and an `incr` replaced the item in between | DELETE's reference keeps the refcount up, so the item is never freed while held; a linkage race, not a stale-storage access |
| `f4983b2068` (2012) *Fix a race condition from 1.4.10 on item_remove* | unprotected refcount tests against the LRU tail | that era's eviction reused the tail **in place** (`do_item_alloc`: `it = search; it->refcount = 1;`), never through `slabs_free`; the reuse a stale reference met was not the slab free list's |
| `c0e5a99745` (2020), `2b97c389f0` (2024), `c65a2fbb13` (2017) | the page mover frees a chunked header the wrong way, changes a CAS during a rescue, unlinks a chunk mid-write | all in `slabs_mover.c`'s territory, the one place a chunk's storage changes class, which the port does not build |
| `b031143f8a` (2020) *Fix over-freeing in internal object cache* | `cache.c`'s own limit test inverted | the allocator's bug, fixed at the pin; not a consumer's |

The item layer's reference counts under per-key locks -- atomics since
`f4983b2068` -- are why: in 1.6.x an item reaches `slabs_free` only when its
last reference is dropped. That is what was searched, not a proof of absence.
The corpus records the boundary rather than manufacturing a case to fill it.

## The contract

The layout and the `case.json` fields are the corpus contract in
[`cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md),
referenced rather than copied. Where this corpus differs:

* **`native-fix-differential`** is an extra axis beside protection: natively
  the pair differs by whether the upstream fix is applied, and the driver
  prints what the case observed. In a domain the pair differs by protection,
  and the program prints nothing.
* **`native-detect`** is not merely unwritten: the objects never reach
  `malloc`'s `free`, so ASan has no event at the push.
* **`cheribsd-revocation`** is an arm the contract does not name: stock
  CheriBSD with its own libc revocation, pages and objects from the platform's
  `malloc`, mode 0 only, with a positive control in the same boot. It is the
  APR and FFmpeg corpora's arm of that name.
* **`poisoncap-*`** are declared and not written: no PoisonCap build of
  memcached's allocators exists.
* **`live_in_pin` is `false`** for both cases with the proof beside it: each
  fix is an ancestor of the 1.6.45 tag (GitHub compare `status=behind`), so
  the shipped allocator is exercised by a pre-fix consumer shape the commit's
  own diff shows -- the FFmpeg corpus's tier.
* **Races become sequences.** memcached is a threaded server and case 1 is a
  race; the fixture performs the interleaving the commit message describes in
  program order, and says so.

## Three targets, one sequence

Real: `cache.c` (and `slabs.c`, initialised and idle) from the 1.6.45 pin,
unmodified but for the two patches the port
[`ports/memcached/allocators`](../../../ports/memcached/allocators/README.md)
applies -- one replaces the includes with the port's shims, the other connects
the free-list transitions to the adapter. Reduced: the consumer. Each `case.c`
writes its sequence inside `MC_CASE(NN)` and is a complete translation unit on
both targets; [`shared/corpus.h`](shared/corpus.h) is the seam.

    shared/build-cases.sh native <out>            then runners/run-native.sh
    shared/build-cases.sh capstone-domain <out>   then runners/capstone-domain/
    shared/build-cases.sh cheribsd <out>          then runners/cheribsd/

The [domain runner's manual](runners/capstone-domain/README.md) has the
commands, the two modes and the oracle. In short: `spatial` must complete,
`sublet` must fault at the labelled probe, the expected address is published by
the run and never hardcoded, and `--negative-control` must make every oracle
say FAIL before a PASS is believed. The [CheriBSD manual](runners/cheribsd/README.md)
runs the same cases against the platform's own `malloc` with libc revocation on
or off, beside a control that shows the revocation can fire at the same shape.

One thing the first run taught, now in `corpus.h`: in this emulator,
arithmetic on a revoked alias faults at the arithmetic (`cincoffsetimm with an
UNTAGGED rs1`), not only a load through it. A case that computed
`&stale->field` after the free faulted before its marker, and the oracle
refused the run -- correctly. Probe addresses are taken while the pointer is
live, and comparisons are of addresses, never of pointers.

## Where this corpus deviates from the contract, and why

`tests/check-corpus.py` in the pymalloc corpus enforces
[SCHEMA.md](../../cpython/pymalloc-repros/SCHEMA.md). Run against these cases it
reports three kinds of problem, all deliberate. They are listed here rather than
silenced, and no copy of that checker is shipped beside them: a fork would be a
second contract, and a checker that fails by design is noise.

| what it reports | why |
|---|---|
| `arm 'cheribsd-revocation' is not in SCHEMA.md` | the contract's CheriBSD arms are the PoisonCap pair. This one has no adapter at all: it is the platform as shipped, and calling it `poisoncap-spatial` would claim an adapter that is not in the binary |
| `arm 'native-fix-differential' is not in SCHEMA.md` | the contract's arms differ by **protection**, the defect present in both. This pair differs by whether the **upstream fix** is applied. Folding it into `spatial`/`sublet` would misname it |
| `case.c declares no PYC_CASE` | the macro is the corpus's seam to its allocator; here it is `MC_CASE`. The rule the checker means -- a case declares the number its directory carries, and the driver refuses a fixture that names another -- is implemented |

Extending the checker to know these is a change to the pymalloc corpus and
belongs in a conversation with it, not a unilateral edit from here.
