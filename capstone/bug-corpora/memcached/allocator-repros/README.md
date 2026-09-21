# memcached allocator defect corpus

Defects in which a pointer to storage that one of memcached 1.6.45's two
nested allocators handed out is used after the owner released it. Neither
allocator returns that storage to `malloc` on the path that matters.

`cache.c` is the per-thread object cache: `do_cache_free` (`cache.c:135`)
pushes a freed object on the cache's `STAILQ`, writing the list link into its
first bytes, and `do_cache_alloc` (`cache.c:80`) pops the same object,
uncleared, for the next request. `free()` is called only over a limit the
three per-thread instances -- connection-queue items, read buffers, pending
IOs -- do not have.

`slabs.c` is the size-class allocator under every item: `item_free`
(`items.c:360`) ends in `slabs_free`, and `do_slabs_free` (`slabs.c:501`)
pushes the chunk onto its class's `slots` list. A chunk never reaches `free()`
at all; a page leaves its class only through the mover, which this port does
not build.

    00_7af02b0c87_rbuf_copied_after_cache_free/       a multiget's read buffer copied after it went back
    01_0ad4de66ae_io_walk_reads_freed_link/           a list walk reads the next-link out of a returned IO
    02_59bd02ce29_tail_repair_frees_referenced_item/  the allocator frees an item somebody is holding
    03_a8c4a82787_refcount_overflow_frees_linked_item/ the count stops counting at 65536
    04_152ddb68f7_unlocked_refcount_drift/            an unlocked decrement loses a concurrent get

| shape | cases |
|---|---|
| stale object pointer / cache.c reuse / read through the dead pointer | 0 |
| stale object pointer / cache.c reuse / list link read through the dead pointer | 1 |
| allocator-forced free of a referenced item / slabs reuse / read through the dead pointer | 2 |
| reference count overflow / item freed with holders remaining / slabs reuse / read through the dead pointer | 3 |
| unlocked refcount update / count drifts below the holders / slabs reuse / read through the dead pointer | 4 |

Five cases in five shapes, across both of the allocators the port carries.
Cases 0 and 1 are consumer mistakes in `cache.c`'s object caches, read one
level apart: the freed object's payload, and the freed object's own list link
-- the field the walk needs next is the one the allocator's push and the next
owner overwrite.

Cases 2, 3 and 4 are all items reaching `slabs_free` while somebody is still
holding them, and they differ in *why the count was wrong*: case 2 overwrites
it deliberately, case 3 lets it overflow, case 4 loses an update to a missing
lock. Nothing in the allocator can tell the three apart -- each free looks
correct at every step -- which is the point of having them separately.

**Case 2 is the only one live in the pin.** Cases 0 and 1 reconstruct shapes
upstream has since fixed; case 2 runs the branch that is in the pinned tree,
reachable in the shipped binary with `-o tail_repair_time=N`.

## What was searched, and what was left out

On 2026-09-21 upstream's whole history to the pin -- 2360 commits -- was
searched for consumer-side defects in which a pointer to storage that
`slabs.c` or `cache.c` handed out is used after the owner released it. The
commit messages were filtered on temporal-safety vocabulary (55 hits) and
classified by which allocator owned the storage. The published CVE record was
searched too, through the CVE Program and NVD APIs, and the open issues were
read.

Nine candidates were in scope. Five are built, in five shapes; the rest are
listed with the reason each is not a separate case:

| candidate | why not (yet) |
|---|---|
| `e3b7d33` (2026-07-02), `bc080ab` (2020) | the same overflow as case 3 through the binary and meta protocols; upstream calls the binary one a remote-code-execution path. One defect, three front ends -- cited in case 3's provenance rather than built three times |
| `f4983b2` (2012) and the 2011-12 `do_item_alloc`/`do_item_get` races | that era reused the LRU tail **in place** rather than through `slabs_free`. The reuse never passes the allocator's seam, so a protected arm would not see it either: the case would fail its own oracle, and saying so here is worth more than a case that cannot discriminate |
| `41aa0a5` (2008) | hash corruption in `do_item_alloc`; pre-dates most of the structure the port builds |

Out of scope until another allocator is ported: the page mover (7 defects on
record, including `d67d187`, which frees busy items deliberately), the proxy's
own pools (4), extstore (3), the logger's bipbuffer (2), the response bundles
(1) and the crawler (1).

Rejected outright, with the maintainer's own adjudication:

| report | why |
|---|---|
| #1306, chunked-item `assert` DoS | asserts are in the debug binary only, the path falls through correctly without them, and the PoC does not reproduce: *"There is no issue, no CVE"* |
| #1308, proxy `raw_line()` underflow | reachable only by a privileged user writing a configuration that would never work |
| #1213, `do_cache_alloc` NPD | reachable only on `malloc` failure, because the `io_cache` has no limit -- and a NULL dereference faults in the protected and the spatial arm alike, so it could not tell them apart |
| CVE-2026-90698 | fixed in 1.6.44; the pin is 1.6.45 |

No published CVE is live at the pin. `master` is the pin
(`compare/1.6.45...master` is `identical`), so there are no post-pin fixes to
mine either; what is live is what upstream has chosen not to fix.

### Fixes that were applied in one place and not another

A fix can leave the same defect standing at a sibling call site, and this
project has one confirmed instance of exactly that: the reference-count
overflow was capped for ASCII in 2017 (`a8c4a82787`, case 3), for meta in 2020
(`bc080ab`), and for the binary protocol only in `e3b7d33` -- 2026-07-02, eight
days before the pin, after nine years in which the same defect was reachable
through a different front end. So the pinned tree was searched for further
instances along three axes. All three came back negative, and the checks are
written down here because a negative result nobody can reproduce is worth
nothing:

| axis | what was checked | result |
|---|---|---|
| the `152ddb68f7` unlocked decrement | every `do_item_remove` call site outside `items.c` (11 in `proto_parser.c`, 3 in `proto_bin.c`, 3 in `proto_text.c`, plus `thread.c`, `memcached.c`, `storage.c`, `crawler.c`, `slabs_mover.c`) for whether the item lock is held | none unfixed. `process_marithmetic_cmd`'s error path *looks* like the pre-fix shape -- `do_item_remove(it)` guarded only by `if (it != NULL)`, `item_unlock` guarded by `if (locked)` -- but `item *it = NULL` at declaration and `it` is only assigned by `do_add_delta`, which runs after `item_lock`. Whenever `it` is non-NULL the lock is held |
| the `a8c4a82787` refcount cap | every site that takes an item reference without going through `limited_get`/`limited_get_locked`: `proto_parser.c:802,923,965,1384,1564`, `proto_text.c:869,885`, `proto_bin.c:1160,1322`, `memcached.c:1549,2242` | none reachable. The overflow needs references to ACCUMULATE, which only a multiget does, and all three multiget front ends are capped. The others take one reference and release it in the same function |
| deliberate reference leaks | `debugitem ref`, which leaks one reference per call by design and would accumulate without bound | `#ifdef MEMCACHED_DEBUG` (`proto_text.c:1511`), and `-DMEMCACHED_DEBUG` appears only in `memcached_debug_CFLAGS` (`Makefile.am:133`), never in `memcached_CPPFLAGS`. Not in the shipped binary |

This is a search along named axes, not a proof of absence. What it does
establish is that the one temporal defect live in the pin is not a fix that
was forgotten somewhere: it is case 2, which upstream left in deliberately.

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
* **`poisoncap-spatial` / `poisoncap-protected`** are the contract's own arms
  and are written: the port's PoisonCap adapter, one binary, the mode chosen at
  run time. The spatial arm additionally requires the adapter to report
  `sweeps=0`, because a control that swept would be a second protected arm.
* **`live_in_pin` is `false`** for both cases with the proof beside it: each
  fix is an ancestor of the 1.6.45 tag (GitHub compare `status=behind`), so
  the shipped allocator is exercised by a pre-fix consumer shape the commit's
  own diff shows -- the FFmpeg corpus's tier.
* **Races become sequences.** memcached is a threaded server and case 1 is a
  race; the fixture performs the interleaving the commit message describes in
  program order, and says so.

## Four targets, one sequence

Real: `cache.c` (and `slabs.c`, initialised and idle) from the 1.6.45 pin,
unmodified but for the two patches the port
[`ports/memcached/allocators`](../../../ports/memcached/allocators/README.md)
applies -- and the same two patches on every target, because what changes
between them is the authority under the allocators, never the allocators -- one replaces the includes with the port's shims, the other connects
the free-list transitions to the adapter. Reduced: the consumer. Each `case.c`
writes its sequence inside `MC_CASE(NN)` and is a complete translation unit on
both targets; [`shared/corpus.h`](shared/corpus.h) is the seam.

    shared/build-cases.sh native <out>            then runners/run-native.sh
    shared/build-cases.sh capstone-domain <out>   then runners/capstone-domain/
    shared/build-cases.sh cheribsd <out>          then runners/cheribsd/
    shared/build-cases.sh poisoncap <out>         then runners/poisoncap/

The [domain runner's manual](runners/capstone-domain/README.md) has the
commands, the two modes and the oracle. In short: `spatial` must complete,
`sublet` must fault at the labelled probe, the expected address is published by
the run and never hardcoded, and `--negative-control` must make every oracle
say FAIL before a PASS is believed. The [CheriBSD manual](runners/cheribsd/README.md)
runs the same cases against the platform's own `malloc` with libc revocation on
or off, beside a control that shows the revocation can fire at the same shape.
The [PoisonCap manual](runners/poisoncap/README.md) runs the pair the contract
names, behind the platform's own instruction controls.

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
