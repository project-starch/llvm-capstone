# PostgreSQL nested-allocator defects — inventory at the 17.0 pin

*What upstream PostgreSQL defects are usable as specimens for the nested-allocator
corpus, which allocator each one's memory came from, and which have a protected
arm today. Assembled 2026-09-18 against the port's pin,
`ports/postgres/memory-contexts/upstream.json` = 17.0.*

## Why this class, and what makes a row count

The corpus needs defects where a **consumer** of a nested allocator keeps using
memory whose lifetime has ended, and where the lifetime ended by an **allocator
event that never reaches `malloc`**. That is the whole point: ASan, MTE and
quarantine all key on `free()`, so they are structurally blind here.

Verified in the pinned source, not assumed:

- **`pfree` returns a chunk to an in-context freelist, not to `malloc`** —
  `aset.c:1141-1143` pushes onto `set->freelist[fidx]`; `aset.c:1001-1013` pops
  the same address for the next `palloc` of that size class. The file says so
  itself at `aset.c:20-22`: *"AllocSetFree() normally doesn't free() memory
  really."*
- **`MemoryContextReset` is bulk and per-object-`free`-less** — `aset.c:554`
  zeroes every freelist, `aset.c:559` rewinds to the keeper block, which is not
  returned to `malloc`.
- **`MemoryContextDelete` recycles the context itself** — `aset.c:627-663`, a
  freelist-eligible context is reset and pushed onto `context_freelists[]`
  rather than freed, so even a fresh context can land on the same addresses.
- **Production builds cannot detect a double `pfree` at all.** Upstream added a
  detector in March 2026 (`0c8b4e9cfc`) but only under `MEMORY_CONTEXT_CHECKING`,
  explicitly rejecting a production check because *"that adds measurable
  overhead"* — and noting the hazard had gone unnoticed *"for more than three
  years."*

A row is **out of scope** if it is allocator-internal (a change to `mmgr`
itself), a pure leak, frontend code over plain `malloc` (ecpg, libpq), requires
true concurrency (single hart), or requires the LLVM JIT (no capstone64 backend).

## Only AllocSet is ported

`ports/postgres/memory-contexts` compiles five context types — `aset`,
`generation`, `slab`, `bump`, `alignedalloc` — but **only `aset.c` is ported to
the Sublet discipline**. The four patches touch `aset.c` and
`memutils_memorychunk.h` and nothing else. Under Sublet the other context types
hit a deliberate refusal stub
(`src/allocators/sublet/unsupported-allocators.c`), which says so:

> "Only aset contexts are ported to Sublet; a generation, slab or bump context
> would need its own sub-pool discipline."

The spatial arm has no such restriction — `backing-allocator.c` provides a
general `malloc`/`free`/`realloc`/`calloc` over the region — so a non-aset
specimen can still run native and spatial. It just has no protected arm.

**So each row below names the allocator of the stale object, read from its
creation site.** Inferring it from the subsystem is wrong twice over: TidStore's
`rt_context` is Bump only on the local `insert_only` path (`tidstore.c:185`),
the shared path is AllocSet (`tidstore.c:223`) — and either way the defect's
stale pointer is to the `TidStore` struct itself, `palloc0`'d in
`CurrentMemoryContext` at `tidstore.c:172`/`:220`.

## Specimens live in the 17.0 pin — no revert needed

| # | Fix commit | Consumer | Lifetime ended by | Allocator | Protected arm |
|---|---|---|---|---|---|
| 1 | `1f5b6a5e5d` | `utils/sort/tuplestore.c` | `pfree` → freelist, then a **second** `pfree` of the same chunk | AllocSet | yes |
| 2 | `3549ffb6af` | `commands/vacuumparallel.c` | `pfree` → freelist | AllocSet | yes |
| 3 | `83ce20d671` | same, 2024 instance | `pfree` → freelist | AllocSet | yes |
| 4 | `ed394c4bdf` | `optimizer/util/inherit.c` | `pfree` via `bms_del_member` | AllocSet | yes |
| 5 | `727bc6ac33f6` | `optimizer/path/joinrels.c` | `pfree` via `bms_free` on **parent-owned** sets | AllocSet | yes |
| 6 | `9d5ce4f1a00a` | `executor/nodeWindowAgg.c` | **`MemoryContextReset`** (bulk) | AllocSet | yes |
| 7 | `a61592253e` | `replication/pgoutput/pgoutput.c` | **`MemoryContextDelete`** of an ancestor | AllocSet | yes |
| 8 | `9e0b4b1ab5` | `replication/logical/reorderbuffer.c` | `pfree` → slab `block->freehead` | **Slab** | **no** |

Seven of eight have both arms today. The mix matters as much as the count: five
are `pfree`-into-freelist, two are context reset/delete, one is slab — so both
lifetime-enders are represented, which is the sharper axis.

### Notes per row

**1 — tuplestore, bug #19438.** `dumptuples()` (`tuplestore.c:1209-1225`) writes
and `pfree`s tuples in a loop but only clears `memtupdeleted` after it. If
`WRITETUP` throws — an undersized `temp_file_limit` on a holdable cursor —
`memtuples[]` retains pointers to chunks already on the freelist, and
`tuplestore_end` walks them and `pfree`s each a second time. In 17.x there is no
double-`pfree` detection, so this silently builds a circular freelist that hands
one chunk out twice. Upstream: *"apparently live entries but in fact pointed to
pfree'd chunks."* The context is the portal's hold context,
`portalmem.c:343-346`, an AllocSet.

**2, 3 — `dead_items` after reset.** Two distinct missed pointer updates ten
months apart, the second documented as the residual the first did not cover.
The 2024 instance is the one that is live in 17.0 and already fixed in 17.5, and
it is why the pin is 17.0. Upstream states the blindness mechanism outright:
*"the chunk freed after the last round of index vacuuming was put on the
context's freelist"*, and the bug *"was evident by nudging the system to
allocate memory in a different area."* The 2024 commit is blunter still:
*"apparently happened not to break anything since the freed chunk would have
been put on the context's freelist, so it was accidentally the same pointer
anyway."*

**4, 5 — `Bitmapset` aliasing.** Both go through
`bitmapset.c`, where `bms_del_member`/`bms_del_members`/`bms_free` `pfree` the
set once it empties and return NULL, leaving any second alias dangling. Row 4
is one missing assignment back to a local; row 5 frees sets the *parent*
`SpecialJoinInfo` still owns, once per partition pair. `bitmapset.c` is the one
consumer file that compiles nearly freestanding — four includes, and a link
closure of ten symbols of which eight are already in the port — so these two can
be driven against the real upstream data structure rather than a model.

**6 — WindowAgg, and the clearest statement of the hazard.** On the run-condition
transition the non-top-level branch NULLifies `ecxt_aggvalues[]`, with a comment
explaining exactly why: *"Some of these might be byref types so we can't have
them pointing to free'd memory."* The `top_window` branch
(`nodeWindowAgg.c:2305-2309`) sets pass-through and `continue`s without doing so,
while `release_partition` has already done the bulk
`MemoryContextReset` (`:1354-1356`) with *"We don't rely on retail pfree."* The
code states the hazard and omits the guard on one path. This is the corpus's
context-reset-ended representative.

**7 — pgoutput, hierarchical.** `RelationSyncCache` lives in
`CacheMemoryContext` (backend lifetime) but each entry's payload lives in
`entry_cxt`, a descendant of the decoding context. An error tears the decoding
context down, taking every `entry_cxt` with it, while the cache survives holding
pointers into freed arenas.

**8 — reorderbuffer, and why it has no protected arm.** `ReorderBufferChange`
comes from `reorderbuffer.c:329`
`SlabContextCreate(..., sizeof(ReorderBufferChange))`. Slab is the one allocator
where same-address reuse is *guaranteed* rather than probabilistic — one chunk
size, LIFO `block->freehead` (`slab.c:731-732` ↔ `:277-286`), no size-class
rounding — which makes it the best vulnerable arm we have and useless as a
protected one until slab is ported.

## Also live, not a row

`1d0fc2499ff2` (`ReinitializeParallelDSM`, AllocSet, context-lifetime mismatch)
is live at 17.0 but self-declared unreachable: *"there is no way to reach a crash
using only code that is present in core PostgreSQL."* Recorded so it is not
re-derived.

## Ruled out, so they are not re-derived

- `0c8b4e9cfc` / `a26ac902f6` — the double-`pfree` detector itself. Allocator-
  internal, so not a specimen; but it is the best available evidence that 17.x
  cannot detect this class in production, and it is quoted above for that.
- `53cb4ec1de` — "relid-set clobber during join removal". The `bms_del_member`
  that frees `em_relids` is present in 17.x, but the read path is not:
  `remove_rel_from_restrictinfo` touches only `clause_relids` and
  `required_relids` and `bms_copy`s both first. The dereference was introduced
  by a sibling commit in the same April 2026 batch. **Not reachable at 17.x.**
  UNRESOLVED residual: whether `em_relids` can alias `clause_relids` rather than
  a side's relids, which would make the `bms_copy` itself a stale read. Settling
  it means reading `add_eq_member`/`process_equivalence` in `initsplan.c`.
- ecpg fixes (frontend, plain `malloc`), JIT-inlining fixes (no capstone64 JIT),
  GIN/GiST "dangling TID" fixes (index entries, not pointers), and the
  replication-slot/shmem family (needs concurrent backends).

## How this list was built, and what the instrument gets wrong

A narrow keyword grep (`use-after-free|dangling pointer|already freed|…`) over
`REL_17_0..REL_17_STABLE` returns 8 commits and **misses real specimens**. Two of
the eight rows above are invisible to it:

- row 1's subject is "Be more careful to preserve consistency of a tuplestore";
- row 6's body says *"pointing to free'd memory"* — **`free'd` does not match
  `freed`**.

A wide vocabulary (adding `pfree`, `palloc`, `memory context`, `stale`,
`no longer valid`, `clobber`, `consistency of`, `lifetime`, `still points`)
returns 61 commits over the same window and yields all eight, with 46 rejected
for the reasons above. The single highest-yield term is **`stale`**. Treat any
count from the narrow pattern as a lower bound with unmeasured recall.

## Reaching further

The revert route reaches defects whose fix is present in the pin. Measured
against 17.5 with `git apply -R --check` over the 100 keyword-matched master
commits: 80 have their fix present, and **only 8 reverse-apply cleanly** — a 10%
rate, because the surrounding code has moved over 25 years. Those 8 are not yet
triaged. Note that a clean reverse-apply proves the patch applies, not that the
revert re-introduces the defect; a fix may have been superseded.

Version choice was measured the same way and does not help: no single pin
reaches 30-50 in-scope specimens. 16.0 is a wash on count and breaks the port
(patch 0004 does not apply; `aset.c` drifted 545 lines across 19 hunks), and
below 16 there is no `MemoryChunk` abstraction at all — `memutils_internal.h`
and `memutils_memorychunk.h` arrive with `c6e0fe1f2a` (2022-08-29), after 15
branched — so patches 0002 and 0003 have no target. **16 is a hard floor and
17.0 is the right pin.** Breadth beyond ~10-12 specimens has to come from other
applications, not an older PostgreSQL.
