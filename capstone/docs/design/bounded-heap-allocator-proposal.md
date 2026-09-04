# Proposal: a real bounded heap allocator (`malloc`/`free`) with C1 narrowing via `SHRINK`

<!-- Decision (see §3): the allocator narrows with `SHRINK`, not `SPLIT`. -->


*Status: DESIGN PROPOSAL — decisions reviewed 2026-07-01 (§7); **both stated
blockers now CLEARED** (2026-07-04 update), so phase 1 is ready to implement.
Grew out of the 2026-06-29 granularity/provenance audit, which found our "heap
narrowing" is **not** a real allocator policy — only two benchmark-local bump
allocators narrow. Tracked as task #78.*

> **2026-07-04 update — blockers cleared.** The two dependencies this proposal
> was gated on have both landed: (1) the untagged-`ldc`/`stc` 128-bit-preservation
> fix (SQLite gaps 3–4) is **implemented + merged**, so a `realloc`/coalesce copy
> that relocates capability-bearing blocks now round-trips tags correctly; (2)
> revocation is **end-to-end** (`revocation-enforcement-proposal.md` §8a,
> `history/03-07-2026_00-00-07_...`), so the optional **revoke-on-free** temporal
> path (§2 "poisoning", open item 5) is now viable. The §4/§5 "blocked" notes
> below are kept for provenance but are **superseded** — nothing external now
> gates phase 1 (spatial) *or* a phase-2 (temporal) follow-on.

## 0. Summary

Today, per-allocation capability bounds exist only as **two benchmark-local bump
allocators** (`rv8_malloc.c`, dtoa's `malloc_beebs`) that call `cap_shrink` on the
returned pointer; both `free` as a no-op and never reuse memory. The audit is
explicit that this must **not** be called "heap narrowing, default on." This
proposal is for a **single reusable bounded allocator** — real `malloc`/`free`
with block reuse — that returns each allocation as a capability narrowed to
exactly its requested size, so any over-read/over-write past an allocation
**faults**, via `SHRINK` (a `SPLIT`-based "root-elimination" allocator was
considered and **rejected** for this use — §3 — with root-elimination deferred to
a separate startup-partition idea). It is a **small purpose-built** allocator, not
a port of an existing one (§2). The narrowing half is implementable now; a fully
general `realloc`/coalescing that must **copy blocks containing capabilities** is
**blocked on the same QEMU limitation as SQLite gaps 3–4** (see §5).

## 1. Motivation — why the current state is not a heap contribution

- `rv8_malloc.c` (`benchmarks/rv8/adapted/`): bump pointer over a static arena,
  16-byte header per allocation, `free` = no-op, `realloc` copies via an inline
  byte loop through the **wide arena capability** (the narrowed user pointer
  can't reach its own header). Returns `cap_shrink(p, b, b+n)`.
- dtoa `malloc_beebs` (`build-beebs-dtoa-capstone.sh`): same shape, injected by
  `sed`.
- **What's missing:** no free-list, no reuse, no coalescing, no general
  `realloc`; each is per-benchmark and arena-sized by hand. There is no allocator
  a *real* program (SQLite, a libc consumer) could link and exercise malloc/free
  churn against — so we cannot claim heap object-granularity as a property of the
  system, only of two toy arenas. Closing this converts C1's heap row from
  "two benchmark allocators" to "a real bounded allocator, measured."

## 2. Design — `cap_bump` → `cap_heap` (bounded free-list allocator)

**Implementation base decision (2026-07-01): small purpose-built allocator, not a
port of jemalloc/tcmalloc/glibc.** Those are large, thread/arena-aware, and
coupled to `mmap`/`sbrk`/OS — we are freestanding over a static arena, so porting
their OS layer dwarfs the allocator itself and destroys auditability. More
fundamentally, the Capstone requirement is **pervasive, not a wrapper**: every
returned pointer must be `SHRINK`-narrowed, and a narrowed user pointer **cannot
reach its own metadata**, so *all* internal metadata access must be re-derived
from the wide arena capability. In a dense allocator (dlmalloc's
`mem2chunk`/`chunk2mem`, footer reads via pointer subtraction) every internal
pointer manipulation is a place the capability model breaks and must be surgically
rewritten — and its correctness assumes ordinary pointer arithmetic. A clean
~300-line first-fit + boundary-tag-coalescing allocator where we control every
derivation is *easier to get correct* than retrofitting, and it is the
security-critical path. The paper claim is "bounded per-allocation capabilities,"
not allocator throughput. **Credibility-upgrade fallback if a reviewer demands a
"real" allocator:** port **dlmalloc** (single-file, boundary-tag,
`MORECORE`→static arena) or **picolibc nano-malloc** (smaller, embedded-oriented)
— a stretch goal, not v1.

A single self-contained TU (`benchmarks/adapted/cap_heap.c`, header
`cap_heap.h`) providing `malloc`/`free`/`realloc`/`calloc` over a
16-byte-aligned static arena, drop-in for the two existing allocators and
linkable by SQLite once the QEMU copy issue (§5) is resolved.

- **Metadata off the user pointer.** Each block has a 16-byte header
  `{ size_t req; size_t flags_and_link; }` at `payload-16`, keeping the payload
  16-aligned (required so a stored capability keeps its tag). The header is
  reached **only through the wide arena capability** (offset math on
  `cap_get_cursor`), never through the narrowed user pointer — exactly the
  `rv8_malloc` realloc trick, generalized.
- **Narrowed return.** `malloc(n)` returns
  `cap_shrink(p, cursor, cursor+n)` — bounds `[p, p+n)`, so over-read/over-write
  past the allocation faults. (Exact in this QEMU at all sizes; a store/reload
  may round outward by one representable grain for `n ≥ 4 KiB` under a faithful
  128-bit encoding — `capability-bounds-model.md`.)
- **Real `free` + reuse.** Segregated or first-fit free list threaded through the
  header link field; freed blocks are reused; adjacent frees coalesce. Because
  `free(p)` receives a **narrowed** capability, it recovers the block header via
  the arena capability (offset of `p` within the arena), not through `p` — this
  is the one Capstone-specific wrinkle vs a stock allocator.
- **Poisoning on free (optional, strong).** On `free`, additionally revoke/clear
  so a retained stale user capability cannot be re-dereferenced. Full temporal
  safety needs the revocation path (task #70, currently dormant); until then
  `free` gives spatial narrowing + reuse but **not** use-after-free protection —
  state this honestly.

## 3. `SPLIT` for the allocator — considered and rejected (use `SHRINK`)

**Decision (2026-07-01): the allocator uses `SHRINK`, not `SPLIT`.**

*Origin of the idea, for the record:* the `SPLIT` **instruction** is from the
Capstone spec (`capstone-spec/parts/cap-man-insn.adoc`; surfaced in
`capability-bounds-model.md` — a true capability split into two adjacent halves,
in the ISA but not wired into LLVM). The **"root-elimination via trusted SPLIT"**
framing is from the internal 2026-06-29 audit, not a published paper. Applying it
to a heap allocator was this author's extrapolation; there is **no literature
precedent for a "SPLIT-based allocator."**

Why `SPLIT` is the wrong fit **for a heap allocator specifically**:
- A `SPLIT`-based `malloc` would carve a chunk off the arena root and hand the
  caller only that narrow linear cap; `free` would have to **merge** it back. But
  `SPLIT` consumes the linear source and yields two halves the allocator must then
  **track and recombine** — strictly more bookkeeping (linear-cap management,
  merge-on-free) than integer-offset math over a static arena.
- Its only benefit over `SHRINK` is "root elimination" — but that helps **only**
  if application code could otherwise reach the allocator's broad root. The
  allocator is **trusted and holds the root in its own private state anyway**, so
  `SHRINK` already delivers the spatial property; `SPLIT` would defend only a
  contrived "in-domain actor reaches the allocator's root cap" threat. Marginal
  benefit, real complexity → not worth it per allocation.

**Where root-elimination *does* belong (separate idea, not this allocator):** a
**one-time trusted startup partition** — carve the domain's initial broad root
into disjoint code/globals/stack/heap regions at boot so no single ambient root
spans everything. That is the defensible Capstone-distinctive claim; it needs the
`int_capstone_cap_split` intrinsic (backend work) and should be its own proposal,
not folded into `malloc`. Tracked as a future item, out of scope here.

## 4. What's implementable now vs blocked

- **Now (no QEMU dependency):** the `SHRINK` bounded allocator with real
  free-list reuse and coalescing, for programs whose `realloc`/copy paths either
  don't move capability-bearing blocks or move them within an aligned arena where
  a byte copy of *non-capability* payload is fine. This already covers the RV8/
  dtoa consumers and lets us **measure** a real allocator (feeds C1, task #75).
- **Blocked on QEMU (tasks #73/#74):** a *fully general* `realloc`/coalescing
  `memcpy` that relocates blocks **containing capabilities** hits the exact
  untagged-`ldc`/`stc` limitation — no in-domain copy preserves both data and
  tags. So a general-purpose malloc that SQLite can lean on for capability-bearing
  reallocation waits on `untagged-cap-loadstore-preservation-proposal.md`. The
  allocator design should be written so that the day untagged `ldc`/`stc` become
  bit-preserving, `realloc`'s block copy becomes correct with no allocator change.
- **Temporal safety (use-after-free)** waits on revocation (task #70).

State all three honestly in the doc and the paper; the immediate deliverable is
**spatial** heap safety + a measurable real allocator, not temporal.

## 5. Interaction with the two open QEMU items

This proposal deliberately **does not** re-introduce a cap-copying `memcpy`. It
depends on:
- `untagged-cap-loadstore-preservation-proposal.md` for a correct capability-
  bearing `realloc` copy (else `realloc` of a block holding pointers corrupts
  them — gap 4).
- `revocation-enforcement-proposal.md` for use-after-free poisoning on `free`.

Both are `capstone-qemu` author decisions. The allocator can land and be measured
on the `SHRINK`-narrowing + spatial-only path before either resolves.

## 6. Test plan

- **Unit / authority probes** (`tests/capstone-authority/`, matches existing
  `heap_oob` style): `malloc(n)` then read/write at `[n]` **faults**; write at
  `[n-1]` succeeds; `free` + re-`malloc` reuses the block (assert same arena
  offset); `realloc` grow/shrink preserves the non-capability payload prefix;
  interior-pointer over-read past a sub-object still not caught (document the
  residual subobject gap).
- **Drop-in swap:** rebuild RV8 (7/7) and dtoa against `cap_heap` in place of the
  bump allocators — must stay green, proving the free-list allocator is a correct
  superset.
- **Measurement (feeds task #75):** allocator overhead (cycles/alloc, arena
  high-water, dynamic `SHRINK` count) with narrowing on vs off.
- **Blocked cases (write now, run after QEMU fix):** `realloc` of a block holding
  a tagged pointer round-trips the pointer; use-after-free of a narrowed cap
  faults (after revocation).

## 7. Decisions

*Decided 2026-07-01 (user review):*
1. **Narrowing primitive:** `SHRINK`, **not** `SPLIT` — see §3 (SPLIT rejected for
   the allocator; root-elimination deferred to a separate startup-partition
   proposal).
2. **Implementation base:** small purpose-built allocator (extend `rv8_malloc` →
   `cap_heap`), **not** a jemalloc/tcmalloc/glibc port — see §2. dlmalloc /
   picolibc nano-malloc kept as a documented credibility-upgrade fallback.
3. **Scope:** a clean, bounded, *small* support allocator (benchmarks + SQLite),
   not a general libc `malloc`. Paper claim is "bounded per-allocation
   capabilities + measured," not "we wrote dlmalloc."

*Still open (recommendations):*
4. **Free-list policy.** ~~First-fit + boundary-tag coalescing vs segregated
   bins.~~ **Superseded — see §7a: the vendored allocator (finally umm_malloc,
   index-based free list + coalescing) owns the policy.**
5. **Poisoning on free** now (spatial reuse only) vs gate behind revocation for
   real UAF protection. **#70 has landed**, so revoke-on-free is now viable —
   still recommend land **spatial only** first (phase 1), add revoke-on-free as a
   phase-2 follow-on (its own small proposal).

## 7a. Decision history (2026-07-04 → 2026-07-05): dlmalloc tried and rejected; **umm_malloc adopted (DONE, RV8 7/7)**

Two revisions landed here. Both are kept for the record.

**Rev 1 (2026-07-04): "port dlmalloc behind a boundary shim" — TRIED, FALSIFIED.**
The idea was: narrow only at the boundary (`SHRINK` on the malloc return; re-widen
via `cincoffset` off the wide arena cap on free/realloc entry) so *stock* dlmalloc
internals run unchanged on wide caps. It compiled and passed the malloc-only
benchmarks, but **hung on the first `free`**. Root cause (compile-time confirmed):
dlmalloc stores its free-list links (`fd`/`bk`/…) **in-band as capability
pointers** and navigates chunks with hand-computed `SIZE_T_SIZE` offset arithmetic
that assumes `sizeof(ptr)==sizeof(size_t)`. On PureCap a capability is 16 B and
16-aligned while `size_t` is 8, so `bk` lands at offset 32 (not 24), the chunk
struct is 48 B (not 32), and dlmalloc's small-bin overlay mis-aliases → corruption
→ `ABORT`. The incompatibility is **below** the shim; no boundary trick can fix it.
Full trail: `history/04-07-2026_11-30-00_dlmalloc-purecap-metadata-incompat.md`.

**Rev 2 (2026-07-05): adopt umm_malloc — DONE.** After surveying the landscape
(big server allocators = OS/thread/`mmap`/C++ coupling, unusable freestanding;
dlmalloc/glibc/nano-malloc = the same in-band `size_t`-offset trap), the selection
criterion became: **metadata must not store hand-offset capabilities in-band.**
umm_malloc (Ralph Hempel, MIT) keeps **all** free-list metadata as `uint16_t` block
*indices* — it never stores a capability in heap metadata, so it is **PureCap-safe
by construction** (a clean paper property: "no capability is ever stored in freed
heap metadata"). Vendored under `benchmarks/rv8/adapted/umm/`; the boundary shim
(`cap_heap.c`, narrow-on-return / re-widen-on-entry) from Rev 1 carried over
**unchanged** — only the allocator underneath swapped.

- **The one port change:** umm returns 4-byte-aligned data; PureCap needs 16
  (user structs store capabilities). Fixed by padding umm's block header to 16 B
  (32 B block, 16-aligned heap array) so data is capability-aligned. Two-line
  vendor edit, documented in `umm/umm_malloc.c` VENDOR-NOTES + `umm_malloc_cfg.h`.
- **Result:** RV8 **7/7 pass on umm**, including every free/realloc-heavy
  benchmark that hung under dlmalloc (dhrystone 1 free, aes 4 frees, miniz 8
  reallocs + 14 frees). Real reuse + coalescing under C1 narrowing, validated.
- dtoa/BEEBS unaffected (dtoa uses its own `malloc_beebs`, not this allocator).

**Phase-1 authority probes — DONE (2026-07-05):** `heap_free_reuse` and
`heap_coalesce` (in `tests/capstone-authority/domains/`) drive the *real* umm
allocator and prove reclaimed/coalesced memory is re-narrowed and an OOB read
still traps; full authority suite green (28/28). `heap_double_free` was
**deliberately deferred to phase 2**: a double-free is a *temporal* fault with no
clean spatial oracle today (umm has no double-free detector, so re-freeing a
dangling pointer silently corrupts rather than faulting). Under phase-2
revoke-on-free the freed pointer becomes untagged, so a double-free would
**tag-fault** — a clean oracle. That probe therefore belongs with the phase-2
work, where it also demonstrates the *value* of revocation.

**Phase 2 (revoke-on-free temporal safety, ties to revocation #70):** separate
proposal — umm's controlled index-based reuse makes a quarantine-before-relist
step tractable, and gives `heap_double_free` / use-after-free a clean tag-fault
oracle.

## 8. Pointers
- Existing allocators: `benchmarks/rv8/adapted/rv8_malloc.c`,
  `benchmarks/beebs/build-beebs-dtoa-capstone.sh` (dtoa `malloc_beebs` sed).
- Primitives: `llvm/include/llvm/IR/IntrinsicsCapstone.td`
  (`int_capstone_cap_shrink` wired; no `split`/`shrinkto`),
  `design/capability-bounds-model.md` (SHRINK/SHRINKTO/SPLIT, representability).
- Dependencies: `design/untagged-cap-loadstore-preservation-proposal.md`
  (realloc copy), `design/revocation-enforcement-proposal.md` (free poisoning).
- Audit driving this: `history/29-06-2026_15-08-22_granularity-provenance-audit.md`
  (heap row + root-elimination reframing).
