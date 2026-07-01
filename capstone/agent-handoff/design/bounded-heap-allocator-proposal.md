# Proposal: a real bounded heap allocator (`malloc`/`free`) with C1 narrowing (and a `SPLIT` path)

*Status: DESIGN PROPOSAL for review before implementation (standing rule for new
research directions). Grew out of the 2026-06-29 granularity/provenance audit,
which found our "heap narrowing" is **not** a real allocator policy — only two
benchmark-local bump allocators narrow. Paths against `capstone-bootstrap` as of
2026-07-01. Tracked as task #78.*

## 0. Summary

Today, per-allocation capability bounds exist only as **two benchmark-local bump
allocators** (`rv8_malloc.c`, dtoa's `malloc_beebs`) that call `cap_shrink` on the
returned pointer; both `free` as a no-op and never reuse memory. The audit is
explicit that this must **not** be called "heap narrowing, default on." This
proposal is for a **single reusable bounded allocator** — real `malloc`/`free`
with block reuse — that returns each allocation as a capability narrowed to
exactly its requested size, so any over-read/over-write past an allocation
**faults**. It also sketches a **`SPLIT`-based** variant that is the concrete
"root-elimination" story the audit flagged as Capstone's novel angle: application
code holds only narrow per-object capabilities carved from an arena, never the
broad arena root. The narrowing half is implementable now; a fully general
`realloc`/coalescing that must **copy blocks containing capabilities** is
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

## 3. The `SPLIT` variant — root-elimination (the novel angle)

The audit's strategic reframing: object bounds re-derive CHERI; Capstone's
novelty is linearity + `SPLIT` + **root elimination**. A `SPLIT`-based allocator
makes that concrete:

- A **trusted arena manager** holds the one broad arena capability. `malloc`
  carves a sub-allocation by **`SPLIT`ting** a chunk off the arena root and
  handing the caller only that narrow (and linear) capability; the caller
  **never sees the arena root**. `free` re-merges. Contrast the `SHRINK` design,
  where the caller's narrow cap is still *derived from* a root that also remains
  ambiently reachable in the allocator's globals.
- This is the difference between "attenuation" (SHRINK: narrow view of a still-present root) and "root elimination" (SPLIT: the broad authority is consumed/partitioned, not just viewed). It's the paper-relevant Capstone-specific claim.
- **Blocker:** `SPLIT` (and `SHRINKTO`) exist in the ISA but are **not wired into
  LLVM** — only `SHRINK` is (`IntrinsicsCapstone.td`; `capability-bounds-model.md`
  §TL;DR). So the SPLIT variant needs a backend intrinsic + isel first. Propose
  doing the `SHRINK` allocator now and the `SPLIT` allocator as a follow-on that
  also delivers the `int_capstone_cap_split` intrinsic (reusable beyond malloc).

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

## 7. Open decisions (for the user / reviewer)

1. **`SHRINK`-only now, `SPLIT` follow-on?** Recommend yes: ship the measurable
   spatial allocator without waiting on a new backend intrinsic, then add the
   `int_capstone_cap_split` intrinsic + the root-elimination allocator as the
   novel-contribution follow-on.
2. **Allocator scope.** Benchmark/SQLite support allocator (a clean, bounded,
   *small* allocator) vs an attempt at a general libc `malloc`. Recommend the
   former — the paper claim is "bounded per-allocation capabilities + measured,"
   not "we wrote dlmalloc."
3. **Free-list policy.** First-fit + boundary-tag coalescing (simple, adequate)
   vs segregated bins. Recommend first-fit first; revisit only if measurement
   shows it matters.
4. **Poisoning on free** now (spatial reuse only) vs gate behind revocation for
   real UAF protection. Recommend land spatial now, add UAF when task #70 lands.

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
