# Proposal: preserve capability tags through relatively-misaligned `memcpy` (SQLite gap 6)

*Status: IMPLEMENTED 2026-07-03 (Option 1). Step-0 diagnostic resolved Case A vs B
(→ Case A); the fix — 16-align `saveBuf` in `sqlite3NestedParse` — is landed and
SQLite runs past gap 6. See
`history/02-07-2026_00-00-00_sqlite-gap5-fix-and-gap6-investigation.md` ("Gap 6
FIXED") for the fix + the new gap-7 blocker. This doc is the fix design.*

**Step 0 result (2026-07-03): Case A, confirmed.** The primary loss is a genuine
tagged 16-aligned source (`src%16=0`, `src_tagged=1`) byte-copied to a
relatively-misaligned destination (`dst%16=12`) — the `char saveBuf[PARSE_TAIL_SZ]`
in **`sqlite3NestedParse`** (`memcpy(saveBuf, PARSE_TAIL(pParse), 256)`), a bare
`char[]` the compiler placed at a 12-mod-16 slot. Because the relative
misalignment is **constant**, **Option 2 cannot help this case** (a tag cannot
live at destination offset 12) — the correct fix is **Option 1** (16-align
`saveBuf`). Option 2 stays a general-hardening nice-to-have, not required for
gap 6.

## Problem (confirmed)

SQLite faults in `sqlite3DeleteTable` on an untagged `Table*`. A storage-slot-keyed
QEMU trace (translate-time pc; correct load base `0x101ff0000`; instrumentation
since reverted) pinned the tag-stripping store to the domain's freestanding
**`memcpy` byte-copy loop** (`memcpy+0x1fc`, `sb`): a 16-byte-aligned tagged
`Table*` is copied **byte-by-byte** (16 `size=1` stores over one granule), which
clears its out-of-band tag. The untagged pointer then propagates (memcpy caller in
the `sqlite3NestedParse` region) into `sqlite3DeleteTable` and faults on
`--pTable->nTabRef`.

`memcpy` (`beebs_freestanding_string.c`) already has a tag-preserving fast path —
it copies the middle one `void*` (=`ldc`/`stc`, tag-preserving) at a time — **but
only when `dst` and `src` share alignment mod 16** (`da == sa`). When they are
**relatively misaligned**, it falls through to a full byte loop for the entire
copy (the code even comments this). Gap 6 is that fall-through running over a live
capability. This is the documented `tagged_cap_memcpy_misaligned` limitation, now
shown to break real software, not just a probe.

## Why this is subtle (the key open question)

`ldc`/`stc` require **16-byte alignment on both ends**. If `src` and `dst` are
relatively misaligned mod 16, then for any offset `o`, `src+o` and `dst+o` have
different alignments — so a capability landing 16-aligned at the destination has a
**non-16-aligned source**, where a tag cannot be represented at all. Two cases
must be told apart, because they need different fixes:

- **Case A — source cap is genuinely tagged (aligned) and destination is
  misaligned relative to it.** Then a relatively-misaligned `memcpy` *cannot*
  faithfully carry the tag (the destination granule can't hold it). The correct
  fix is upstream: make the copy **not** relatively misaligned.
- **Case B — the destination granule is 16-aligned and holds a live cap, and the
  copy overwrites it with untagged bytes from a misaligned/plain source.** Then
  the tag being lost is the **destination's pre-existing** tag; SQLite expected
  the post-`memcpy` field to still be a valid capability, i.e. it assumes the copy
  carries a capability that the source does not actually hold as one.

The gap-6 evidence (destination granule `0x1023ffa80` is 16-aligned and was
`stc`'d tagged *before* the byte copy) points at **Case B / an alignment mismatch
of a cap-bearing struct**: the two structs differ in mod-16 alignment, so what
should be a `void*`-granular `ldc`/`stc` move degrades to bytes.

**One cheap diagnostic disambiguates and must be step 0 of implementation:**
re-instrument `memcpy`'s entry (or the store helper) to capture, for the culprit
copy, `dst`, `src`, `n`, and whether the **source** granule of the capability is
tagged (`cap_mem_map_query`). That tells us Case A vs B and whether the source
alignment is fixable.

## Options

### Option 1 — 16-align capability-bearing structures (recommended first)

If the culprit copy is relatively misaligned because a cap-bearing SQLite struct
(or its allocation) is under-aligned, force **16-byte alignment** on the
allocation / struct so `da == sa == 0` and `memcpy`'s existing fast path applies.
- Pros: reuses the already-correct fast path; no `memcpy` change; localized.
- Cons: requires identifying the specific struct(s); alignment must hold at the
  allocator too (MEMSYS5 `sqlite3_config(..., 64)` alignment already 16-friendly).
- This is the same class as the dtoa arena / `Bigint.next` 16-alignment fix.

### Option 2 — tag-aware `memcpy` for the aligned-destination sub-case

Extend `memcpy`/`memmove`: even when `da != sa`, for each destination granule that
is 16-aligned **and** whose source granule is *also* 16-aligned (possible only in
mixed-alignment runs where a sub-range re-aligns), move it with a `void*`
`ldc`/`stc`. Where the source is misaligned, fall back to bytes (unavoidable).
- Pros: hardens the library generally.
- Cons: only helps when a source cap is actually aligned; does not fix Case A;
  more complex loop.

### Option 3 — explicit-clear contract (defensive, not a real fix)

Leave `memcpy` byte-based for misaligned copies but make the *loss* loud/eager
rather than silent (already the de-facto behaviour: the untagged pointer faults on
first deref). Not a fix; documents the limit. Rejected as the primary fix.

## Recommendation

1. **Diagnostic (step 0):** capture the culprit `memcpy(dst, src, n)` and the
   source granule's tag state → decide Case A vs B.
2. If it is an under-aligned cap-bearing struct (expected): **Option 1** (16-align
   the struct/allocation) — smallest, reuses the correct path.
3. Land **Option 2** as a general hardening of the shared `memcpy`/`memmove` if the
   diagnostic shows aligned-source sub-ranges are recoverable.
4. Add an **authority probe** `tagged_cap_memcpy_relmisaligned`: a tagged cap in a
   struct copied by `memcpy` with `src`/`dst` at different mod-16 offsets; assert
   the tag survives after the fix (and, as the pre-fix oracle, that it is stripped
   today). This complements the existing `tagged_cap_memcpy_{aligned,misaligned}`
   probes.

## Scope / non-goals

- This is a **runtime-library + struct-alignment** fix, not a backend ISel change.
  It connects to gap 2 (clang `memcpy`-from-private-template of cap aggregates) and
  the sub-capability aggregate-copy fix, but does not require touching them.
- It does not change the 16-byte tag-granularity model; it works within it. The
  paper framing (a capability machine makes `memcpy` security-relevant — byte
  identity is not tag identity) is recorded in
  `design/research-decisions-log.md`.

## Validation gate (after implementation)

- New authority probe `tagged_cap_memcpy_relmisaligned` passes; existing
  `tagged_cap_memcpy_{aligned,misaligned}` still pass.
- `run-sqlite-memory.sh` runs **past** gap 6 (reaches the next gap or the success
  markers `row name=alpha/beta/gamma`, `__CAPSTONE_SQLITE_MEMORY_PASSED__`).
- Full backend/runtime gate unaffected: Capstone lit, CoreMark, BEEBS 82/82,
  RV8 7/7 (the shared `memcpy` is used by BEEBS/RV8, so any change must keep them
  green).
