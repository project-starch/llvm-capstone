# Proposal: make untagged `ldc`/`stc` bit-preserving over the full 128-bit word (QEMU)

*Status: **IMPLEMENTED (2026-07-01)** on `capstone-qemu` branch
`fix/untagged-ldc-stc-128bit-preservation`. The runtime author confirmed this is
a **bug** in the current emulator and that the intended model is **goal 1**
(untagged `ldc`/`stc` bit-exact over the full 128-bit word), and that the spec
document should state it — so **Option A was chosen** and a matching spec note is
tracked on a `capstone-spec` branch. Validated: authority-suite round-trip probe
`untagged_cap_roundtrip` (both 64-bit halves survive; retval `0x22990003`) plus
the full authority suite green (20/20 unchanged + the new probe). Grew out of the
SQLite bring-up (`sqlite-marshalling-feasibility.md` / gaps 3–4); the same
limitation also blocked a correct capability-aware `memcpy`. See §8 for the
implementation record. Paths/line numbers are against `capstone-bootstrap` as of
2026-07-01.*

## 0. Summary

On this target an **untagged** capability register holds only a **64-bit
scalar**, and the untagged capability load/store paths move only the low 64 bits:
untagged `stc` writes `lo = scalar`, **`hi = 0`**, and an untagged capability load
recovers only `lo`. A 16-byte memory word that holds *plain data* (not a tagged
capability) therefore does **not** survive an `ldc`/`stc` round-trip — its high 8
bytes are zeroed. Consequently there is **no in-domain `memcpy` that preserves
both plain data and capability tags**: copying 16-byte chunks with `ldc`/`stc`
corrupts data, and copying them with scalar `ld`/`sd` drops tags. We propose
making untagged `ldc`/`stc` **bit-exact over the full 128-bit word** (as CHERI's
untagged `clc`/`csc` are), after which a trivial `ldc`/`stc`-middle `memcpy`
copies data and capabilities correctly with no runtime tag query. This is a
`capstone-qemu` (submodule) change only; the compiler/toolchain are unaffected.

## 1. Why this surfaced — SQLite gaps 3 and 4

- **Gap 3** (agent-diagnosed, `/tmp/capstone/gap3-diagnostics-results.md`): SQLite
  grows `Table.aCol` via `sqlite3DbRealloc` → memsys5 `memcpy(new, old, nOld)`.
  Our freestanding **byte-loop** `memcpy` copies the address bits of
  `Column.zCnName` but not its out-of-band tag; the reloaded pointer is untagged
  and `strlen` faults. Correct diagnosis: a Capstone-faithful `memcpy` must
  preserve tags for capability-sized chunks.
- **Gap 4** (root-caused this session): the "fix" — a tag-preserving `memcpy` that
  copies the 16-byte aligned middle via `*(void**)d = *(void*const*)s` (i.e.
  `ldc`/`stc`) — made `CREATE TABLE` return `SQLITE_CORRUPT (rc=11)`. A heap
  overlap probe showed the copied region differed from the source in **16 bytes,
  first_bad = 8, last_bad = 31**, i.e. byte 8 of each 16-byte chunk came back `0`.
  The `ldc`/`stc` middle silently **zeroed the high 8 bytes of every non-capability
  chunk** — schema text and row structs — producing the corruption.

So gap 3 (byte copy drops tags) and gap 4 (cap copy zeroes data) are **two faces
of one limitation**: no single in-domain copy primitive round-trips 128 bits of
memory content faithfully. The tag-preserving `memcpy` has been **reverted** to
the committed byte loop (loud gap-3 fault, no silent corruption; benchmarks stay
correct). Host fuzzing (104,593 cases) missed this because host `void*` copy is a
plain 8-byte move with no capability re-encoding.

## 2. Exact current path, and why 128 bits are not preserved

- **Untagged register is 64-bit.** `struct CapRegVal { union { capfat_t cap;
  capaddr_t scalar; } val; bool tag; }` with `capaddr_t = uint64_t`
  (`cap.h:10,79–88`). When `tag == false` the meaningful state is the single
  `scalar` word; the high 64 bits of the original memory word have nowhere to
  live.
- **Untagged `stc` zeroes `hi`.** The store path compresses via
  `helper_compress_cap` (`op_helper.c:1041`); its untagged branch
  (`op_helper.c:1045–1051`) sets `cap_compress_result_lo = reg_v->val.scalar` and
  **`cap_compress_result_hi = 0`**, and the two words are stored. Byte 8 = 0 in
  the probe matches exactly.
- **Untagged load recovers only `lo`.** `load_capregval`'s untagged branch
  (`capstone_helper.c:68–71`) does `v->val.scalar = address_space_ldq(as, addr)` —
  a single 8-byte read; the high word at `addr+8` is never brought into the
  register. (The related context-switch store, `store_capregval:84–87`, likewise
  writes only `addr` and leaves `addr+8` untouched.)
- **Scalar `sd` drops tags.** The alternative — copy chunks with scalar
  `ld`/`sd` — routes stores through `cap_mem_map_remove_range`
  (`helper_remove_cap_mem_map`, `op_helper.c:1067`), clearing any capability tag
  in the destination range. So scalar copy preserves data but destroys tags.
- **No runtime tag query to branch on.** A `memcpy` cannot "load a chunk, ask if
  it was a capability, then choose `ldc`/`stc` vs `ld`/`sd`": `helper_cslcc`
  (`op_helper.c:662`) asserts a **tagged** operand and its tag-query field is
  stubbed. There is no in-domain "is the value at this address tagged?" primitive.

Net: for a *tagged* 16-byte word, `ldc`/`stc` are already bit-exact (compress /
uncompress + `cm_map`). For an *untagged* word, the high 64 bits are lost on both
load and store. The fix is to make the untagged path preserve those bits too.

## 3. Intended architectural semantics

An untagged capability load/store should be a **faithful 128-bit memory move**:
`stc` of an untagged register must write the full 16 bytes it logically holds, and
`ldc` of a non-capability 16-byte word must recover all 16 bytes, such that
`ldc; stc` (or `stc; ldc`) is the identity on arbitrary memory contents. Required
properties:

- **Tagged round-trip unchanged.** A tagged capability still stores/loads via
  compress/uncompress + `cm_map`; tags and bounds are preserved exactly as today.
- **Untagged round-trip is bit-exact.** All 128 bits survive; no zeroing, no
  truncation, and the destination is **not** left tagged (an untagged store must
  clear the `cm_map` entry as it does now).
- **This is the CHERI contract.** CHERI's `clc`/`csc` move the full capability
  width whether or not the tag is set; the tag is separate metadata. Adopting the
  same rule makes a naive `ldc`/`stc`-middle `memcpy` correct for data and caps
  alike — matching how CHERI/PureCap libc `memcpy` is written.

## 4. The two-word-untagged fix vs a per-chunk tag-query `memcpy`

**Option A — widen the untagged path to carry the full 128 bits (recommended).**
Make the untagged register representation hold both words and make untagged
`ldc`/`stc` move both.

- *Pros:* one localized `capstone-qemu` change; makes the ISA self-consistent
  (untagged cap load/store becomes bit-preserving like the tagged path and like
  CHERI); a trivial `ldc`/`stc`-middle `memcpy` then works for data **and** caps
  with **no** tag query; benefits every domain, not just SQLite.
- *Cons:* touches the `capregval_t` union / untagged store-hi path and must be
  audited so scalar consumers still read `lo` correctly (`val.scalar` semantics),
  and so the untagged store still clears the `cm_map` tag. Needs the author's
  sign-off on the register representation.

**Option B — keep 64-bit untagged, add a runtime tag query, branch per chunk in
`memcpy`.** Introduce an in-domain "is this address tagged?" primitive; `memcpy`
loads a chunk, queries, and picks `ldc`/`stc` vs `ld`/`sd`.

- *Cons:* requires a *new* ISA/helper primitive (un-stub `lcc`'s tag query or add
  one), a branch per 16-byte chunk on the hot copy path, and it still can't
  reconstruct a *tagged* value from a `ld`/`sd` copy — so it only helps if every
  copied capability is individually re-materialized, which `memcpy` cannot do
  generically. Strictly more mechanism for a worse result.

**Recommendation: Option A.** It is the minimal correct change, it removes the
need for any tag query, and it makes untagged cap load/store bit-preserving, which
is arguably the *correct* emulator behavior independent of SQLite.

## 5. Minimal QEMU change sketch (for review, not yet implemented)

1. Give the untagged register path a second word (e.g. carry `scalar_hi`
   alongside `val.scalar`, or reuse the compressed `lo/hi` slots), so an untagged
   register can hold the full 16 bytes it was loaded from.
2. `load_capregval` untagged branch (`capstone_helper.c:68–71`): read **both**
   `addr` and `addr+8` into the untagged representation (mirror the tagged
   branch's two `address_space_ldq`s), still with `tag = false`.
3. `helper_compress_cap` untagged branch (`op_helper.c:1045–1051`): set
   `result_hi` to the preserved high word instead of `0`; `store_capregval`
   untagged branch (`capstone_helper.c:84–87`): write **both** words and keep the
   `cap_mem_map_remove` (destination stays untagged).
4. Re-point the freestanding `memcpy`/`memmove`
   (`benchmarks/beebs/adapted/beebs_freestanding_string.c`) to copy the 16-byte
   aligned middle with `ldc`/`stc` and byte head/tail — **only after** the QEMU
   change lands and the round-trip test in §6 is green. Until then it stays the
   byte loop.

No compiler, `libcapstone`, or kernel-module change is required.

## 6. Test matrix (before = current gap; after = the fix)

| # | Case | Before | After |
|---|------|--------|-------|
| 1 | **Untagged 128-bit round-trip probe** — store a 16-byte data pattern, `stc` an untagged reg over it, `ldc` back, compare all 16 bytes | high 8 bytes read back `0` | bit-exact |
| 2 | **`memcpy` of a struct mixing data + a tagged pointer** (the `runtime-bytecopy-capability.c` shape, `ldc`/`stc` middle) | data byte 8 zeroed / tag dropped depending on primitive | both data and tag preserved |
| 3 | **SQLite gap 3/4** — `CREATE TABLE` + `INSERT` + `SELECT` on `:memory:` with the `ldc`/`stc` `memcpy` | rc=11 `SQLITE_CORRUPT` (or gap-3 fault with byte loop) | `__CAPSTONE_SQLITE_MEMORY_PASSED__` |
| 4 | **No-regression gate** — RV8 7/7, CoreMark CRC, BEEBS 82/82 with the `ldc`/`stc` `memcpy` | (byte loop: green) | still green — proves the widened path didn't perturb tagged/scalar copies |
| 5 | **Untagged store clears tag** — `stc` untagged over a slot that previously held a capability, then `ldc`; result must be untagged | untagged (correct today) | still untagged (no false tag) |

Case 1 is a tiny new runtime probe (the decisive unit test for the QEMU change).
Case 2 reuses the committed `benchmarks/sqlite/probes/runtime-bytecopy-capability.c`.
Case 3 is the existing SQLite memory driver. Case 4 is the standing aggregate gate.
Case 5 guards against over-tagging.

## 7. Open questions for the runtime/QEMU author

1. **Register representation.** Is widening the untagged `capregval_t` path to
   carry the full 16 bytes acceptable, or is there a reason the untagged register
   is intentionally 64-bit (spilling, migration, some invariant we're missing)?
2. **Intended untagged `ldc`/`stc` semantics.** Should untagged capability
   load/store be bit-preserving over 128 bits (CHERI `clc`/`csc` semantics), i.e.
   is the current `hi = 0` / single-word load a **bug** rather than by design?
3. **Tag query.** Is there any intended in-domain "is this address/value tagged?"
   primitive (would `lcc`'s stubbed tag case be filled in)? If Option A lands we
   don't need one, but it affects whether Option B is even on the table.
4. **`store_capregval` context-switch path.** The context-switch store also writes
   only `addr` (not `addr+8`) for untagged values (`capstone_helper.c:84–87`) —
   is that a related latent gap for scalar state saved across domain switches, or
   is that path only ever used for tagged/PC caps?

## 8. Status and next step

**Implemented (2026-07-01), Option A, on `capstone-qemu` branch
`fix/untagged-ldc-stc-128bit-preservation`.** The runtime author confirmed the
`hi = 0` behaviour is a bug and goal 1 (full 128-bit round-trip) is the intended
model. Changes (six sites, one localized representation change):

1. `target/riscv/cap.h` — added `capaddr_t scalar_hi` to `struct CapRegVal` (the
   high 64 bits of an untagged register; meaningful only when `tag == false`);
   `capregval_set_scalar` zeroes it so a genuine 64-bit value still stores
   `hi = 0`. Safe re: migration — `env.gpr` VMState is already commented out
   (`machine.c`).
2. `op_helper.c helper_reg_set_cap_compressed` (the `ldc` register-set path) —
   for the untagged case (incl. a revoked cap demoted to untagged) store the raw
   loaded words `val.scalar = lo`, `scalar_hi = hi` (`cap_uncompress` is lossy
   for arbitrary bit patterns, so we keep the raw words).
3. `op_helper.c helper_compress_cap` (the `stc` path) — untagged writes
   `result_hi = reg_v->scalar_hi` instead of `0`.
4. `capstone_helper.c load_capregval`/`store_capregval` (context-switch / pc-swap
   path) — untagged branch now loads/stores **both** words.
5. Three direct scalar writers (`common-semi-target.h`, `gdbstub.c`,
   `cpu_helper.c` exception cause) zero `scalar_hi` to avoid a stale high word.

**Validated.** New authority probe `untagged_cap_roundtrip` copies a 16-byte
plain-data word (non-zero high half) via an untagged `ldc`/`stc` pair and checks
both halves survive → retval `0x22990003` (before the fix: `0x22990001`). Full
authority suite green (20/20 prior probes unchanged + this one). QEMU builds
clean (89/89).

**Next:** re-enable the tag-preserving `memcpy` (copy the 16-byte aligned middle
via `ldc`/`stc`, byte head/tail) now that the round-trip is bit-exact, then
re-drive SQLite `CREATE TABLE` to clear gaps 3–4 (tasks #73/#74). The
`ldc`/`stc` `memcpy` was correctly withheld until this landed; it is now safe.

## Pointers
- SQLite gaps: `sqlite-marshalling-feasibility.md`,
  `/tmp/capstone/gap3-diagnostics-results.md`,
  `../../benchmarks/sqlite/probes/runtime-bytecopy-capability.c`,
  `../../benchmarks/sqlite/sqlite_capstone_domain.c`.
- `memcpy`/`memmove`: `../../benchmarks/beebs/adapted/beebs_freestanding_string.c`.
- QEMU: `target/riscv/cap.h:10,79–88` (`capregval_t`, `capaddr_t`),
  `target/riscv/capstone_helper.c:57–88` (`load_capregval`/`store_capregval`),
  `target/riscv/op_helper.c:1041–1055` (`helper_compress_cap` untagged `hi=0`),
  `target/riscv/op_helper.c:1067` (`helper_remove_cap_mem_map`, scalar-store tag
  clear), `target/riscv/op_helper.c:662` (`helper_cslcc`, stubbed tag query).
- Related: `revocation-enforcement-proposal.md` (the other open `capstone-qemu`
  semantics item to raise with the author).
