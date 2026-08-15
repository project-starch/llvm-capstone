# S-06 workarounds — the revert list

**Purpose.** When S-06 is fixed in silicon, these are the software workarounds that exist only
because it was not. This file says exactly what to remove, what to keep, and what test must prove
the removal is safe. Written 2026-08-14, while the workarounds are live.

**The thing this file exists to prevent:** the S-06 workarounds sit in the same files, and in one
case on adjacent lines, as workarounds for *different* silicon defects (S-04, linearity, R-14).
Reverting "the memcpy silicon stuff" would take those with it and reintroduce bugs that have
nothing to do with S-06. §2 is therefore as important as §1.

---

## 0. What S-06 is, in one paragraph

A bare `ldc`/`stc` pair copying PLAIN (untagged) 16-byte data loses the granule's high 8 bytes.
`wt_dcache_mem.sv:308` returns a literal `'0` for the user/metadata half when the line's shadow tag
is clear, so the high half is either never written (`st_wr_cap` low, bank 1 untouched) or written
with metadata manufactured by `compress_bounds`' cursorless branch. Either way it is gone. The
minimal repro `s06agg` returns **5** on silicon where a correct machine returns **15**.

---

## 1. REVERT THESE when silicon is fixed

### 1.1 `BEEBS_LDC_HIGH_HALF_FIXUP` — the library memcpy's compare-and-repair

* **What**: `BEEBS_CHUNK_COPY` in `capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c`
  writes both halves with plain stores first, then lays a guarded `stc` on top, so a plain granule
  keeps its high half and a real capability keeps its tag.
* **Enabled by**: `SQLITE_LDC_HIGH_HALF_FIXUP=1` →
  `build-sqlite-silicon.sh:1493` (`_ldc_fixup`, **default 0**) → `-DBEEBS_LDC_HIGH_HALF_FIXUP`
  in `SUPPORT_DEFS` (`:1496`).
* **Revert**: delete the `#if defined(BEEBS_LDC_HIGH_HALF_FIXUP)` arm of `BEEBS_CHUNK_COPY`,
  keeping the `#else` one-liner; remove `_ldc_fixup` and the `-D` from `SUPPORT_DEFS`.
* **Cost while it stays**: changes the primitive every SQLite/BEEBS timing number is measured on.

### 1.2 `-capstone-guard-cap-granule-copies` — the compiler's aggregate-copy guard

* **What**: `llvm/lib/Target/Capstone/CapstoneCapGranuleCopy.cpp`, an IR pass expanding
  16-byte-aligned aggregate copies into per-granule IR whose capability store is guarded by an LCC
  type query. `cl::init(false)` — off unless asked for.
* **Depends on**: the intrinsic `int_capstone_cap_get_type`
  (`llvm/include/llvm/IR/IntrinsicsCapstone.td:80`) and its pattern
  (`llvm/lib/Target/Capstone/CapstoneInstrInfo.td:2437`, lowering to `LCC $src, 1`).
* **Test**: `llvm/test/CodeGen/Capstone/cap-granule-copy-guard.ll`. **Its BARE arm deliberately
  pins the UNFIXED sequence**, so if the pass is deleted this test must be deleted with it — it
  will not merely start passing.
* **Revert**: delete the pass, its registration, the intrinsic, the pattern, and the lit test.
* **Cost while it stays**: **+33660 bytes of `.text`** in the SQLite domain and a branch per
  granule. Measured: `.text` 0x13eb3c → 0x1472b8.
* **Note**: it emits `lcc` field 1, which is only TOTAL on enabler silicon. That is why it is not
  default-on for the target.

### 1.3 Anything that enables 1.1/1.2 in a build

* `capstone/benchmarks/sqlite/build-sqlite-silicon.sh` — the `_ldc_fixup` knob and any
  `EXTRA_MLLVM` default that turns the guard on.
* Any baked-domain recipe passing `SQLITE_LDC_HIGH_HALF_FIXUP=1` or
  `-mllvm -capstone-guard-cap-granule-copies` (see `bake-sqlite-doms.sh` invocations in the history
  notes).

---

## 2. DO **NOT** revert these — different defects, same neighbourhood

| knob / change | file | what it actually works around |
|---|---|---|
| `BEEBS_MEMCPY_OPTNONE` | `beebs_freestanding_string.c` | **S-04**: a 7-byte aligned memcpy whose stores do not stick at -O1. Nothing to do with high halves. |
| `BEEBS_STRING_WRITERS_OPTNONE` | same file | the S-04 family for memmove/memset/strcpy. Measured independently (stage 167). |
| `BEEBS_STRING_LINEAR_SAFE` | same file | **linearity**, not S-06: keeps the string primitives from copying linear capabilities. |
| R-14 `aBuiltinFunc` static initialiser | `build-sqlite-silicon.sh` | a different silicon issue in the copy loop for that table. |
| QEMU `scalar_hi` | `capstone-qemu/target/riscv/cap.h:91,129` | **KEEP.** This makes QEMU's untagged `ldc`/`stc` bit-exact, i.e. CORRECT. When silicon is fixed the two agree; removing it would make QEMU wrong. |
| RTL LCC-total type query | `capstone-ariane` commit `55b7f88bc` | an ENABLER the guard depends on, and a useful total query in its own right. Removing the guard does not require removing this. |

---

## 3. Acceptance gates for the revert

Run these **before** deleting anything, on the fixed bitstream, and require the "fixed" column.
Each rung already carries a positive control; a clean result from a rung whose control has not been
shown to fire is not evidence.

> ## GATE PASSED ON SILICON — 2026-08-15, bitstream `caplifive_s06fixs08fix.bit`
>
> One control-validated boot (`k800` = 4):
>
> | rung | broken silicon | **measured now** |
> |---|---|---|
> | `s06agg` | 5 | **15** |
> | `s06aggcap` | 7 | **15** |
> | `s06aggwide` | 237 | **255** |
>
> These are the UNFIXED rungs — no software workaround in the build — which is exactly the
> condition this document set. **The decisive criterion ("`s06agg` returns 15 with no software
> workaround") is met.** §1 may now be reverted; §2 must still NOT be.
>
> Note the bitstream also carries the S-08 fix (dom-switch stores honouring the switcher's per-row
> `metadata_en`). The first `caplifive_s06fullfix.bit` could not run domains at all, so this is the
> first bitstream on which the gate was ever readable.

| test | today (broken silicon) | required after the fix |
|---|---|---|
| `s06agg` | 5 | **15** |
| `s06aggcap` | 7 | **15** |
| `s06aggwide` | 237 | **255** |
| SQLite `G6` (after the row loop) | rc=3 with three rows, **guard ON only** | same with **both workarounds OFF** |
| SQLite `G7` (after finalize) | rc=0, **guard ON only** | same with **both workarounds OFF** |
| lit `cap-granule-copy-guard.ll` | passes both arms | delete with the pass |

**The decisive one is `s06agg` returning 15 with no software workaround in the build.** If it still
returns 5, silicon is not fixed and nothing here should be removed regardless of what else passes.

---

## 4. What the revert does NOT fix

There is a **second, independent** defect on the SQLite path, still open as of 2026-08-14: mcause
**25** (UNEXPECTED_OPERAND, a lost TAG) at `memcpy+0x2a8`, in extended phase 2→3. It is not S-06 —
different cause code, different mechanism — and five silicon measurements show the ordinary
capability round trip is sound (tag, bounds, stores through the pointer, a scalar load of the
granule, and a full cache eviction all survive). Do not expect the S-06 fix to clear it, and do not
read a still-failing SQLite run after the revert as evidence that the S-06 fix did not work. See
`history/14-08-2026_02-30-00_sqlite-wedge-is-out-of-bounds-on-Mem.md`.
