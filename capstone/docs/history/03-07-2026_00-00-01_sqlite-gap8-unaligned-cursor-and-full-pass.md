# SQLite gap 8 — unaligned capability access (BtCursor offset) + FULL PASS

*Status: FIXED 2026-07-03. This was the **last** SQLite bring-up gap: with it
resolved, the SQLite in-memory domain runs **end to end** (create/insert/select)
and emits all success markers.*

## Symptom

Past gap 7, the domain aborted QEMU with
`[CAPSTONE] Unaligned cap access (addr = 0x102237588)` then
`riscv_cpu_do_interrupt: Assertion 'env->priv < PRV_C'` (`cpu_helper.c`). A 16-byte
capability access (`ldc`/`stc`) requires a 16-aligned address; the QEMU assert is a
**secondary** effect (an exception raised in domain PRV_C mode is not deliverable
by the current interrupt path — a separate QEMU robustness gap, noted below).

## Localization (QEMU diagnostic, since reverted; submodule clean)

Printed the guest pc at the unaligned branch of `_helper_access_with_cap`:
```
GAP8-UNALIGNED stc addr=0x102237588 addr%16=8 pc=0x102060bc4 rs1=x12 base_cursor=0x1022374b8 imm=208
```
- an **`stc`** at `pc = sqlite3VdbeExec` region; symbolized to **`btreeCursor+0xfc`**.
- base `x12` = `0x1022374b8` (heap, `sqlite_heap` arena), storing a capability at
  offset `208` (a 16-aligned offset) → `0x102237588`, which is `8 mod 16`.
- The base object is at arena offset `0xe04d8` = memsys5 block `0xe04c0` (16-aligned)
  **+ 24**. So the cap-bearing `BtCursor` sits at **offset 24** inside the block.

## Root cause

`allocateCursor` packs a `VdbeCursor` and, for a btree cursor, an embedded
`BtCursor` into one allocation, placing the `BtCursor` at
`&pMem->z[SZ_VDBECURSOR(nField)]`. `SZ_VDBECURSOR(N) = ROUND8(offsetof(VdbeCursor,
aType)) + (N+1)*sizeof(u64)` is only **8-aligned** (SQLite assumes 8-byte max
alignment via `ROUND8`). The `BtCursor` holds capability fields (`pBt`, `pBtree`,
…); at an 8-aligned base its 16-aligned-offset cap fields land `8 mod 16`, so
`stc`/`ldc` fault. A global `ROUND8`→16 does **not** fix it: the `(N+1)*sizeof(u64)`
term re-introduces the 8-offset for odd `N+1`.

## Fix

`build-sqlite-capstone.sh` `sed`-patches `allocateCursor` to 16-align the embedded
`BtCursor` offset (and the matching allocation size):
```c
nByte = SZ_VDBECURSOR(nField);                         -> nByte = (SZ_VDBECURSOR(nField)+15)&~15;
&pMem->z[SZ_VDBECURSOR(nField)]  (BtCursor placement)  -> &pMem->z[(SZ_VDBECURSOR(nField)+15)&~15]
```
Both use the same rounded expression, so the allocation is sized correctly and the
`BtCursor` lands 16-aligned. SQLite-source only; no shared code / QEMU / compiler
change, so BEEBS/RV8/CoreMark are unaffected.

## Result — SQLite runs end to end

```
row name=alpha value=11
row name=beta value=22
row name=gamma value=33
__CAPSTONE_SQLITE_MEMORY_PASSED__
```
Confirmed with pristine QEMU (submodule clean). SQLite 3.53.3 now compiles, links,
and **executes** as a pure-capability domain: `CREATE TABLE`, `INSERT`, and
`SELECT` return correct rows. Gaps 1–8 all resolved.

## Follow-ups (separate, not blockers)

- **QEMU exception delivery in PRV_C:** an in-domain capability fault (unaligned
  cap access, and by extension bounds/tag faults) currently trips
  `riscv_cpu_do_interrupt: Assertion 'env->priv < PRV_C'` — QEMU aborts instead of
  delivering a clean, catchable fault. Worth fixing so cap faults are deliverable
  (would also let an authority probe exercise "unaligned cap store faults" without
  aborting the harness).
- **The alignment gaps are a class**: gap 6 (`saveBuf`) and gap 8 (`BtCursor`) are
  both "SQLite hand-packs a pointer-bearing region at an 8-aligned offset." Other
  instances may surface in wider SQLite workloads; each is a localized 16-alignment
  patch. A general audit (or a `ROUND8`→16 experiment gated behind the
  `(N+1)*8` caveat) could pre-empt them.
