# Plan — cap-table gp codegen for gp-free domains (silicon-correct global access)

## Why

The gp-free domain crash on captype-fixed CVA6 is root-caused and the fix is
**validated on silicon** (see
`history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md`, UPDATE 23-07 "FIX
VALIDATED"): a gp derived by `__split`-ing the **code** image is unusable as a data
base on the RTL; the working model (capstone-c, proven by
`tests/runtime-qemu/gp-free-domain/start-gpfree-captable.S` on the board, retval
554745961) derives gp from the **data** region (`sp`/cscratch) and reaches globals
through a **cap-table**: gp = base of an array of per-global capabilities, each a
data cap over that global's storage in data memory.

Our current `-capstone-gp-free` codegen instead emits `scc gp, &g` into the code
image (the thing the RTL rejects). This plan teaches codegen to emit the
cap-table pattern so an arbitrary real program's globals work on silicon.

## ABI (matches capstone-c codegen.rs:264-298; validated by the probe)

- **gp** = base of an N-entry cap-table in the domain data region; entry stride =
  capability width (16 B). Global with index `i` has its capability at `gp[i]`
  (byte offset `i*16`).
- **Access** to global `i`'s base pointer: `ldc rd, gp, i*16` → `rd` is a data cap
  whose cursor is the global's base. Interior offset (array element / struct
  field) is added with `cincoffset rd, rd, <off>` as today.
- **Storage + init**: each global lives in **data memory** carved from `sp` at
  domain entry. Initialized globals get their bytes copied from a read-only
  template kept in the code image (LMA); `.bss` globals are zeroed. This is a
  per-global, cap-table variant of the classic embedded `.data` copy.
- **Index order** is the single source of truth shared by (1) the access codegen
  and (2) the table/init builder. Must be deterministic per module.

## Staging (each stage: regression gate with flag OFF must stay byte-identical)

New gate flag **`-capstone-gp-captable`** (`cl::init(false)`), separate from
`-capstone-gp-free`; when off, codegen is unchanged (corpus green by construction).
gp-captable implies the gp-free call/ret lowering (plain jal/jalr) but replaces the
global-addressing half.

### Stage 1 — access-side lowering — DONE (commits 433a02f3, dfff7590)
- Gated flag `-capstone-gp-captable` + `getGpCaptableIndex()` (deterministic
  `M.globals()` enumeration: defined, sized, non-thread-local vars).
- `CapstoneISelLowering.cpp::lowerGlobalAddress`: under the flag, an indexable
  global lowers to `load i128 (gp + index*16)` — an invariant, entry-chained load
  that selection folds into `ldc rd, index*16(gp)`; the loaded data cap is used
  directly as the base. Non-indexable refs fall through to the default LGA path.
  (Chose ISelLowering over expandCapGlobalBase because the access is a chained
  memory load, which needs the DAG load machinery, not a pseudo expansion.)
- Lit `cap-gp-captable.ll`: `ldc … (gp)` + `-NOT scc` under the flag; unchanged
  `cincoffset gp; …` off. Corpus 41/41 green (flag-off byte-identical).

### Stage 2 — global descriptor table (compiler-emitted) — DONE (commit 6afca3bb)
- `CapstoneAsmPrinter::emitGpCaptableTable` emits `.capstone_gp_table` under the
  flag: `u64 count`, then per global in `getGpCaptableIndex` order
  `{u64 size, u64 align, i64 init_off}`. `init_off` is 0 for a zero global, else a
  PC-relative `template - .` diff (position-independent, same rationale as
  `.capstone_cap_init`). Absent when the flag is off; N=0 globals => no section.
- Lit `cap-gp-captable.ll` extended (descriptor + `CAP-NOT`); corpus 41/41 green.

### Stage 3 — entry-glue table builder (generic, runtime)  [NEXT]
- **DECISION on initializer values:** build the cap-table with **zero-initialized**
  storage in the glue, and set initializer *values* via **runtime stores** (reuse
  the `CapstoneCapGlobalInit` PC-relative-init-table + `start.S`-runner mechanism,
  extended to materialize *all* initialized domain globals, not just capability
  leaves, as gp-captable stores). This **avoids the image-read problem** entirely:
  the glue never has to read an init template through PCC (an execute cap can't
  serve as a data base — the same RTL rule that motivated this whole change), and
  it reuses proven infra. So the glue only needs `{size}` from the descriptor;
  `init_off` stays in the descriptor as an unused-for-now alternative.
- Glue (generalize `start-gpfree-captable.S`): `sp = cscratch`; carve the N-entry
  cap-table off the top (`split gp,sp,end - N*16`); for each i, carve `size[i]`
  bytes from `sp`, zero them, `delin`, `stc` into `gp[i]`; remainder of `sp` =
  stack. Reentry re-derives gp from `sp` (no gp memory round-trip).
- Then the runtime-store init routine runs (gp-captable stores) to set values.

### Stage 4 — real-app proof
- Pick a small integer benchmark with real globals + a call graph, build
  `-capstone-gp-captable`, run in a domain on QEMU (gp-fabrication OFF), then on the
  board (existing firmware — monitor unchanged). Expect correct result, clean exit.
- Full regression corpus stays green with the flag OFF.

## Open question routed to the board owner (`/tmp/capstone/boardowner-msg.md`)
Whether a **single** data cap over a data-region globals block (with direct
`scc gp` addressing) suffices instead of full per-global cap-table indirection. If
yes, a simpler Stage-1/3 variant (one data cap, `scc gp,&g` where &g is in the data
region) is possible; the cap-table path here is the guaranteed-correct default and
proceeds regardless.

## Constraints
Branch `capstone-gp-free`. Gated + corpus-green-when-off is the structural
guarantee. No real-person names; monitor stays a LOCAL experiment (this plan needs
NO monitor/firmware change — the probe ran on existing firmware). Bug-fix notes →
history/; this active WIP plan lives here in plans/.
