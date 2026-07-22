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

### Stage 1 — access-side lowering (this stage)
- Add the gated flag.
- Assign each eligible domain global a stable index via a deterministic pre-pass
  over `M.globals()` (defined, non-external, address-reachable data globals),
  recorded in a target map keyed by GlobalValue.
- In `expandCapGlobalBase` (or the DAG global-address lowering), under the flag,
  emit `ldc rd, gp, index*16` instead of `scc gp, VA; delin`. The pcrel VA
  materialization (`PseudoLLA`) is no longer needed for the base.
- Lit: `llvm/test/CodeGen/Capstone/cap-gp-captable.ll` — a global load/store emits
  `ldc … gp, <i*16>` under the flag and is byte-identical (still `scc`/`cincoffset`)
  with it off. Assert no `scc … gp` under the flag.

### Stage 2 — global descriptor table (compiler-emitted)
- Emit a read-only descriptor section (e.g. `.capstone_gp_table`) enumerating, in
  index order, `{size, align, has_init, init_template_symbol}` per global, plus the
  count N. The linker keeps initializers as a read-only template in the image.
- Keep empty (integer-only / no-global) domains a clean no-op (N=0).

### Stage 3 — entry-glue table + init builder (generic, runtime)
- A generic entry-glue init routine (generalized from `start-gpfree-captable.S`)
  reads the descriptor, and for each global: carve storage from `sp` (data
  authority) with `split`, copy the init template (or zero for `.bss`), `delin`,
  `stc` the cap into `gp[i]`; the cap-table itself is carved off the top of `sp`
  first (as in the probe). Remainder of `sp` = stack.
- Reentry re-derives gp from `sp` (like capstone-c) — no gp memory round-trip.

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
