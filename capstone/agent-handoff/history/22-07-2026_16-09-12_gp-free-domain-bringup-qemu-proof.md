# gp-free domain bring-up: a real globals-using app runs in a pure-cap domain, gp-free, on QEMU (silicon-shaped ABI)

**Date:** 2026-07-22
**Status:** DONE on QEMU (functional, silicon-faithful). Remaining: the same
monitor change on the FPGA (caplifive-system) copy + a board run for cycles.
**Branch:** `capstone-gp-free` (off `capstone-bootstrap`).

## Result

A real integer app (`.rodata`/`.data` globals + a non-inlined `helper()` call
graph) now runs **correctly** in a pure-capability domain **gp-free / cjalr-free**,
with the `gp = PCC(cursor 0)` fabrication **disabled** and an image-covering `gp`
delivered by the **monitor** via the cscratch stack region — no hardware/QEMU gp
magic. `retval = 0x2110C069` (exact); static `cjalr=0`, `cincoffset-gp=0`,
`scc-gp=3`. Default domains still pass with the rebuilt monitor (authority ✓).
Test: `tests/runtime-qemu/gp-free-domain/` (`build-and-run.sh` →
`__CAPSTONE_GPFREE_DOMAIN_PASSED__`).

## Why (the two blockers, both ours)

Real domains only ran because our QEMU fork *fabricates* `gp` (patch `7aca0540`);
they never ran on silicon. Two layers, confirmed by the board owner (2026-07-22):

1. **Global addressing.** Backend reaches globals via `cincoffset gp, <abs>`,
   needing `gp = PCC(cursor 0)`. On the RTL `gp` is **not** hardware-restored on
   `cscall`, and an image-bounded cap forced to cursor 0 is **unrepresentable**
   (capability compression). So that gp cannot exist on silicon.
2. **Calls/returns.** `cjalr` needs a code capability, unformable gp-free on a
   fresh entry.

## Fix (three layers, gated)

- **Compiler `-capstone-gp-free`** (LLVM, `cl::init(false)`, byte-identical off —
  whole corpus unaffected; lit 40/40):
  - `CapstoneAsmPrinter` — `PseudoRET`/`PseudoCALLIndirect`/`PseudoTAILIndirect`
    lower to plain `jalr` within PCC instead of `CJALR`.
  - `CapstoneISelDAGToDAG` `selectCall` — a direct call reuses the PC-relative
    `PseudoLLA` address as a plain `jalr` target (no `cincoffset gp`, no cjalr;
    avoids the unwired Capstone `PseudoCALL` CALL-relocation).
  - `CapstoneExpandPseudoInsts` `expandCapGlobalBase` — under the flag, global
    addressing uses **`SCC`** (set cursor to the absolute VA) instead of
    `CIncOffset` (add to cursor). SCC works with any representable
    (in-bounds-cursor) image-covering gp; CIncOffset needs cursor 0. Covers data
    loads and the cap-global-init path.
- **Monitor** `create_domain` (OpenSBI, caplifive-buildroot copy): mints an
  image-covering data cap with `C_GEN_CAP(gp, base, base+code_size)` (in-bounds
  cursor) and stores it in the **top-16 slot of the domain's stack (cscratch)
  region** through a separately generated writer cap (dom_data untouched). This is
  the board owner's confirmed channel (same as the reference `capstone-c`, which
  delivers via cscratch). No `ctvec` dependence. `.c.S` is compiled by capstone-c;
  its pattern rule watches the wrapper not the `#include`, so force-regen
  (`rm sbi_capstone_dom.c.S`) before `make build A=opensbi-rebuild`.
- **Entry glue** `start-gpfree-cscratch.S`: `ldc gp, END-16` + `delin` on entry
  (reentry reloads the saved non-linear gp), plain `call domain_main`.
- **QEMU** `op_helper.c`: the four gp fabrication sites are guarded by
  `capstone_gp_fabricate()` (env `CAPSTONE_GP_FABRICATE`, default on = legacy).
  `CAPSTONE_GP_STANDIN=1` makes `helper_cscall` deliver a representable gp as a
  monitor stand-in (for validating the codegen without a monitor rebuild).

## Board owner's guidance (2026-07-22), all honored

cursor-0 unrepresentable → `scc`/in-bounds cursor ✓; gp not hw-swapped → software
(monitor + glue) delivers it ✓; cscratch is the delivery channel (capstone-c uses
it) ✓; no single "intended" mechanism — our image-gp + `scc` is a valid
alternative to capstone-c's per-global cap table ✓; QEMU is functional-only, perf
from the board ✓. See memory `project_silicon_gp_delivery_boardowner_guidance`.

## Errors fixed along the way

- Direct-call `PseudoCALL` crashed the MC layer (`fixup_capstone_invalid` — the
  auipc+jalr CALL relocation is not wired for Capstone). Fixed by reusing
  `PseudoLLA` + plain `jalr` (a working `%pcrel` fixup) instead.
- Dead cap-init loop left one static `cjalr` in the binary; dropped it (integer
  app has no capability globals).
- Buildroot: `.c.S` not regenerated on `#include` edits (pattern rule watches the
  wrapper) → force `rm` the `.c.S`.

## What remains

Apply the identical `create_domain` change to the FPGA monitor
(`caplifive-system/.../capstone-sbi/sbi_capstone.c`), rebuild the board image,
build the gp-free globals domain for the board, and run — a silicon smoke that
empirically confirms cscratch delivery on the RTL, then the cycle number
(Experiment A ambient PureCap cost — secondary; the paper's headline is the 1-6%
boundary overhead from the SQLite study). Submodule (QEMU + monitor) edits are
kept as local experiments (no submodule-source commits), reproducible from the
snippets above and `tests/runtime-qemu/gp-free-domain/README.md`.
