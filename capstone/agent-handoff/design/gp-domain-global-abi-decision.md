# Design decision — how a gp-free domain addresses globals on real silicon

**Decision.** A domain's global pointer `gp` is a **per-global capability table**
derived, in the domain entry glue, from the **data-allocation capability the
monitor provides at domain creation** (delivered via `cscratch` = `dom_data`).
Each global lives in that data region with its own data capability, and code
reaches global `i` by loading its capability from `gp[i]` (`ldc rd, i*16(gp)`).
`gp` is **never** a partition of the executable code image.

**Status:** confirmed correct by the board owner and validated on hardware.

## Why (the constraint that forces this)

On the captype-fixed CVA6 Capstone RTL, a capability split from the **code**
(execute) image cannot be used as a **data** base: any `scc gp; delin; load/store`
through it wedges the hart (M-mode, PC→0). This is not shrink-, store-, nesting-,
or ABI-related — even a bare read faults. Full root-cause trail:
`history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md` (UPDATE 23-07). QEMU is
permissive and hid this; the RTL enforces it.

The board owner's guidance (verbatim intent): *"You must derive the cap for gp
from something readable/writable. capstone-c creates that from a capability
intended for data allocation, which is provided when the domain is first
created."* Our design matches this exactly: `dom_data` (the cscratch region the
monitor mints at `create_domain`) **is** that data-allocation capability, and the
glue splits `gp` (and each global's storage) from it — identical to capstone-c
(`capstone-c/src/codegen.rs:264-298`).

## Mechanism (validated)

- Entry glue (`tests/runtime-qemu/gp-free-domain/start-gpfree-captable*.S`):
  `sp = cscratch`; carve an N-entry cap-table off the top of `sp`
  (`split gp, sp, end-N*16`); for each global carve `size` bytes from `sp`, zero
  them, `stc` the cap into `gp[i]`; remainder of `sp` = stack. gp is re-derived on
  every entry — **no gp memory round-trip, no code-image access**.
- Compiler (`-capstone-gp-captable`, gated, corpus byte-identical when off):
  `lowerGlobalAddress` emits `ldc rd, i*16(gp)` (indexable globals); an AsmPrinter
  `.capstone_gp_table` descriptor gives sizes to the build (consumed at *build*
  time, not runtime — a runtime descriptor read would hit the same
  execute-cap-data-read wall). See `plans/gp-captable-codegen-plan.md`.
- Because sizes must be known where the table is built and the glue cannot read
  the image, the table build uses **compile-time-baked sizes** (capstone-c does
  the same), not a runtime table walk.

## Validation

- Hand-crafted probe: **passes on silicon** (retval 554745961) — the mechanism is
  hardware-correct (`history/...` UPDATE 23-07 "FIX VALIDATED").
- Compiler-built domain (`captable_zeroinit_app.c`, `-capstone-gp-captable`):
  QEMU correct (554745964); on silicon it **creates, enters, does `ldc gp[0]`
  global access, and exits cleanly** — the crash is gone. One open
  implementation bug: the returned value is off (+10/element) on silicon only;
  under diagnosis (needs a gdb board session), not a design flaw.

## Consequences / open items

- Writable globals must live in the data region, not the code image → the linker
  layout and any initializer copy must respect that (initializers handled by
  runtime stores, reusing the `CapstoneCapGlobalInit` PC-relative init table).
- Monitor is **unchanged** — the glue ignores the older code-split gp it still
  stashes; a follow-up can drop that and guard the degenerate no-globals SPLIT.

Related: `capability-globals-init-decision.md`,
`plans/gp-captable-codegen-plan.md`, `../history/22-07-2026_18-05-00_*`.
