# Proposal: deriving the domain `gp` data cap on silicon (replace C_GEN_CAP)

**Status:** PROPOSAL (2026-07-22), branch `capstone-gp-free`. Fixes the one
remaining silicon blocker in the gp-free Experiment-A work. Root cause + full trail:
`history/22-07-2026_18-05-00_gp-free-silicon-smoke-firmware-fixed-createdomain-hangs.md`.
Propose-before-implement per the repo convention.

## The problem (one line)

Our monitor `create_domain` mints the domain's `gp` (an image-covering data cap)
with **`C_GEN_CAP`**, a **QEMU-only debug instruction** (`helper_csdebuggencap`,
custom-2 funct7 `0x40`) that **fabricates** a capability from `(base,end)`. Real
silicon forbids fabrication (capability monotonicity); the RTL decoder leaves `0x40`
unimplemented → garbage cap → the follow-up `stc` faults → M-mode hangs in
`capstone_error`=`while(1)`.

## What the reference does (found in-tree — no board owner needed to learn this)

- **`capstone-sbi-domain` package `create_domain`** (the pristine reference,
  `caplifive-system/sw/buildroot/package/capstone-sbi-domain/.../sbi_capstone.c`)
  builds every domain cap with **`split_out_cap` → `__split`** (real SPLIT, funct
  `0x06`, DERIVES from the monitor's RWX memory pool) and **delivers NO `gp` at
  all**. `USE_GEN_CAP` is off; `C_GEN_CAP` is never used on the real path.
- **capstone-c's domain model deliberately avoids static writable globals in the
  image.** Domains take a `runtime`/metaparam capability and get memory via
  `runtime->malloc(...)` / `runtime->shared[...]` (see `capstone-c/samples/enclave.c`).
  So the reference never needs an image-covering `gp` — which is *exactly the
  compatibility gap our gp-free work closes* (running unmodified C with static
  globals addressed via `gp`).
- Legit primitives on the RTL: `SPLIT` (partition a cap), `CAPPERM`/`CAPBOUND`/
  `CAPCREATE` (custom-3). **None can ADD permissions** (monotonicity) — a writable
  `gp` must come from a writable parent, not from the executable `dom_code`.

## Proposed fix — derive `gp` by SPLIT-partitioning the image into code|data

Partition the domain image `[base, base+code_size)` at the text/globals boundary and
derive two disjoint caps from the monitor's RWX pool (both via SPLIT — no
fabrication, no added perms, no aliasing):

- **PCC** = `[base, base+text_size)` → execute (as today, `C_SET_CURSOR` to entry).
- **gp_cap** = `[base+text_size, base+code_size)` → read/write; covers the whole
  globals span (`.rodata`+`.data`+`.bss`). This is `gp`.

Deliver `gp_cap` to the domain via the **cscratch top slot** (our existing channel —
the entry glue already does `ldc gp, END-16; delin`), or equivalently a sealed-region
slot. The gp-free codegen is unchanged: all globals are reached with `scc gp,<abs>`,
and every global address lands inside `gp_cap`'s bounds.

### Why this is correct on silicon
SPLIT partitions a linear cap into two disjoint linear caps — implemented on the RTL,
used pervasively by the reference `create_domain` already. The two results don't
overlap (no aliasing), and neither adds a permission the parent lacked (the pool is
RWX). `gp_cap` is R/W over the globals only; PCC is execute over the code only.

### The one input we need: the text/globals boundary (`text_size`)
`create_domain` currently gets `code_size` (whole image). It needs the split point.
Options (pick in this order):
1. **Loader passes it.** The `.dom` loader (our controller / the kernel driver) reads
   `text_size` from the ELF (end of the last executable `PT_LOAD` / `.text`) and
   passes it as a new `create_domain` arg. Cleanest; no linker changes.
2. **Linker symbol.** Emit `__capstone_globals_start` in the domain linker script;
   the loader reads it. Keeps the ABI explicit.
3. Align the image so text and globals are on separate SPLIT-able boundaries.

### Layout requirement
The domain linker script must place **all globals contiguously after all code**
(`.text` then `.rodata`/`.data`/`.bss`), so a single split yields an executable code
cap and a single R/W globals cap. Our gp-free codegen already routes *every* global
(incl. `.rodata`) through `gp`, so read-only globals in the R/W `gp_cap` are fine
(over-permissioned, harmless). Validate no PC-relative `.rodata` access remains under
`-capstone-gp-free` (jump tables/constant pools) — at `-O0` this is already the case
for the proof app.

## Implementation sketch (small, localized)

1. **Monitor `create_domain`** (local experiment file): delete the `C_GEN_CAP` block;
   after `dom_code`/`dom_seal`/`dom_data` are split, additionally `__split` `dom_code`
   at `base+text_size` into `pcc_part` (→ set_cursor to entry → `dom_seal[0]`) and
   `gp_part` (→ `__delin` → store into the cscratch top slot via `dom_data`).
2. **Boundary plumbing**: extend the `IOCTL_DOM_CREATE` args + `create_domain`
   signature with `text_size`; loader fills it from the ELF.
3. **Entry glue / codegen**: unchanged (`ldc gp,END-16; delin`; `scc gp,<abs>`).
4. **Rebuild fw** via the recipe (memory `project_fpga_fw_payload_build_recipe`);
   one board run → controller prints `gpfree-fpga: created domain ID` (create no
   longer hangs) → retval `554745961`.

## Do we need Jason?

**No blocker on him** — the code answered the "how" (reference derives via SPLIT,
never `C_GEN_CAP`; capstone-c avoids image-globals). We can implement + test path 1
now. **One optional async confirmation** (send, don't wait): *"To run unmodified C
with static globals in a domain we derive an image `gp` by SPLIT-partitioning the
domain image into an execute code cap and an R/W globals cap (globals laid out after
text), delivered via the cscratch slot — instead of the QEMU-only C_GEN_CAP. Does
that match how capstone-c would deliver an image data cap, or is there a preferred
cap-table slot for it?"*

## Open risks
- If any domain needs both execute AND read of the same bytes (e.g. embedded
  read-only data inside `.text`), the strict code|data split needs the linker to
  hoist that data into the globals span. Check the proof app first; generalize with a
  linker-script pass if needed.
- Confirm `stc` of a linear (delin'd) cap into the cscratch slot works with the
  split-derived `gp_cap` (it's the same store shape that hung only because the source
  cap was garbage).

See memories: [[project_silicon_gp_delivery_boardowner_guidance]],
[[project_fpga_fw_payload_build_recipe]].
