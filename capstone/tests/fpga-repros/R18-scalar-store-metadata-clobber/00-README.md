# R-18 — a scalar in the upper half of a 16-byte cache row is silently zeroed on silicon

**Status: reproducible silicon defect. NO mechanism — every mechanism we proposed has been refuted,
including by our own tests. This package is a REPRODUCER, not an explanation.**

## The defect in one paragraph

A `-O0` loop that mixes capability traffic with ordinary scalar locals can have one of those scalars
**silently set to zero** part-way through — no trap, no tag violation, nothing in any log. The same
byte-identical binary computes the correct answer under QEMU and the wrong one on the FPGA. Which
variable is hit depends only on where the compiler placed it. If a *loop-control* variable lands in
the affected position the loop runs **extra iterations** instead of producing a wrong value.

## What is established

* **Reproducible and deterministic.** Four frozen, checksummed images below. `c8` returned
  67699255 on **seven** separate boots (cycles 44067–44098).

  *Correction (2026-08-07).* An earlier revision said the failing arm and its control "differ only
  in where the `-O0` allocator placed the accumulator". That was wrong — they also differ in entry
  VA, read straight off the artifacts:

  | image | entry VA | result |
  |---|---|---|
  | `c8`  | `0xf0000` | 567 — damaged |
  | `c0`  | `0x30000` | correct |
  | `sn8` | `0x30000` | 567 — damaged |
  | `sn0` | `0xf0000` | correct |

  Taken pair-by-pair that is a confound. Taken together it is the opposite: the VAs are **crossed**
  between the two pairs, so each VA hosts one damaged and one correct image, and base VA is
  *excluded* as the cause while `FDREG_SHIFT` — the accumulator's row offset — tracks the damage
  across both. The corrected claim is stronger than the one it replaces, but it has to be stated
  this way rather than by asserting a single variable that was not in fact held.
* **Necessary condition:** the victim is always in the **upper 8 bytes of its 16-byte cache row**
  (row offset 8 or 12, never 0 or 4) — 9 of 9 builds where the victim was measured directly.
  **It is necessary, not sufficient:** roughly 10 *undamaged* upper-half scalars appear across the
  same dataset, so this constrains the search rather than explaining anything.
* **The slot is overwritten, not skipped.** With the accumulator initialised to a sentinel of
  1,000,000 it returns **567**, not 1000567 — so the location is written and counted up from there.
  *(Caveat: N=1. Worth repeating with several sentinel values.)*
* **QEMU is correct** for every variant.

## What we RULED OUT (so you need not spend time on these)

| ruled out | how |
|---|---|
| an over-wide capability store | a witness immediately above the store reads back bit-exact after 576 stores; `extract_transfer_size` pins STC at 8 bytes/one beat |
| store misclassification via the write-user sideband (`st_wr_cap = \|wr_user_i` → dual-bank write) | directed test `scalar-store-cap-operand.S`: a plain `sw` whose data register **provably** holds a real capability (CAPPRINT: Type 1, Perm 7, bounds set) is **not** misclassified and does **not** dual-bank write — **PASS** |
| any single-address anchor for the victim | fitted against all builds; best 13/19 |
| distance from the capability store | same value reproduced at 3× the distance, different row, different frame size |

We also do **not** recommend gating the WB-port metadata forward on validity: `issue_read_operands.sv:765`
already has `cap_result.result_metadata` in **both** arms of its ternary, so that change would be a
**no-op**. We nearly sent that as a fix and withdrew it.

## The one lead we have not tested

All three measured reset points — **9, 72, 558** — are multiples of the inner trip count (9), i.e.
they land exactly on **outer-pass boundaries** (p ≈ 1/729 under a uniform-over-iterations null). That
points at something happening **once per outer pass** rather than once per iteration. Stage 28
(`FDREG_INNER`) decouples the inner trip count so this is falsifiable; built, not yet run.

## Why we cannot take it further from here

The effect has never been reproduced in Verilator, across six directed tests at both RTL revisions.
The failing code runs inside a capability domain after `capenter` on a monitor-carved stack, and we
could not construct a bare-metal directed test that reproduces it. Narrowing further needs
visibility we do not have from outside the RTL — the signal that would settle it is `st_wr_cap` and
`bank_we` at the cycle the victim's dword retires.

---

## Relationship to `RTL-store-user-metadata/` — read that first

That package already established the RTL routing, and this one does **not** rediscover it:

> every store routes the capability-metadata shadow of its **data** register into the dcache
> write-user sideband, ungated — `load_store_unit.sv:1003-1020`, `store_unit.sv:344-346`,
> `store_buffer.sv:172-176`; and the write buffer tracks `data` per byte but `user` as **one flat
> field with no per-byte mask** (`wt_dcache.sv:70-79`).

It also identified bit 27 as `bounds.cursorless` in `cap_metadata_t`
(`core/include/ariane_pkg.sv:609-637`).

Its status line reads *"code-level RTL observation, NOT a demonstrated software-visible defect"*,
and it left one thing explicitly open:

> could **not** trace a path from `data_wuser` into a plain `lw`'s returned data … That remains
> UNRESOLVED and would need the `wt_dcache_mem.sv` fill/writeback merge path.

This package was originally written claiming to CLOSE that open question via the store side. **That
claim is retracted** (see the box above): no path from a scalar store's `wr_user_i` to a non-zero
value has been demonstrated anywhere. The open question above is still OPEN. What this package
contributes is the **software-visible measurement**, not an explanation. The routing is prior work.

## What the repro shows

Four frozen images in `src/`, all instrumentation **mode 0** (`fdreg_fpga_app.c` sets
`LADDER_INSTR_MODE 0`) — mode 4 is a confirmed miscompute trigger and a previous `0x08000000`
sighting was traced to it, so a defect repro must not carry it. Verified: zero `minstret` reads.

| image | accumulator lands | expected on silicon |
|---|---|---|
| `c0.dom` | row offset 4 — **lower** half | `0x04090240` (p=64, k=9, qc=576) **correct** |
| `c8.dom` | row offset 12 — **upper** half | `0x04090237` → qc=**567** |
| `sn0.dom` | lower half, accumulator starts at 1,000,000 | `1000576` **correct** |
| `sn8.dom` | upper half, accumulator starts at 1,000,000 | **567** |

`sn8` is the decisive one. The accumulator is initialised to **1,000,000**; if increments were
merely being lost it would return 1000567. It returns **567** — the slot was *overwritten* and
counted up from there.

QEMU computes the correct value for all four.

## Reproducing

    source capstone/tests/capstone-test-env.sh
    bash capstone/tests/fpga-repros/R18-scalar-store-metadata-clobber/run.sh

The images are **frozen and checksummed** (`SHA256SUMS`) because the effect depends entirely on
where the `-O0` allocator places the accumulator; rebuilding can move it and cure the fault.
`run.sh` verifies the checksums before doing anything. To inspect a layout:

    python3 capstone/tests/runtime-qemu/silicon-ladder/extract-frame-layout.py src/c8.dom src/c0.dom

### Rebuilding the frozen images from source

Verified 2026-08-07: this reproduces `c8.dom` **byte-for-byte** (`sha256` starts `9ecd8c6f9eb2b23d`).

    source capstone/tests/capstone-test-env.sh
    cd capstone/tests/runtime-qemu/silicon-ladder
    DOMAIN_GLUE=interp DOMAIN_BASE_VA=0xf0000 \
      DOMAIN_EXTRA_CFLAGS="-DFDREG_STAGE=19 -DFDREG_SHIFT=8 -DFDREG_GAP=0" \
      bash build-ladder-domain.sh fdreg_fpga_app.c /tmp/c8.dom

Per-image parameters: `c8` = SHIFT 8 @ `0xf0000`; `c0` = SHIFT 0 @ `0x30000`; `sn8` = SHIFT 8
sentinel @ `0x30000`; `sn0` = SHIFT 0 sentinel @ `0xf0000`.

Three things about this recipe are worth stating because each one cost a rebuild:

* **`DOMAIN_GLUE=interp` is load-bearing.** Without it the build takes the generated-prologue path,
  which emits `lla fdreg_defs` for the large-RO copy of the 1296-byte `fdreg_defs` array. That
  symbol is `static`, so it has local binding and the link fails with
  `ld.lld: error: undefined symbol: fdreg_defs`. The generator's guard only rejects `.L` symbols,
  not local-binding ones — a real latent bug in `gen-gp-captable-glue.py`, filed here rather than
  worked around silently.
* **The base VAs above are the ones in the artifacts.** An earlier write-up of this recipe gave
  `0x30000` for `c8` and `0x60000` for `c0`; neither matches, and following it does not reproduce
  the frozen images.
* **`src/fdreg_kernel.h` is now the header the images were built from.** It previously carried no
  stages 30+, so `gvf0`/`gvf6` — checksummed into this package — could not be rebuilt from it.

## Supporting evidence, from the full trail

* The victim is in the **upper half of its row in 9 of 9** builds where it was measured directly
  (row offset 8 or 12, never 0 or 4). Undamaged builds also carry upper-half scalars, so this is a
  real constraint rather than an artifact of where slots land.
* One build returns **`0x08000237 = 0x08000000 + 567`**, and `0x08000000` is exactly
  `compress_cap` of a **null capability** — so that value is metadata-shaped, not a number.
  *(A `clobber + (576 − reset)` decomposition of every victim was also recorded; it is an
  arithmetic identity with two free parameters per observation and has no predictive content.
  Do not cite it.)*
* **All three measured reset points — 9, 72, 558 — are multiples of 9**, i.e. they land on
  OUTER-PASS boundaries. That is the most interesting unexplained fact here, and it points at
  something that happens once per outer pass rather than at the victim's own store.
* A loop-**control** variable in the affected slot produces **extra iterations** instead of a wrong
  value, and cycle counts confirm the extra iterations really executed (69081 vs 44001).

## What this is not

* **Not reproduced in Verilator.** `verif/tests/custom/capstone/stc-neighbour-load.S` and
  `stc-counter-pair.S` pass at both RTL revisions, cycle-for-cycle identical, across five rounds of
  added fidelity. They are bare-metal M-mode and we could not construct a directed test that
  produces stale WB-forwarded metadata on a scalar store's `rs2`. **The clean simulation means the
  trigger was never created — it neither confirms nor refutes the chain.** This is the single
  biggest gap.
* **Not a stable rate.** One clobber in 576 iterations in most builds, 558 in one. No account of
  why; it is the strongest argument that more than one thing may be involved.
* **Not the mode-4 harness artifact.** All images here are mode 0, verified.

## Bitstream

Measured on `caplifive_65536_nodes.bit`. The chain is present at `capstone-ariane` HEAD
`458982093` and at `7aac52f93` (the commit this bitstream is built from); `git diff` between them
touches none of the files involved.

## Fix

**No fix is proposed, because the cause is not established.** An earlier version of this file
suggested gating the WB-port forward on validity "matching the scoreboard-port version" — that
would have been a **no-op**: `issue_read_operands.sv:765` has `cap_result.result_metadata` in both
arms of its ternary and does not sanitise to zero.

The one change defensible on its own merits, independent of this measurement, is to classify
capability stores **by opcode** rather than by `|wr_user_i` (`wt_dcache_mem.sv:138`) — because the
null-capability encoding is `0x08000000`, so a value-based test misclassifies null-cap stores.
That needs a reflash and is the project lead's call.

## What would settle the actual cause — none of it needs the board

1. Directed Verilator test that creates the condition: a capstone-FLU op producing `aN`, then
   `sw aN, 12(sp)` into a bank-1 slot; read `wr_user_i` off the RVFI trace. If it is zero, the
   store-misclassification family is dead. ~13 s.
2. The missing control: a passive witness at victim−8 **and** victim+8 in the `c8` geometry.
   Self-clobber and dual-bank splash predict different slots; neither appearing kills the family.
3. Repeat the sentinel — `sn8` is **N=1**, and it is the sole basis for "zeroed rather than lost".
   Sweep `FDREG_SENTINEL` over ≥3 values; the answer must be sentinel-independent at 567.
4. Explain `c8` vs `gp16`. Until a condition is stated that is true in one and false in the other,
   there is a necessary condition and no cause.

Full report: `capstone/agent-handoff/history/07-08-2026_RETRACTED_scalar-store-metadata-mechanism.md`.
Trail: `capstone/agent-handoff/history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.
