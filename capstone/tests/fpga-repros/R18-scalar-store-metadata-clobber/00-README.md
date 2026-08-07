# R-18 — a plain scalar store in the UPPER half of a 16-byte cache row is silently zeroed

**Status: software-visible defect, reliably reproduced on silicon. NOT root-caused — the causal
chain first recorded here was RETRACTED the same day by an adversarial audit. NOT reproduced in
Verilator.**

> **The four frozen images and their expected values are unaffected and remain valid** — they are
> measurements. What was withdrawn is the *explanation*: see R-18 in `ISSUES.md`. In particular the
> asserted `issue_read_operands.sv` asymmetry does not exist (both arms of the scoreboard-port
> ternary return `result_metadata`), an ordinary `addi` forwards ZERO metadata
> (`ex_stage.sv:1081`), and `c8` fails while `gp16`/`gp32`/`t16` succeed at the **same bank and
> byte lanes** — so bank geometry is a necessary condition, not a cause. Do not hand the mechanism
> to the hardware owner; the reproducer itself is still the right artifact to hand over.

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

## The `wt_dcache_mem.sv` structure — real, but NOT shown to cause the measurement

| line | what it shows |
|---|---|
| `wt_dcache_mem.sv:138` | `assign st_wr_cap = \|wr_user_i;` — a store is classified as a capability store **by VALUE, not by opcode**. Combined with the ungated routing above, an ordinary `sw` carrying non-zero metadata is misclassified. |
| `wt_dcache_mem.sv:230-238` | a classified store sets `bank_req = '1; bank_we = '1` — it writes **both** banks of the 16-byte row, not the one its address selects. |
| `wt_dcache_mem.sv:156-158` | `bank_wdata[k][j] = … (((st_wr_cap) && (k==1)) ? wr_user_i : wr_data_i)` — **bank 1 (the upper 8 bytes) is the only bank that can receive `wr_user_i` instead of the store's data.** `bank_be` applies the same byte-enable to both banks, so for a store addressed into bank 1 the metadata lands on its **own** byte lanes. |

**What is defensible from the above, on its own:** `st_wr_cap = |wr_user_i` classifies capability
stores **by value rather than by opcode**, and the compressed encoding of a **null** capability is
`0x08000000`, not zero (`ariane_pkg.sv:753-834`). So a store carrying null-cap metadata is
misclassified and dual-bank written. That is a clean structural defect worth reporting by itself.

**What is NOT established:** that this is what produces the 567. It requires `wr_user_i != 0` on an
ordinary `sw`, which has never been measured; `ex_stage.sv:1081` zeroes the FLU writeback for
non-capstone ops; and `c8` fails while `gp16`/`gp32`/`t16` succeed at the **same bank and byte
lanes**, which this mechanism cannot distinguish.

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

Full report: `capstone/agent-handoff/ref/SILICON-DEFECT-scalar-store-metadata-clobber.md`.
Trail: `capstone/agent-handoff/history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.
