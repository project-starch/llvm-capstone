# R-18 — a plain scalar store in the UPPER half of a 16-byte cache row is silently zeroed

**Status: software-visible defect, reproduced on silicon, root-caused in RTL. NOT reproduced in
Verilator (see "What this is not").**

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

**This package closes that open question — through the STORE side rather than the load side, and
through exactly the file that was named.** The new contribution is (a) the software-visible
consequence, and (b) the `wt_dcache_mem.sv` links below. The routing itself is prior work.

## The two additional links, in `wt_dcache_mem.sv`

| line | what it shows |
|---|---|
| `wt_dcache_mem.sv:138` | `assign st_wr_cap = \|wr_user_i;` — a store is classified as a capability store **by VALUE, not by opcode**. Combined with the ungated routing above, an ordinary `sw` carrying non-zero metadata is misclassified. |
| `wt_dcache_mem.sv:230-238` | a classified store sets `bank_req = '1; bank_we = '1` — it writes **both** banks of the 16-byte row, not the one its address selects. |
| `wt_dcache_mem.sv:156-158` | `bank_wdata[k][j] = … (((st_wr_cap) && (k==1)) ? wr_user_i : wr_data_i)` — **bank 1 (the upper 8 bytes) is the only bank that can receive `wr_user_i` instead of the store's data.** `bank_be` applies the same byte-enable to both banks, so for a store addressed into bank 1 the metadata lands on its **own** byte lanes. |

**Net:** a plain `sw` whose address is in the upper 8 bytes of a 16-byte row can have its own slot
written with capability metadata instead of its data. Where those metadata bytes are zero at the
store's lanes, the variable is **silently set to zero** — no trap, no tag violation.

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
* Every victim decomposes as `clobber_value + (576 − reset_iteration)`: `567 = 0 + (576−9)`,
  `504 = 0 + (576−72)`, `18 = 0 + (576−558)`, and one build returns
  **`0x08000237 = 0x08000000 + 567`** — clobbered with bit 27 set, i.e. `bounds.cursorless`.
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

## Suggested fix

Either link breaks the chain, and both need a reflash:

* gate the WB-port forward on validity in `core/issue_read_operands.sv:690` (its scoreboard-port
  sibling already checks `cap_result.valid`; the `rs1`/`rs3` siblings have the same shape);
* and/or classify capability stores **by opcode** rather than by `|wr_user_i` in
  `wt_dcache_mem.sv:138`.

Full report: `capstone/agent-handoff/ref/SILICON-DEFECT-scalar-store-metadata-clobber.md`.
Trail: `capstone/agent-handoff/history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.
