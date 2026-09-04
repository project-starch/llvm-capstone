# S-10b is NOT synthesizable — and the memory reading that missed it

**Date:** 2026-08-21
**Verdict:** the S-10b **defect** is real and reproduced; the S-10b **fix as written**
(`c867dfcbb`) cannot be built. It must come off every merge path.

## The single-variable result

`c2211c9a8` differs from the last **successful** build `80843404c` in RTL by **S-10b alone**
(`core/store_buffer.sv`, the three `[11:3]` -> `[11:4]` hazard compares). No constraint. It failed:

```
ERROR: [DRC LUTLP-1] Combinatorial Loop Alert: 69 LUT cells form a combinatorial loop.
       One net in the loop is i_ariane/i_cva6/csr_regfile_i/gen_register
ERROR: [Vivado 12-1345] Error(s) found during DRC. Bitgen not run.
ERROR: [Common 17-39] 'write_bitstream' failed due to earlier errors.
```

Three builds, and the attribution is now clean:

| build | RTL vs reference | constraint | outcome | wall | peak RSS |
|---|---|---|---|---|---|
| `80843404c` | S-07 + S-10 | no | **exit 0**, no LUTLP at all | 1h45m | 21.53 GB |
| `eaa4e7984` | + **S-10b** | yes | `LUTLP-1`, **61** LUTs | 1h45m | 21.20 GB |
| `c2211c9a8` | + **S-10b** | no | `LUTLP-1`, **69** LUTs | **6h30m** | **29.77 GB** |

Same loop both times — 7 cells in `ex_stage_i/rev_node`, 7 in `ex_stage_i/lsu_i/i_load_unit`,
plus `csr_regfile_i`. The multicycle constraint is **exonerated** as the cause of that failure;
it only changed the loop's size and made P&R converge faster by relaxing the timing it had to
meet.

Neither successful build mentions `LUTLP` or `Combinatorial Loop` even once, and that zero is
meaningful: those logs carry 106 `CRITICAL WARNING` lines and 44 `DRC` lines, so the matcher is
demonstrably live.

**Mechanism, consistent with the loop membership but NOT independently proven:** `store_buffer`'s
`page_offset_matches_o` feeds `load_unit`'s stall logic, and `load_unit/cap_clear_addr_q` is inside
the loop cone. Narrowing the compare changed the function feeding a cone that already carried a
combinational loop. Recorded as the leading hypothesis, not as established.

## The lesson this is a textbook case of

CLAUDE.md already says it: **"Lint and audit are NECESSARY, NOT SUFFICIENT. Only synthesis proves
synthesizability."** S-10b was described in this repo — by me — as *"a strict reduction: 9-bit ->
8-bit comparator, no new signal, no new fan-in"*, and that description is **structurally accurate
and still wrong about the outcome**. A strict reduction inside one module closed a loop in another,
because the signal it produces crosses into a cone that already looped. CLAUDE.md's other standing
warning covers exactly this and was not applied: *"Never feed a new signal into a cone that already
carries a combinational loop."* Changing an existing signal's function counts.

## The reading error worth recording separately

At **4963 s** (82 min) the trace showed peak **21.13 GB** and RSS flat at 15.86 GB for two minutes.
That was reported as *"memory is fine, synth_design is done, roughly 25 minutes left."*

The memory half was right and stayed right for hours — peak held at 21.13 GB through hour 4, and
first exceeded the previous worst (21.53 GB) only at **19877 s = 5h31m**, reaching 29.77 GB at
5h39m. The **runtime** half was wrong by 5 hours.

Two distinct mistakes:

1. **A mid-run peak was compared against other runs' FINAL peaks.** A peak-so-far is a lower
   bound; a final peak is a value. They are not comparable, and saying "21.13 sits mid-envelope"
   implied a completed measurement.
2. **Flat RSS was read as "converging".** It was the opposite: a router failing to converge holds
   memory steady while burning wall-clock. The signal that was already available and ignored was
   **elapsed time against the 1h45m reference** — at 82 minutes it was fine, but by hour 3 it was
   the only thing that mattered.

The kill criteria written in `20-08-2026_19-30-00_s10-loop-risk-overstated.md` would **not** have
fired: they say *"wall time exceeds ~2x the reference **with RSS not plateauing**"*. That
conjunction is wrong. **Runtime alone past ~2x is a failure signal regardless of memory** — here
it would have flagged the run at ~3h30m and saved three hours.

## Consequences

* **S-10b comes off the merge path.** `s10b-fix` (`c867dfcbb`), `s10-merge-candidate`
  (`c2211c9a8`), `timing-multicycle` (`fb228796e`) and `timing-directive-explore` (`8696dfdc9`)
  all contain it and are therefore all **unbuildable**. Do not hand any of them to synthesis, the
  board, or another lane.
* **The S-10b DEFECT still stands.** Both routes were reproduced in simulation (DATA route: a
  read returning `0x0` before and `0xfedcba9876543210` after; TAG route: 0/8 legs trapping
  pre-fix, 8/8 after) and the cost was measured at +0.06% overall. The store-buffer word-vs-granule
  hazard compare is real. Only this **implementation** of the fix is dead.
* **A different fix shape is needed** — one that does not put the widened compare into the
  `load_unit` stall cone. Not designed here.
* **`f231b5af0` is unaffected** and remains the silicon-proven cut: it predates S-10 and S-10b
  entirely.

## No artifacts beyond the log

`work-fpga/` holds only `.xci` files — **no routed `.dcp`**, so no timing forensics exist for this
run and none can be produced without re-running it. The collector correctly reported
`SKIPPING timing enumeration`. Place-and-route did complete (`Phase 13 Post-Route Event
Processing`); the failure is at `write_bitstream` DRC.
