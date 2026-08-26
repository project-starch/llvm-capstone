> ## ⚠ SUPERSEDED 2026-08-26 — THE GRANULE FILTER WAS DROPPED
>
> This table was written for a bitstream that filtered the LDC recorder on CSR `0x811`, so that
> only the subject granule could be recorded. **That design failed to route** (8 h 38 m, a 56-bit
> granule bus plus a 52-bit comparator across four module levels, at <1% LUT margin) and was
> abandoned. The instrument now going to synthesis is **`s12-ldc-rolling-min`**: rolling,
> unfiltered, with the tag on switch 219 and **no address filter at all**.
>
> What that changes, row by row: every row below that assumes "should not occur once the filter
> is in" is void, because there is no filter. Identifying the recorded load is now the
> **driver's** job — it compares the recorded granule against the subject computed from the `s0`
> GDB read at the wedge, and a foreign granule is a **hard VOID**, never evidence. See
> `00-README.md` and the driver's subject gate.
>
> Kept rather than deleted because the reasoning about positive controls still applies, and
> because a decision table that quietly vanished would be worse than one marked superseded.

# S-12 — the decision table for the pending bitstream, written BEFORE the run

This exists because "a decision table shipped with the bitstream is what makes it the last one".
Every row is committed in advance so no reading can be invented after the fact to fit whatever
comes back.

## What the bitstream adds

    load_unit.sv        drop `&& !s07_ldc0_valid_q`      strict fanin reduction; this term
                                                          already shipped working in
                                                          caplifive_s07debug_18august.bit
                                                          before 83a7d061f reverted it
                        + granule compare against watchpoint_addr on the capture
    load_store_unit.sv  thread watchpoint_addr through
    ex_stage.sv         thread watchpoint_addr through
    cva6.sv             connect the existing watchpoint_addr to the LSU
    ariane_pkg.sv       comment fix only, no logic

With the freeze on `recent_nontrivial_trap_seen_log_q` (already present, `cva6.sv:1015`), the
record becomes **the LAST untagged LDC at the watched granule before the fault** — which, two
instructions after the reload, is the subject.

## Why ONE register arms both sides, and why that is the safety property

`watchpoint_addr` is a single register (`cva6.sv:897`, CSR `0x811`). The **store** watchpoint
(group 9) and the new **LDC** granule filter compare against the *same* one. Therefore:

* one userspace `csrw` arms both, so there is no second address to get wrong;
* **group 9 firing is the positive control for the LDC filter's address.**

That matters because **one of the two answers is an ABSENCE**, and an absence from a mis-armed
filter is indistinguishable from an absence that means something. This makes the run enforce its
own control instead of us remembering to.

## THE TABLE

| group 9 (store side) | LDC record | reading |
|---|---|---|
| **empty** | anything | **VOID.** The address is wrong or arming failed. Neither side carries a verdict. Re-derive the slot and re-run. Do **not** read the LDC record. |
| fires at the subject store | **records an untagged LDC at the watched granule** | **THE LOAD RETURNED UNTAGGED.** The fault is in the load's memory access, and `src` names the leg: `0` L1 hit, `1` miss refill, `2` write-buffer forward. |
| fires at the subject store | **empty** | **THE LOAD WAS FINE.** Memory delivered a tagged value and the fault is in **operand delivery** — everything between the load's writeback and the FLU's `operand_a`. |
| fires | records at a **different** granule | **INCONCLUSIVE.** Should not occur once the filter is in (that is what the filter is for); if it does, the filter is not working and the run is void. |

`src` encoding is RTL, not a driver comment: `wt_dcache_mem.sv:338`
`rd_ctag_src_o = (|wbuffer_be) ? 2'd2 : rd_ctag_src`, with `2'd1` refill (`:327`) and `2'd0` L1
hit (`:332`).

## Preconditions for the run to count at all

1. **A known-good control arm returns first.** A boot whose control fails carries no verdict.
2. **The DBAS guard passes on the subject arm.** The watchpoint address is derived from a previous
   boot's wedge, and DBAS is *not* stable across boots or arm positions — this same investigation
   saw `0x82400000` and `0x82800000` in one boot. On a mismatch the driver refuses to interpret,
   and that refusal is the point.
3. **The selftest fires**, so a zero is a controlled negative.
4. **N > 1.** The fault is sporadic — the same un-probed binary has returned normally on later
   draws — so a single non-firing boot is not evidence.

## What this CANNOT settle, stated now rather than discovered later

* If the record fires, it names the **leg**, not the defect. "Write-buffer forward supplied an
  untagged tag" is a localisation, not a root cause.
* If the record is empty with group 9 firing, "operand delivery" is a **bracket**, not a mechanism —
  and every specific delivery mechanism proposed so far has been excluded (R-20 class, wrong-producer
  scoreboard selection, domain-switch `cnull`, load-syncer mispair). An empty record therefore
  points at a mechanism nobody has yet named.
* The DRAM shadow-tag byte remains **irrelevant** to both rows. It reports DRAM; the load consumes
  the L1 array (`wt_dcache_mem.sv:143`, `:319-338`), and the documented desync runs DRAM-stale-high
  with L1-correct-low. It is not evidence about the load and must not be quoted as such.
