# The S-10 fix adds a combinational loop, measured single-variable — and the flown bitstream carries it

Found 2026-08-25 while lint-gating an unrelated RTL change. **This corrects a claim I made to the
board lane earlier**, which was that the loop belonged only to a rejected variant of the fix.

## The measurement

The lint gate, run on two commits whose only `core/` difference is `wt_dcache_mem.sv` (+62/−1 —
the S-10 fix itself), with no `anvil_build` change between them so the generated `.anvil.sv` files
are valid for both:

| commit | S-10 fix | UNOPTFLAT | gate |
|---|---|---|---|
| `39b21639d` | ABSENT | **39** | **PASS** |
| `80843404c` | PRESENT | **40** | **FAIL** |

Diffing the signal *sets*, not the counts, names it exactly:

```
cva6.gen_cache_wt.i_cache_subsystem.i_wt_dcache.rd_ctag
```

One signal added, none removed. Single-variable, because that one file is the entire delta.

## Why this corrects the record

I told the board lane that the `UNOPTFLAT 39 -> 40` recorded in `wt_dcache_mem.sv`'s own comment
belonged to a **rejected** variant — the one that would have added a fourth `rd_ctag_src_o` code —
and that the shipped fix was therefore clean. The comment does say the rejected variant cost a
loop. **The shipped fix costs one too, and it is on `rd_ctag`.**

`4fee13b2d`'s own subject said so all along — *"S-10 FIX: works in simulation, and costs a
combinational loop — NOT ready to merge"* — and nothing in `f231b5af0..80843404c` resolves it. It
was merged at `3d3ed1502` "for synthesis validation", which is a legitimate reason to build it and
not a reason to fly it.

**What I got right and should not be over-corrected:** this is a *Verilator* UNOPTFLAT, not proof
of a physical loop. The bitstream built, so `write_bitstream` DRC `LUTLP-1` passed — with S-10b as
the positive control that this check fires, since its real loop *did* block bitgen. The two
statements are compatible: the design is buildable and it carries a flagged hazard.

## Why it still matters

CLAUDE.md is explicit that a hash is ready when synthesis has RUN, and that **feeding a new signal
into a cone that already carries a combinational loop is the highest-risk edit available, with
every check we have blind to it.** `rd_ctag` is now such a cone, and it sits on the capability tag
read path — the exact path under investigation.

Consequences:

- **`caplifive_s10fix_80843404c.bit`, the image flown for most of this investigation, carries it.**
  That does not invalidate results taken on it, but it belongs in any writeup as an
  uncontrolled difference alongside the WNS gap (−16.400 there against −10.629 at `39b21639d`).
- **The S-10 fix should not be merged to `fpga-testing-dev` in this state.** Its own commit says
  so; this is the measurement behind that sentence.
- **A new RTL change must not be based on `80843404c`**, because the gate cannot pass there and a
  failing gate cannot distinguish the new change from the inherited one.

## What was done about it

The S-07 recorder clear (`84ed6eafb`) was rebased from `80843404c` onto **`39b21639d`**, where the
gate PASSES at 39 with the signal set byte-identical to that base. `39b21639d` is also already
proven synthesizable — it has an exit-0 archive with a bitstream — which is the strongest available
prior for a 32-line additive change on top.

**The baseline was deliberately NOT re-baselined to 40.** Updating the reference to make the gate
pass would have hidden a real, attributable regression, which is the failure mode the gate exists
to prevent.
