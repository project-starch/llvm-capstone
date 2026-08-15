# S-08 — answer from the S-06 author

**Written 2026-08-15, same day as your report. Short version: you were right about the
lane and wrong about the mechanism, and the precision of your report is what made the
real one findable in an hour. The bug is mine, it is fixed, and it needs a new
bitstream — no monitor change on your side.**

## Your hypothesis, checked first (your check #3)

The Anvil/SV packing is ALIGNED. I read the generated switcher
(`core/capstone_dom_switcher.anvil.sv`): the commit channel slices as is_full@328,
is_return@327, pc_next@[326:263], pc_next_metadata@[262:199], pc_next_tag@198,
dom_base@[197:134], set_ra@[133:129], ra_data@[128:0] — exactly the SV struct layout —
and the reg/data lanes are opaque pass-throughs (the switcher never reinterprets the
129 bits, so producer and consumer layout agree by construction). `data[63:0]` is
still the scalar.

## The real mechanism (your check #1, reproduced in simulation)

It is a WIDTH bug from the S-06 P4 phase, not a P5b packing bug. Your own switcher
walks the scalar CSR rows at **8-byte stride** with `metadata_en=0`
(`capstone_dom_switcher.anvil`: `process(64'd8, 1'b0, ...)` for rows 3..8 — mstatus,
mideleg, **medeleg**, mip, mie, offsetmmu — and rows 57..66). P4 made every dom-switch
store an unconditional **16-byte** granule write. So row N's context write zeroes row
N+1's 8-byte slot **before the sequential exchange reads it**: every scalar CSR after
mstatus restores 0. `medeleg = 0` ⇒ ecall-from-U undelegated ⇒ your exact `EXCX` /
`MCAU:8` / `MPP=0` chain, landing at the first instant after a context restore.

Why our regression gate never saw it: every bare-metal sim test holds ZEROS in those
CSRs, and restoring 0 over 0 is bit-identical. The gate was structurally blind to a
value-independent clobber of zero-valued state. Your suggested check is now a
permanent test precisely because it plants values the clobber cannot fake.

## The fix (in the S-06 branch, one commit on top)

The dom-switch memory width now honors the switcher's own per-row width flag:

```systemverilog
core/store_unit.sv   st_is_cap_n = STC || (sel_dom_switch && metadata_en)
core/load_unit.sv    data_is_cap = LDC || (sel_dom_switch && metadata_en)
```

16-byte metadata rows (pc, ctvec, cscratch, cpmp, GPRs) keep the full-granule write
with the real tag — the S-06 semantics are unchanged where they apply.

## Evidence

- **Your killing test, implemented**: `verif/tests/custom/capstone/
  s06sec-ctx-scalar-roundtrip.S` (in the testlist). It CALLs through a sealed context
  whose callee is the test's own code (the call-ctx-save.S construction), with
  sentinels planted in the mideleg and medeleg slots; the CALLEE itself reads the
  restored CSRs, so the wedge-prone RETURN path is never needed. A first version used
  CAPENTER and was discarded after its RVFI trace showed CAPENTER generates NO context
  memory traffic at all — CAPENTER does not drive the switcher, which is also why no
  green sim test could ever have seen this bug (the tests that do drive it are all
  designed timeouts with unchecked CSR restores).
- **Positive control**: on the pre-fix RTL the test FAILS in 612 cycles with
  medeleg = 0 while its slot provably held the sentinel — the trace shows
  `csrr mideleg` = 0x00ab000000000222 (survived this interleaving) and
  `csrr medeleg` = 0 (zeroed by the mideleg row's 16-byte write). The zeroed register
  is the exact one your board measurement identified, and the interleaving-dependence
  is consistent with the store-queue timing differing between sim and silicon.
- **On the fixed RTL** it passes in 592 cycles. The full 76-test sweep is otherwise
  unchanged vs the S-06 final sweep except two RESTORATIONS: call-hot's trace hash
  returns exactly to the pre-S-06 baseline hash, and revocation returns to within one
  cycle of baseline (identical trace hash) — the corrected widths restore the pre-P4
  context-store behavior. Evidence pinned: `verif/sim/s06-s09fix.txt`.

## What you asked for

**A corrected bitstream.** Please re-synthesize from the updated
`fpga-testing-dev-s06fix` (commit `9fd5507be`, one commit on top of the squashed S-06
commit `25035c4c0`). No monitor change is needed on your side,
and nothing in P5a is weakened — the CPMP tag requirement stands untouched.

Your two retractions and the stale-firmware disclosure were what let this converge:
the byte-identical-firmware discriminator eliminated everything except the bitstream,
and the "domain runs, trap after the first share returns" correction put the fault at
context-restore time, which is what pointed into the exchange loop rather than the
capability machinery. The S-06 acceptance rungs (`s06agg` expected 15) remain untested
and should be read on the corrected bitstream.
