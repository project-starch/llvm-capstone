# The rev-node cone cut on P1: pre-registration (2026-11-21)

Written before the synthesis-only gate and the full build. RTL: capstone-ariane `p1-revrd-cut` 5155e8e02 = P1
d83d3a633 + one change in `core/ex_stage.sv` (the commit message is the design). The forced mutants are on
`p1-revrd-cut-mutants` b5c955856, simulation only.

## Why

The board lane read these on P1's routed checkpoint, N = 1 per tree (board/meas-pw 26c577393d93, 00cc2382f1eb,
70c6aafff17c):
- All of P1's worst 500 paths cross `ex_stage_i/rev_node`: 461 enter at the unit's record read data and 39 at its read
  valid.
- The worst is -13.173 ns at 40 ns, `dom_switcher cur_idx_q[3] -> issue_read_operands fu_data_q[0][cap_metadata_a][19]`,
  111 levels. Its arrival at the crossing is 26.625 ns, splitting the path 25.344 before and 27.912 after.
- On the spatial-only core (1f1ed11f9) the same start and end families meet timing 62-86 levels shorter.

The wording is fixed until a cut build measures it: "P1's critical paths run through the rev_node cone".

## The cut and its cost

The D$ read response the unit and its taps read is registered, together with the read index and namespace flag it is
decoded under. The grant stays combinational.

Cost, written before the RTL: +1 cycle per record read, an upper bound. From the record-cache model's P1 organisation
on the board's images:

| program | record reads | cost against P1's in-context cycles |
|---|---|---|
| gcc train01 | 16.45 M | +0.039 % (of 41.889 G) |
| mcf test | 181.24 M | +0.125 % (of 144.686 G) |
| perl test | 252.1 M | no silicon base |

## Considered and not batched: a second cut at the next family

The predicted next limiter (cur_idx_q -> lsu_bypass mem_q CE, -4.631 at 56 levels with rev_node disabled) ships uncut.
- **What it is.** mem_q's enable is `lsu_req_valid_i || pop_ld_i || pop_st_i` (lsu_bypass.sv:81-96). That is the LSU's
  issue and its acceptance by the load and store units, which depends on the D$ grant, so the family is the LSU <-> D$
  handshake, a member of the standing UNOPTFLAT list.
- **Cost.** A register there costs a cycle per memory access, not per record read, orders of magnitude above the
  0.2 % rule before any gain.
- **Risk.** It feeds a cone that already carries a combinational loop.
- **The number is an estimate.** It is the whole hierarchy disabled, and the same family reads +3.525 at 48 levels on
  the spatial core.

This build measures where the limiter really is. A second design follows only if WNS still misses 40 ns.

## Pre-synthesis gates

- lint;
- the 95-sweep against P1's pwsw6: values and RVFI traces identical, and cycles may move only by the cut;
- the table-namespace arms on P1 against the cut: every value identical;
- the forced mutants: CUT_MUT_STALE must be caught by those arms, and CUT_MUT_LIVE is predicted not caught (the
  invariant).

The sweep enters the table namespace almost nowhere, so the arms, not the sweep, are the pairing gate.

**Lint on 5155e8e02 (read 2026-11-21):** PASS. Every counter equals P1's: LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0,
UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0. The 39 UNOPTFLAT
names are identical to the bitstream-1 tree's. Lint counts constructs and cannot see path depth, so this says the cut
adds no hazard; it says nothing about what the cut buys.

**Pairing gate (read 2026-11-21, S12_MEM_DELAY 12; P1 side = the bitstream-1 tree 9e801ab5e, whose ex_stage is P1's):**
12 table-namespace arms (va-alloc-csret, va-alloc, va-runv, va-srevroot, va-flat, vp3-idx18-a1/a2/a3, va-runv-r59,
r12-recl-stale, va-csmint-emu, va-fetchsplit-s):

| tree | values against P1 | end |
|---|---|---|
| cut 5155e8e02 | 12 of 12 identical | all SUCCESS; cycles move by -77 to +12 per arm |
| CUT_MUT_LIVE (b5c955856) | 12 of 12 identical, cycles equal to the cut's | unobservable, as predicted: the invariant held in every arm |
| CUT_MUT_STALE (b5c955856) | 11 of 12 differ | those 11 end FAILED at the 2 M-cycle limit; r12-recl-stale keeps its values, 6,743 -> 72,015 cycles |

So the arms catch a real mispairing, and the cut's pass is a reading.

**95-sweep on 5155e8e02 against P1's pwsw6 (read 2026-11-21, dirty=no):**
- RVFI traces identical on 92 of 92, with 26 parked (prefix only);
- values and readings identical on 91;
- cycles moved on 2 tests by +1 each (s07-ldc-overlap-displace 6,719 -> 6,720, s07-ldc-chain-forward 11,378 -> 11,379),
  the cut's one cycle per record read;
- the one "differ" is r30-fill-init's retired count, 997,110 -> 997,109. It is a loop parked at the 2 M-cycle limit, and
  its RVFI prefix is identical.

As pre-registered: only cycles move.

## Keys for the synthesis-only gate (P1's gate as reference: core 162,403 LUT, mapping 5,041.7 MB)

| key | prediction | if missed |
|---|---|---|
| mapping peak | within 10 % of P1's 5,041.7 MB; run peak inside the 40 GB guard | guard kill: partition read, no retry |
| loops | 0, as P1 | members named |
| core LUT | P1 +- 2,500 (the redistribution floor); the change adds about 200 flops and removes no logic | per-module census |
| core FF | P1 + 150 to + 400 (the registered response, index and flag; retiming may move flops) | census |

## Keys for the full build (P1's full build: LUT 187,368, FF 76,543, WNS -13.173, congestion 4)

| key | prediction | if missed |
|---|---|---|
| placed LUT / FF | P1 +- 2,500 / P1 + 0 to + 1,000 | census |
| routed | 0 errors (ROUTING-FAILURE BRANCH: placed area and congestion only, no WNS, not flashable) | |
| congestion | <= 5 | |
| loops | 0 | |
| **WNS** | **about -4.6 ns, inside [-8.3, -0.9]**: the next family on P1's placement with the whole rev_node hierarchy disabled (cur_idx_q -> lsu_bypass mem_q CE, -4.631, 56 levels), widened by the 3.74 ns null band. Both sides of the cut read positive on P1's placement (+14.7 before, +11.7 after) | inside P1's null band [-16.913, -9.433]: the cut bought nothing measurable and the critical cone is elsewhere in the unit; worst-path families on the routed checkpoint before any flash |
| worst-500 paths through `rv_rsp_*` / the crossing | 0 paths crossing the old rd_data pin combinationally | named |

Limits stated in advance: the next-family number comes from disabling the whole hierarchy, which is wider than this
cut, since the unit's internal paths and the post-cut side remain; retiming makes hierarchy approximate; and a new
placement moves everything. A WNS inside [-8.3, -0.9] supports "the cut moved the critical path out of the rev_node
cone"; it does not say by how much on another placement. The clock stays at 40 ns, and any gain is reported as slack.

## The synthesis-only gate, read (board lane, board/meas-pw 4d64c19cb53b), and a RE-KEY of both FF keys

Readings against P1's gate: mapping peak 5,050.2 MB (+8.5), loops 0, core LUT +471: PASS. **Core FF +101 against the
key's +150 to +400: the key FAILED by its letter.** Module moves, ungraded: ex_stage +5,039 LUT / +106 FF, dyn_unit
+7,421, issue_stage -4,218, lsu_i -774, rev_node -1,094, netting +471.

**The FF key was mis-sized.** Its lower edge was the cut's own bit count with no allowance for redistribution: 155 bits
(data_rvalid 1, data_rdata 64, data_ruser 64 since the user width is XLEN, the index 25, the namespace flag 1; data_rid,
data_rtag and the grant are not read through the register). It is a band of 250 flops on a design-level count whose
null redistribution this lane keys at +-2,000 everywhere else (the spatial gates, every full build). That is the
LUT-floor error of 2026-10-22 in flop units. The full-build FF key below (+0 to +1,000) has the same fault.

What the FF key was meant to show is that the cut's registers exist. The replacement keys are written here AFTER the
gate reading and BEFORE any full launch, and say so:

| key | prediction | if missed |
|---|---|---|
| the cut's own flops in the gate checkpoint, by name (`rv_rsp_q_reg*`, `rv_rsp_idx_q_reg*`, `rv_rsp_vctx_q_reg`, and any retimed or merged variant of those names, counted separately) | 155, less any bit the D$ drives constant, each such bit named from source | fewer than 150 unexplained: no full build; the missing bits named |
| gate core FF | P1 +- 2,000 | census |
| full build placed FF (replaces "+0 to +1,000") | P1 +- 2,000 | census |

Every other key above stands unchanged.
