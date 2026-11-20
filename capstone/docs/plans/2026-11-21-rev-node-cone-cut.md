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

## Pre-synthesis gates

- lint;
- the 95-sweep against P1's pwsw6: values and RVFI traces identical, and cycles may move only by the cut;
- the table-namespace arms on P1 against the cut: every value identical;
- the forced mutants: CUT_MUT_STALE must be caught by those arms, and CUT_MUT_LIVE is predicted not caught (the
  invariant).

The sweep enters the table namespace almost nowhere, so the arms, not the sweep, are the pairing gate.

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
