# The rev-node cone cut, second cut on P1: pre-registration (2026-11-22)

Written before any simulation of the RTL and before any synthesis. RTL: capstone-ariane `p1-revrd-cut2` 830c0d479 = cut 1
5155e8e02 + a one-entry buffer on the unit's record READ request (core/ex_stage.sv only). Cut 1's record and its KILL:
`2026-11-21-rev-node-cone-cut.md` (dev d9dde0a25034).

## Why

Cut 1 registered the record read response, and no worst path crosses that entry any more. WNS stayed inside P1's null
band, because all 500 worst paths cross the unit by a second entry: the D$ port-2 read GRANT, combinational into the
unit's events (board lane, cell walk on cut-5155e8e02-full; source: rv_rd_res.data_gnt = rv_rd_res_c.data_gnt into
`_mem_ch_read_req_ack`). The board lane's predictor on that routed checkpoint (read-only, arcs disabled in memory) with
the 18 read-grant pins disabled: worst -5.606, worst through rev_node +0.019, 0 of the worst 500 crossing rev_node. The
next limit is issue_read_operands' own operand path (fu_data_q cap_metadata_a -> cap_metadata_b, 101 levels, no rev_node
cell). Disabling the write grant as well changes nothing. **-5.606 is a bound, not a prediction**: the previous bound
missed by 5.5 ns.

## The design and why it is safe

The unit's read request fills a one-entry buffer; the unit's ack is the buffer's empty flag, a flop. The walker, the D$
request and the read-index latch see the buffered request under the old names, so the index is still latched at the
grant of the request it decodes (cut 1's pairing kept). Read from capstone_rev_node.anvil:
- the unit is single-outstanding: `get_rev_node` is `send read_req >> recv read_res`;
- it never presents a read and a write together: every one of the 49 memory calls is sequenced with `>>` (the one
  parenthesised group runs a register set beside a write, then `>>` a read), so a buffered read cannot overtake a write;
- its flush thread aborts nothing (`recv mem_ch.flush >> cycle 1`), so a buffered read is always answered to the
  request that made it, and the buffer is NOT cleared on flush;
- the walker's dead-record ack is gated by the (now buffered) request valid.
The grant now ends at flops (the buffer's valid and the index latch).

## Cost, stated before any reading

+1 cycle per record read, SERIAL with cut 1's +1: +2 in all. On the record-cache model's P1 counts, gcc train01 +0.078 %
and mcf test +0.25 % of P1's in-context cycles, both upper bounds. mcf is above the 0.2 % figure used against the
lsu_bypass cut. The gain is slack at a clock that stays at 40 ns. **Whether this cost is worth that slack is put to the
runtime lane before any synthesis** (2026-11-22); simulation proceeds meanwhile.

## Pre-synthesis gates, predicted

- Lint: every counter equal to cut 1's, except UNOPTFLAT, which may lose the D$ read-port ack member, because the
  buffer removes a combinational dependency on the grant. A change is named by member, and a loss of exactly that
  member is the instrument firing, not a defect.
- Cut 1's twelve pairing arms (va-alloc-csret, va-alloc, va-runv, va-srevroot, va-flat, vp3-idx18-a1/a2/a3, va-runv-r59,
  r12-recl-stale, va-csmint-emu, va-fetchsplit-s): every value equal to cut 1's and P1's; cycles equal to or above
  cut 1's (+1 per record read).
- Forced mutant CUT2_MUT_HOLD (the buffer keeps a read one cycle past its grant, so the walker issues it twice): caught
  by at least one of the twelve arms. If none catches it, the arms cannot see a duplicated read; that is reported, and
  a directed arm is written before any synthesis.
- The 95-sweep against P1's pwsw6: values and RVFI identical; cycles may move.

## Keys for synthesis, if the cost is accepted (written now, binding)

Synthesis-only gate against cut 1's gate (board lane): mapping within 10 % of cut 1's 5,050.2 MB inside the 40 GB guard;
loops 0; core LUT within +-2,500 and core FF within +-2,000 of cut 1's gate; the buffer's flops by name, rq_v_q 1 and
rq_addr_q up to 30 (merges reported).

Full build against cut 1's seal (LUT 187,801, FF 76,644, WNS -10.130):
| key | prediction | if missed |
|---|---|---|
| worst-500 paths crossing rev_node | 0 (the mechanistic claim) | named; the claim is refuted |
| WNS | better than P1's null-band edge -9.433, i.e. a measurable gain; reported against the -5.606 bound, expected on its pessimistic side | inside P1's band [-16.913, -9.433] again: the predictor method is refuted twice, and the cone programme on P1 stops |
| placed LUT / FF | cut 1 +- 2,500 / +- 2,000 | census |
| routed | 0 errors (ROUTING-FAILURE BRANCH: placed area and congestion only, no WNS, not flashable) | |
| congestion / loops | <= 5 / 0 | |

## The predictor for cut 2 as built (board lane, read-only on cut-5155e8e02-full, logs PRED4 / PRED5)

The first predictor's 18 "grant" pins included 2 write-side pins and 1 flop input (the board lane's correction). With
only the 15 read-side pins disabled, and the 2 write pins and the flop live, which is cut 2 as built: worst -5.606,
worst through rev_node +0.019 (entering at the unit's result selector, from issue_read_operands fu_data_q, not at a D$
input), 0 of the worst 500 crossing rev_node. With 10 read-grant pins disabled and the 5 mem_wait_flag-named pins live as
well: the same. Neither the write grant nor those 5 pins is needed. Still a bound on one placement; the WNS key above
stands as written.

## STOPPED before any simulation or synthesis (the runtime lane's answer on the cost, 2026-11-22)

The runtime lane's answer to the cost question: the cone programme on P1 stops here, and cut 1 stands as the
measurement.
- The board clock stays at 40 ns and no clock change is planned. At about -10 -> -6 ns timing still fails, so the slack
  buys nothing that would be used.
- +0.25 % on mcf (an upper bound) is above the 0.2 % the lsu_bypass cut was held to.
- Line B rebuilds this path, so the timing work goes into B5b's design.
The design stays parked on capstone-ariane p1-revrd-cut2 830c0d479: never simulated, never synthesised. Its gates were
stopped before they produced a reading (lint-cut2 killed mid-run, no output kept).

**Carried into B5b-1's design** (the finding, not the fix): after the response is registered, P1's worst paths enter
the lifetime unit at the D$ read grant, combinational into its event logic, and from there reach the DYN result
selector and the PC tracker. A design whose unit sees the memory port only through flops on both the request and the
response side removes it. With both, the next limit on this placement is issue_read_operands' own operand path (bound
-5.606).
