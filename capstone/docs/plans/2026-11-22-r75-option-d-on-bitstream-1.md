# R-75 option (d) on bitstream 1: pre-registration of the synthesis-only gate and the full build (2026-11-22)

Written before any synthesis. **The lead ruled option (d)** on 2026-11-22, in the runtime lane's session (relayed by the
runtime lane; land/rtl-sublet-p2's B5b scope doc lists the options).

RTL: capstone-ariane `b1-r75d` 1c5d31474 = bitstream 1 9e801ab5e + the R-75 arms (aac6daa91 = 887dcb343's tests) +
option (d) (499390b66 = b7ea390a1's core diff, identical by content) + R75_EARLYCLR set in the config package
(7e7f1f1e5; `+define+R75_OFF` restores bitstream 1 on the same tree) + a SUP_TRACE firing print (1c5d31474, invisible to
synthesis and to lint).
- **What (d) does:** from the cycle the live lifetime namespace (vctx_table_i, srevroot_i) differs from the held copy
  until the flush that follows has cleared the LSU's ALLOW/DEAD records, no PC verdict is given and no PC probe starts.
  The 2026-10-20 ruling and the PC tracker are unchanged.
- **Where the new term goes:** a 1-bit and a 56-bit compare in ex_stage (r75_ns_pending), read in the LSU. It ends at
  the pcc_v_q flop and at the probe-pending register (load_store_unit.sv :1802); the probe request leaves the LSU from
  flops only (:1640), so the term does not reach the rev-node unit combinationally. It is in the LSU's verdict cone, the
  loop-risk class, hence the synthesis-only gate.

## Pre-synthesis gates (readings)

- **Lint** on 7e7f1f1e5 (the synthesised content of 1c5d31474): PASS, every counter equal to P1's: LATCH 44,
  MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2,
  SELRANGE 100, ANVIL_UNOPTFLAT 0.
- **The R-75 arms, two-sided on the same tree** (7e7f1f1e5, dirty=no, S12_MEM_DELAY 12):

| arm | (d) on | +define+R75_OFF (bitstream 1) |
|---|---|---|
| va-rc-trap | SUCCESS, 93,870 cycles, 0 handler entries | FAILED at the 2 M-cycle limit, 34,069 handler entries (R-75) |
| va-rc-trap-p | SUCCESS, 93,933 | FAILED at the limit, 272,541 handler entries |
| va-rc-trap-z | SUCCESS, 93,990 | SUCCESS, 93,990 |
| va-rc-trap-zp | SUCCESS, 94,094 | SUCCESS, 94,094 |
| va-fetchsplit-s | SUCCESS, 83,106 | SUCCESS, 83,106 |

  The (d)-on cycles equal the P1 prototype's (b7ea390a1: 93,870 and 93,933).
- **Running:** the same arms with the firing print (their R75-BLK / R75-HOLD counts are the print's positive control),
  then the 95-sweep against bitstream 1's swb1, with the firing count. Prediction: values and RVFI identical on all 92;
  the firing count is reported as read. The full build waits for the sweep.

## Synthesis-only gate keys (reference: P1's gate, core 162,403 LUT, mapping 5,041.7 MB; bitstream 1 adds 9 csr lines)

| key | prediction | if missed |
|---|---|---|
| mapping peak | within 10 % of 5,041.7 MB; run peak inside the 40 GB guard | guard kill: partition read, no retry |
| loops | 0 by every probe, as P1 | members named |
| core LUT / FF | P1's gate +- 2,500 / +- 2,000 | per-module census |
| the term's own cells | reported (a 57-bit compare and two gates, tens of LUT) | |

## Full-build keys (reference: bitstream 1's seal, 01707b37…: LUT 188,594, FF 76,492, WNS -14.595, congestion 5)

| key | prediction | if missed |
|---|---|---|
| placed LUT / FF | 188,594 +- 2,500 / 76,492 +- 2,000 | census |
| routed | 0 errors (ROUTING-FAILURE BRANCH: placed area and congestion only, no WNS, not flashable) | |
| congestion / loops | <= 5 / 0 | |
| WNS | inside bitstream 1's null band, -14.595 +- 3.74 = [-18.34, -10.86] | outside: worst-path families before any flash |
| worst-500 paths through r75_ns_pending or the srevroot compare | 0 (the term ends at flops) | named |

Silicon: the runtime lane's (the R-75 shape on the board, and a bitstream-1 regression set).
