# Spatial-only core (programme 7.1 item 2) — pre-registration of the full build (2026-11-20)

Written before the full build is launched.

## What is built

capstone-ariane `spatial-only-b` 1f1ed11f9 = P1 d83d3a633 + one define, `CAPSTONE_SPATIAL_ONLY`, set in the config package
(`+define+CAPSTONE_LIFETIME_ON` restores P1 on the same tree in simulation). Under it:
- step A (cf7a00ce6): `core/capstone_rev_node_stub.sv` replaces the rev-node unit with the same ports, every request
  answered "complete, no effect" from flops one cycle later (init/mrev/derive: status 0 and the requester's id; create:
  ancestor = child = rs1's id; query/rev/drop/delin 1; kill/deepen/retire 0; the probe always accepted; the memory
  channel idle); every verdict lookup reads live; the PC tracker never holds or denies;
- step B (1f1ed11f9): REVOKE, MREV and DROP are illegal instructions.
No structure is removed explicitly: step A's synthesis-only gate shows synthesis removes the dead lifetime logic itself.

## Which ops become illegal, and why only those (the 2026-11-20 census)

Executed on the twin (the nested lane; LDC the positive control), FRCDKMSspa3 images gcc train01 3d8bfb5a85a3 / mcf test
17b2fa0b796e / perlbench test cdd368de816c: CDERIVE 26,065 / 318 / 163,069, CREVOKE 22,441 / 276 / 46,242 (the SDK's
__heap_mapping / __heap_unmapping); REVOKE, TIGHTEN, DELIN, SPLIT, MREV, DROP, CALL, RETURN, CSSUPERVISE, CSRETIRE, CSNODE 0.
Static: board kernel (CSRUNV 8, CSCHECKR 1, CSCHECKW 2, CSKINDS 4, CSMINT 1, CSRETIRE 1), drivers (none), smoke domains
(CDERIVE, CREVOKE), monitor (DELIN, SPLIT, CSNODE, CALL, RETURN, CSSUPERVISE). Only REVOKE, MREV and DROP are zero in every
binary the +Spatial board stage runs.

**A term of its own:** the +Spatial images EXECUTE CDERIVE and CREVOKE, which P1 performs and this core makes no-ops, so a
reading on this core differs from +Spatial-on-P1 by those operations as well as by area and timing. A second variable on
the board: the monitor's CSCHECK emulation reads liveness from the table (`capstone_cscheck.S:196-255`), which this core
never writes, so the spatial arm boots the runtime lane's firmware variant with that block compiled out
(`-DCSC_SPATIAL_ONLY`, bounds and permission checks kept).

## Pre-synthesis gates (readings)

All S12_MEM_DELAY 12, against P1 d83d3a633's sweep (`sim14-wt/.sup/pwsw6`).
- **Lint** (`rtl-lint-gate.sh`, step A cf7a00ce6 and step B 1f1ed11f9 alike): PASS. LATCH 44, MULTIDRIVEN 3,
  ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0, all
  equal to P1. UNUSEDSIGNAL 1480 -> 1351 (-145 inside the generated rev-node unit, no longer instantiated; +16 on the
  stub's unread inputs). UNOPTFLAT keeps 39 names with one member renamed: the D$ read-port ack loop is now reported at
  port 0 instead of port 2, port 2 being the stub's idle record port.
- **Synthesis-only gate of step A** (board lane, P1's gate as reference): mapping 4,914 MB (P1 5,042), 0 loops by every
  probe, 2 [Synth 8-327] as on P1; core -7,774 LUT / -3,696 FF; ex_stage -11,239 (the stub -5,353, the LSU's own cells
  -2,474 incl. all 880 LUTRAM, csr_buffer -2,188 inside the redistribution floor); dyn_unit +1,722 and csr_regfile +1,420
  inside the floor. Guard RSS peak 32.0 GB (P1's gate 27.35), reported, unexplained. Step B adds three decode terms.
- **95-sweep, step A against P1:** RVFI 89 identical / 3 differ, all in the predicted may-differ set (data-sharing,
  excode-base-audit, r31-revoke-cursor); values 85 / 7, the extra four being r30-fill-init and s06sec-csr-raw-no-forge
  (predicted) and cpmp-su-mode and cpmp-if-check (NOT predicted: NOT_CAP loops parked at the 2 M-cycle cut, +3 traps
  each from the shorter reset; RVFI prefixes identical).
- **95-sweep, step B against step A:** 87 identical in values and RVFI; the 5 that differ (revocation, data-sharing,
  excode-base-audit, s07-wbuf-stall-corners, r31-revoke-cursor) all issue REVOKE or MREV, as predicted.
- **sp-illegal** (step B): 3 traps, causes 0x2 0x2 0x2, the ldc's value 0x5a5a, as pre-registered; with
  `+define+CAPSTONE_LIFETIME_ON` the same arm reads causes 26, 26, 25 (no cause 2).
- **Must still pass (measured):** the 85 tests identical to P1 in step A, plus cpmp-su-mode and cpmp-if-check by RVFI
  prefix; the lifetime-dependent set (revocation, data-sharing, excode-base-audit, s07-wbuf-stall-corners,
  r31-revoke-cursor, r30-fill-init, s06sec-csr-raw-no-forge) differs by design.

## Full-build keys

References: P1's full build (`pw-d83d3a633-full-out/grade-full.txt`: LUT 187,368, FF 76,543, slices 50,800, WNS
-13.173, congestion 4) and step A's synthesis-only gate (board/meas-pw 2f6d63a7e6eb: core -7,774 LUT, design 181,926
synthesised against P1's gate 189,693).

| key | prediction | if missed |
|---|---|---|
| run peak memory | within the 40 GB guard (step A's gate peaked at 32.0 GB, P1's 27.35: reported, unexplained) | guard kill -> the partition read, no retry on identical inputs |
| placed LUT | P1 - 7,800, within +-2,500: [177,068, 182,068] | per-module census against P1's routed hierarchy |
| placed FF | P1 - 3,700, within +-2,000: [70,843, 74,843] | as above |
| routed | 0 errors | ROUTING FAILURE BRANCH: placed area and congestion recorded, no WNS, not flashable |
| placed congestion | <= 5 | as routing |
| check_timing loops | 0 | worst-loop members named |
| WNS | reported, graded against P1's null band [-16.913, -9.433]; a value ABOVE the band (removal freeing placement) is reported as such, not as a failure | below the band -> worst-path families before any flash |

## Silicon keys (the runtime lane's board)

Tier 1 with the CSC_SPATIAL_ONLY firmware: its per-step expectations are written by the runtime lane before the boot
(revocation steps differ by design). The +Spatial images' cycles against +Spatial on P1 are the paper's measured row,
read with the term above.
