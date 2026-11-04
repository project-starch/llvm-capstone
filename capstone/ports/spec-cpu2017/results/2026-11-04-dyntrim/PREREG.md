# Stage A, the DYN-unit handshake trim, on board 2: pre-registration

This file is written and pushed before the flash.

**The change.**
- **Commit.** capstone-ariane c5a-dyntrim c9c7952bd, on top of C.5a (core 1fc2cd5aa). rtl record: llvm-capstone land/rtl-dyntrim, `docs/plans/2026-11-04-dyn-handshake-trim.md`; synthesis-only gate 9fd186b15046.
- **What it does.**
  - The DYN unit's ready is taken from its rtr in the same cycle.
  - An LDC no longer claims or writes rs1. It always handed rs1 back unchanged.
  - The rest is C.5a.
- **The seal (synth lane).**
  - Placed LUT 169,642 (C.5a 169,738), FF 63,646, slices 93.84 % (C.5a 95.57 %).
  - Congestion level 4 (C.5a 5).
  - 0 failed or unrouted nets, 0 loops.
  - WNS −13.869 ns (C.5a −13.758), shown for information. The lead has lifted the negative-WNS rule.
  - .bit sha256 `790ac2c096a0cd7361129572713333410745a4bc3158c581d7a9c25a5c1e97f2`. It was pulled from the synth tarball and checked locally.
- **The flash is authorised** by the lead's 4 Nov mandate: RTL updates, synthesis and flashing without asking.

**Board and order** (`dt-b2.sh`). The run starts after board 2's C.5a session (c5a-spec-b2.sh) ends with DONE and board 2 is back on bitstream 3. Same board, so stage A against C.5a is a same-board comparison.
1. Flash `caplifive_vp2c_c9c7952bd` and read the resident name back.
2. **R-67 kit**, `r67-grade.py <run> fixed --bitstream c9c7952bd`. Stage A keeps C.5a's fix, so FIXED is predicted. Anything else stops the run, and the reading goes to the lead.
3. **hpm kit**, `hpm-grade.py <run> dyntrim perf/c5a-b2/hpm` (land/rtl-c5a-kits 210768baa03d). The rtl lane's pre-registration:
   - hpm-3's stall falls below 10 % of C.5a's (sim 511 → 2), and its cycles drop 0.5-1.5 per LDC.
   - Everything else is equal within 1 %.
   - If C.5a's open l1d_miss +1 reappears, the verdict is FINDING. That is the same open item.
4. **slot kit**, `slot-grade.py <run> dyntrim` (land/rtl-slot-kit 98ff08e6f3fb).
   - P1d: the same doubleword ≈ the page alias. The other doubleword ≈ the other line, and is more than 5 % below.
   - All in the ±10 % sim band. sim: pair to another doubleword about 6.4 per op over the loop; same doubleword unchanged.
5. **ObjBounds kit**, `objb-grade.py <run> dyntrim perf/c5a-b2/objb`.
   - Exact images equal C.5a's, and objb-pcc does not return.
   - Against C.5a: s2, s4 (both passes), s6 and s7 lower; s1 and s5 pass 1 equal.
   - sim ratios: s2 0.804, s6 0.805, s7 0.841, s4 pass 2 0.869, s4 pass 1 0.957.
6. **SPEC test.** It runs on kit bkit19-perf-dt-b2: bkit12-perf-b3-b2 with `BOARD_BITSTREAM=caplifive_vp2c_c9c7952bd`, launcher and module checked by hash.
   - The stages are C.5a's list: gcc, imagick, nab, lbm, xz, mcf test, the 3 Nov images. Output into out-dt-b2.
   - It runs only if all of these hold:
     - R-67 is rc 0;
     - ObjBounds is rc 0;
     - the hpm grade reads neither WRONG BITSTREAM nor NO DATA;
     - no image hit a HARD STOP.
7. Flash bitstream 3 back.

**SPEC predictions** (`dt-spec-grade.py`, against the same stage of out-c5a-b2: same images, same board):

| # | Prediction |
|---|---|
| DA1 | The control passes. Every arm is rc 0 and specdiff PASS, its outputs equal the twin's, and its bench line is fresh. instret is within 0.1 % of C.5a's. |
| DA2 | On mcf, xz and gcc, stage A takes at least 0.2 % fewer in-context cycles than C.5a, and no more than 5 % fewer. Their LDC+STC are 2-3.4 % of cycles at one cycle each, and the trim takes about one cycle per LDC plus about two per other-slot pair. nab, imagick and lbm are reported: their expected drop is under 1 %, near the run-to-run spread. |

**Grader controls.**
- C.5a's own runs given as stage A read DA2 FAILS, rc 1.
- A mock of C.5a × 0.98 reads holds, rc 0.
- No data reads rc 2.
- The bitstream 3 runs given as stage A read rc 1.

**What would refute what.**
- **DA1 fails.** Stage A runs these programs differently. No cycle verdict stands, and the reading goes to the rtl lane.
- **DA2 fails with ratio ≥ 0.998.** The trim does not reach compiled code: the compiled LDC/STC are not back to back, or something else gates them. The slot kit then separates "the trim did not reach silicon" from "the trim does not reach compiled code".

**Dry run.** perf/dt-b2/dryrun has six branches:
- A-ok: success;
- B-r67: R-67 not FIXED;
- C-hard: HARD STOP in the ObjBounds kit;
- D-objb: ObjBounds not rc 0;
- E-notdone: the C.5a session not done;
- F-defaults: the real paths' preconditions.

**Limits.** N = 1 per arm. Test inputs only.
