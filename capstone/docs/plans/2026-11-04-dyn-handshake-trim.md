# DYN handshake trim (stage A): record and synthesis pre-registration

*RTL lane, 2026-11-04. Written before the synthesis-only gate runs. The readings are added under it when they land.*

## The change

capstone-ariane `c9c7952bd` (lane branch `c5a-dyntrim`, local to the RTL host, NOT on origin). It sits on `08c9fbd22`,
whose core equals C.5a's `1fc2cd5aa`. The branch is not pushed because it carries C.5a commits that exist only on the
external collaborator's remote; the synth lane fetches it from the RTL host. It changes three files:

- `ex_stage.sv`. `capstone_dyn_ready_o` now follows `capstone_dyn_ready` in the same cycle, instead of the registered
  `capstone_dyn_ready_q`. `capstone_dyn_ready` is the unit's rtr, an OR of its thread flops. One op in flight still holds:
  `!capstone_dyn_valid_i` gates the dispatch cycle, and rtr stays low until the unit loops back.
- `issue_read_operands.sv` and `commit_stage.sv`. An LDC hands rs1 back unchanged, so it no longer:
  - sets `gpr_clobber_vld` on rs1;
  - has a `stall_waw_rs1`;
  - has its rs1 capability lane written at commit.
- The R-27 (b) comment is corrected. An LDC/STC CAN dispatch in a flush cycle: 8 instances, in the C.5a record
  (`2026-11-03-c5a-bitstream.md`). It is harmless because the DYN unit's flush endpoint drops the op in the same cycle.
- A sim-only `ISSUE_HOLD_TRACE` block in each of ex_stage and issue_read_operands.

**Why two changes.** The two are co-binding: each alone measured zero on `stc-ldc-slot`. The sim-only trace shows the
reason. The registered ready held the DYN op for a cycle, and the LDC's claim on its base register held the next
capability op until the LDC committed.

## Readings (simulation, memory delay 12)

Cycles per iteration of 512, against C.5a.

| arm | C.5a | stage A |
|---|---|---|
| ldc alone | 5.31 | 4.32 |
| stc; ldc through one base, other doubleword or line | 11.46 | 9.47 |
| stc; ldc same doubleword (the compiled spill/reload shape) | 11.46 | 11.46 |
| stc alone | 6.54 | 6.54 |
| sd; ld same / other doubleword, bare loop | 9.44 / 5.47 / 3.10 | unchanged |

- **The same-doubleword pair does not move.** The ldc then waits on the store's page offset (WAIT_PAGE_OFFSET), and
  the store buffer has no forwarding.
- **STC→STC does not move.** It is gated by the rs1-lane WAW release one cycle after commit, which this change leaves
  alone.

## Gates passed before synthesis

- **`rtl-lint-gate.sh`: PASS.**
  - LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1430,
    CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0.
  - Every counter equals C.5a's except UNUSEDSIGNAL 1429 → 1430, which is `capstone_dyn_ready_q`, now read only by sim
    traces.
  - Lint ran on the tree before the comment-only edits.
- **The 95-test sweep against C.5a's sweep.**
  - SWEEP-RAN 95, with 92 logs; the 3 missing are `rand` and `perf_*_random`, missing on both sides.
  - Values are identical.
  - RVFI traces are identical in 91 of 92.
    - `cap-tag-cache` differs only in registers derived from mcycle. It is parked on both sides.
    - `s07-ldc-chain-forward` is parked on both sides and retires +6 in the same budget, over an identical prefix.
  - Cycles: 11 tests are faster (`stc-counter-pair` −3,640) and none is slower.
- **claim-auditor: PLAUSIBLE-BUT-UNPROVEN.**
  - **What it rests on.** The old LDC rs1 rewrite re-canonicalised SEALEDRET's unused bits, so equivalence needs every
    tagged register to be canonical already. No producer of a non-canonical tagged register exists: tags are set only
    by capability stores of registers that are already canonical. So a directed test cannot be built, and this
    argument is the record.
  - **Synthesis risk.** `capstone_dyn_ready` now reaches issue's `fu_busy`, which is the `issue_pointer` loop cone, and
    `gpr_clobber_vld` is on the standing UNOPTFLAT list. The lint gate's loop names are unchanged. Hence a
    synthesis-only gate before any build.

## Synthesis-only gate: pre-registered keys

The reference for every key is C.5a's own synthesis-only checkpoint of `1fc2cd5aa`, the same flow and settings.

1. **Mapping.** The peak at the technology-mapping marker is in class, 4.8–5.6 GB (C.5a 5,082.0 MB). No single
   process exceeds 9 GB.
2. **Loops.** The checks, each with its matcher's control firing:
   - "Found timing loop": 0.
   - Synth 8-326: 0.
   - check_timing: 0 loops.
3. **The change reached synthesis.** Read from synth's tree and checkpoint:
   - synth's `ex_stage.sv` contains `DYN handshake trim (2026-11-04)`;
   - `capstone_dyn_ready_q` flop cells read 0 on stage A, because the register is now unconsumed and is pruned;
   - the same name reads ≥ 1 on C.5a's checkpoint. If C.5a also reads 0, this key gives NO READING, not a pass.
4. **Stems** (own cells against C.5a):
   - each of ex_stage, issue_read_operands, scoreboard and commit_stage: LUT within ±2,500, the per-module
     redistribution floor;
   - ex+issue summed: LUT within ±2,500;
   - commit_stage: FF within ±5;
   - core LUT within ±2,000 of 145,391;
   - LUTRAM: 870 ± 50;
   - core FF is reported, not graded.
   - The source predicts tens of LUTs: one compare-and-gate per LDC term, and one fewer flop on ready.
5. **Per-module delta.** Every module with |ΔLUT| > 500 against C.5a is listed with its LUT and FF. This is reported,
   not graded, because re-homing is expected.
6. **Timing.** A synthesis-only run has no routed WNS. A post-synthesis timing estimate is reported if synth reads one.
   It is not graded and is not a WNS.

**Failure branches.**
- A mapping blowup (any process > 9 GB, or the 34 GB watchdog) is graded by genome and Synth 8-326 targets. There is no
  retry.
- A new "Found timing loop" header, or any member not in C.5a's union, fails the gate. It goes back to the RTL lane
  before any full build.

**After the gate.**
- The full build waits for the slot-kit silicon readings, per the SPEC lane's ruling of 2026-11-04.
- If those readings point to store-buffer forwarding, forwarding joins this bitstream. It comes after a design note
  and its own synthesis-only gate.
- Otherwise the full build of stage A goes once this gate passes.
- Full-build keys, written now: placed LUT within 169,738 ± 2,000; no placer or router window above level 5; routing
  with 0 failed nets; check_timing 0 loops; TIMING-23 0; LUTLP-1 0; WNS reported against C.5a's −13.758 (the null band
  is several ns); .bit sha256. A routing failure gives placed area and a congestion census, and NO WNS.

## Considered and left out

**The pcc_hold-from-flops lever** (SPEC lane's ruling, 2026-11-04). The −11.5 ns pcc family is 2.2 ns better than
C.5a's WNS. The WNS is bitstream 3's own `cpmp_q` → `fu_data_q` family, so the lever cannot move it. The from-flops form
costs a cycle per PC-capability change. The zero-cycle variant would take the hold off the type/perm/bounds priority
chain; it buys margin only, and that is not worth a re-proof in this bitstream.

## Synthesis-only gate: readings and grade (2026-11-04)

**The run.**
- The synth lane took c9c7952bd as a thin bundle on 1fc2cd5aa, which it verified by sha256 and `git bundle verify`.
- Scope it checked:
  - 1fc2cd5aa..08c9fbd22 touches only verif/ and capstone-docs/;
  - 08c9fbd22..c9c7952bd touches exactly the three core files (+47/−6);
  - 0 of 24 submodules and no build scripts changed.
- It stopped after synthesis, as planned: 1h16m, run peak 21.80 GB, largest single process 4.48 GB.

| Key | Reading (C.5a → stage A) | Grade |
|---|---|---|
| 1 | mapping peak 5,034.1 MB (C.5a 5,082.0) | PASS |
| 2 | "Found timing loop" 0, Synth 8-326 0 (control vp3g 7 / 73); check_timing 0 loops (control 1); no headers or members on either tree | PASS |
| 3 | marker present in synth's ex_stage.sv; `capstone_dyn_ready_q` flops 0 vs 1 on C.5a (readable); POS `pc_revnode_valid_q` 1 / 1 | PASS |
| 4 | ex_stage 47,956 → 47,008 (−948); issue_read_operands 28,686 → 26,079 (−2,607) = own cells 19,086 → 20,189 (+1,103) + cap metadata regfile 3,848 → 1,854 (−1,994); scoreboard 12,831 → 14,333 (+1,502); commit_stage LUT 203 → 104, FF 243 → 243; ex+issue −2,053; core LUT 145,179 (−212); LUTRAM 870; core FF 41,423 (−3, reported) | PASS on the key as written (own cells); see below |
| 5 | modules with \|ΔLUT\| > 500: dom_switcher 8,065 → 10,105 (+2,040, FF 1,157 = 1,157); csr_regfile_i 28,966 → 27,460 (−1,506); lsu_i 16,736 → 15,698 (−1,038; lsu_bypass −760); id_stage_i 958 → 1,521 (+563); issue_stage_i 41,517 → 40,412 (−1,105) | reported |
| 6 | report_timing_summary NA on both trees (check_timing: 54,293 no_clock endpoints; the synth checkpoint carries no usable clock) | NO READING, as allowed |

**Key 4, the one clause the synth lane read as failing.**
- Synth graded issue_read_operands by its hierarchical total, −2,607, which is 107 outside ±2,500.
- The key as written grades OWN cells, and issue_read_operands' own cells read +1,103. That is inside the band, so
  this is a PASS as written, not a correction made after the reading.
- The hierarchical drop sits in the capability-metadata regfile child (−1,994). That file is not in the change, so the
  drop is re-homing, as are dom_switcher +2,040 and csr_regfile_i −1,506. Neither of those files is in the change
  either. The core total moves −212.
- The other three rows (ex_stage, scoreboard, commit_stage) were read as hierarchical totals and are inside the band.
  Their own-cell rows were not read; a confirming reading is requested, and it does not gate.
  **Confirming reading (synth, own cells, C.5a → A, LUT / FF):** ex_stage_i 2,255 → 1,912 (−343) / 836 → 832;
  scoreboard 12,831 → 14,333 (+1,502) / 7,080 → 7,097 (no child rows); commit_stage_i 197 → 98 (−99) / 236 → 236;
  issue_read_operands 19,086 → 20,189 (+1,103) / 658 → 674. Every own-cell row is inside ±2,500: key 4 CONFIRMED.
- Gate ruling: PASS. The full build stays HELD for the slot-kit silicon readings, per the SPEC lane's ruling.

## Full build: released alone (2026-11-04)

**The decision.** The SPEC lane released the hold: stage A builds alone, and is not batched with forwarding.
- Forwarding still needs a design note, sim, lint, an auditor and its own synthesis-only gate, a day or more.
- Stage A is gated and small (core −212 LUT).
- So it is built now, to be ready to flash. If forwarding goes ahead, it becomes its own bitstream on top of stage A.
- This departs from the batching default deliberately, and the cost is written here: a second build cycle if
  forwarding follows.

**Keys.** The full-build keys above stand as written. They are graded against C.5a's seal of 1fc2cd5aa:
- placed LUT within 169,738 ± 2,000;
- no placer or router window above level 5;
- 0 failed, unrouted or partially routed nets;
- check_timing 0 loops, TIMING-23 0, LUTLP-1 0, each with its control;
- WNS reported against −13.758;
- .bit sha256.

A routing failure gives the placed area and a congestion census, and NO WNS.

**Predicted first-boot reading** (silicon; the sim figures are at memory delay 12). On the slot kit, against C.5a's
slot run:
- ldc alone, and stc/ldc to another doubleword: lower;
- the same-doubleword pair, stc alone and sd/ld: unchanged;
- exact fields (instret, value, traps, cause): equal to every other bitstream's.

The sim deltas, ldc −1.0 and pair −2.0 per iteration, are the PATTERN being tested. They are not a silicon
prediction: sim cannot reproduce the silicon pair cost.
