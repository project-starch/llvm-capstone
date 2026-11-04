# Adaptive fault-ahead on silicon (board 2, ObjBounds): pre-registration

This file is written and pushed before the boots.

**The change.** The external collaborator's adaptive fault-ahead is ported to the vp2 board module.
- Source: runtime/fewer-exits 1c6fedf5d71f, on dev in a476237c079b.
- Port: lane perf/fault-ahead-vp2 a6fba52f75c3, on top of runtime/bench-hpm 6f7356368bb1, the source of the perf kits' module.
- **What it does.** Each context keeps four sequential fault streams. After `fault_ahead_after` (2) faults in a row, each further fault pins a window ahead of it: 4 pages, then doubling, up to `fault_ahead_max`. The window code is the existing `pin_ahead()`.
- **The port's differences from upstream:**
  - only where the vp2 module differs (no rx_bytes offset, no `a->windows`);
  - the default is off: `fault_ahead_max = 0`, a 0644 parameter. It is switched on through the kit's `adapter/module/params`, written through sysfs after insmod and read back (`@@PARAM`).
- **The module.** Board build `6e854c46c654c843`, by the perf kits' recipe (6.4.14, flat table, `-DCAPSTONE_VM_BENCH_COUNTERS`, rv64g, no 2-byte instructions). QEMU build `1d408b099741bab8`.
- **Only the module changes.** The launcher, images and bitstream are the ones the Spatial baselines ran with.
- The collaborator's runtime half (fewer transport exits, ABI 7) is NOT in this test. It needs new images.

**Kits.** Both are bkit15-objb-b2 with the module replaced:
- `bkit18-faoff-objb-b2`: no params;
- `bkit18-fa-objb-b2`: `fault_ahead_max=64`.

The scripts are copies, kept in `perf/fa/board/`:
- `spec-board.sh` takes the module's hash from `SB_MODULE`;
- `chain-board.sh` is unchanged.

**QEMU gate.** lbm test, off and on, on kit-hpmq-base with the module replaced. Its readings and the other five benchmarks' fault counts with fault-ahead on are given in Addendum 1, before the ON arm.

**Board order** (`fa-b2.sh`):
1. Board 2's chain stops at a sentinel after the outside-time `times` stage.
2. **OFF arm:** imagick test, module present, fault-ahead off.
3. **ON arm:** imagick, gcc, nab, lbm, xz, mcf test, fault-ahead on. It runs only if the OFF arm reads as the baseline.
4. Board 2's chain resumes: the mcf evaluation pair, then train.

The baselines are out-objb's runs of the same stages.

**Predictions** (`fa-grade.py`; controls: the baselines given as the ON arm read rc 1, a mock with faults /60 and outside −500 s reads rc 0, no data reads rc 2):

| # | Arm | Prediction |
|---|---|---|
| F1 | both | The control passes. Every arm is rc 0 and specdiff PASS, its outputs equal the twin's, and instret is within 0.1 % of the baseline. The read-back is exactly `fault_ahead_max=64` on ON and nothing on OFF. |
| F2 | OFF | In-kernel faults equal the baseline's exactly (imagick 815). |
| F3 | OFF | Run time is within 5 % of the baseline. |
| F4 | ON | lbm takes at least 10x fewer faults than its baseline (104,719). The collaborator's QEMU reading for that is 104,725 → 1,660. Every other stage takes no more faults than its baseline. |
| F5 | ON | Outside time (real − in-context cycles / 25 MHz) is lower than the baseline on lbm (631.5 s), mcf (3,459 s) and xz (6,424 s). It is reported on the other three. |
| F6 | ON | In-context cycles are within 3 % of the baseline: the program is unchanged. |

**What would refute what.**
- **F1 or F2 fails on OFF.** The ported module changes behaviour while off. The ON arm does not run.
- **F4 holds but F5 fails.** Faults are not what the outside time pays for. The per-exit cost is something else, which the outside-time `times` boot reads.
- **F6 fails.** Pinning ahead changes in-context cost, for example through TLB or cache state.

**Limits.**
- N = 1.
- Board 2 only. Spatial images only.
- Test inputs, whose faults are mostly first touches.

## Addendum 1 (4 Nov 13:20, before any board boot): the QEMU gate

The run is lbm test, both arms of `stages-test-hfl/519.lbm_r-test`, on kit-hpmq-base with the QEMU build of the ported module (1d408b099741bab8).

| | OFF (no params) | ON (`fault_ahead_max=64`, read back once) | census-q34 twin |
|---|---|---|---|
| faults | 104,719 | 1,646 | 104,719 |
| CSRUNV entries | 106,839 | 3,766 | 106,840 |
| in-context instret, Spatial arm | 7,278,940,280 | 7,277,599,059 (−0.018 %) | |
| rc, specdiff | 0, PASS | 0, PASS | |

- **Outputs.** The real outputs (`lbm.out`, `100_100_130_cf_a.of`) are hash-equal across OFF, ON and the twin. Only `lbm.err` differs, and it differs between OFF and the twin as well: it carries the module's stats line.
- **What this means for the board.** OFF reproduces the twin's fault path exactly. ON cuts lbm's faults 63.6x and its entries 28x. F4's 10x for lbm stands.
- The other five benchmarks' QEMU fault counts with fault-ahead on are running and will be reported in Addendum 2. They are not predictions.
