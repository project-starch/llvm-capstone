# Every optimisation together, on board 1: pre-registration

This file is written and pushed before the boots.

**What runs together.**
1. **The compiler evaluation's devmk images:** dev compiler 724bf0bea11e plus `+movc-keeps-integer-source`, in the stages `stages-eval-b1/*-devmk`. Their twin is census-q34, and their outputs equal the 3 Nov images' outputs.
2. **C.5a**, caplifive_vp2c_1fc2cd5aa, sealed sha256 5b7b1edc...d764, flashed on board 1 and read back.
3. **The adaptive fault-ahead module.** This is the external collaborator's change, ported to vp2 (6e854c46c654c843, `fault_ahead_max=64`, read back through sysfs).

**Kit.** bkit20-all-b1 = bkit17-perf-c5a-b1 with the module and params file replaced. Board 1 runs it after its fault-ahead run and the counter boots (perf/fa/hpm-after-fa.sh, which must end with HA END). Stage list `all-b1.list` (gcc, imagick, nab, lbm, xz, mcf test), into out-all-b1. Then bitstream 3 is flashed back.

**References**, all on board 1:
- the 3 Nov baseline (spec/board/out, bitstream 3);
- the same devmk images on bitstream 3 with the stock module (out-hfl/*-devmk). For mcf only the dev-compiler step exists, out-hfl/...-dev, because its devmk run was stopped;
- Plain test (out-hfl/*-test-plain-rv64).

**Predictions** (`all-grade.py`; controls: the devmk-on-bitstream-3 runs given as this run read AL2 FAILS rc 1, no data reads rc 2):

| # | Prediction |
|---|---|
| AL1 | The control passes. Every arm is rc 0 and specdiff PASS, its outputs equal the twin's, and its bench line is fresh. instret is within 0.1 % of the devmk-on-bitstream-3 run (same images); not graded for mcf, whose reference is the dev step. |
| AL2 | Run time is below the devmk-on-bitstream-3 run's on every arm (mcf: below the dev step's). C.5a removes the lifetime check's cycles, and fault-ahead removes faults and exits. |

**Reported, not predicted:**
- run time and in-context cycles against the 3 Nov baseline;
- run time against Plain's, the slowdown, beside the 3 Nov slowdown.

On the devmk images alone, the slowdown already falls: gcc 6.82x → 5.32x, imagick 1.67x → 1.48x, nab 1.36x → 1.24x, xz 1.51x → 1.44x, lbm 1.11x → 1.08x.

**Limits.**
- N = 1.
- Board 1 only.
- These are the evaluation images. The paper's numbers stay on the 3 Nov images.

**Dry run** in perf/all-b1/dryrun:
- A-ok: success;
- B-haStop: the counter boots did not end;
- C-flashfail: the C.5a flash fails;
- D-defaults: the real paths' preconditions.

spec-board.sh (the perf/fa copy, with SB_MODULE) accepts kit bkit20-all-b1.
