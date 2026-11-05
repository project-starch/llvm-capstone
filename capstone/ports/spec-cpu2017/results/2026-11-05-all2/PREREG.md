# Everything together plus the five outside-time switches, board 1: pre-registration

Written and pushed before the flash (5 Nov, 08:40). Script `all2-b1.sh`, grader `all2-grade.py`, lists `gcc.list` and
`rest.list`, all in this folder.

## The run

**Configuration.** The all-together run (results/2026-11-04-all) with the five outside-time switches added. Those
switches were measured one at a time on 5 Nov (results/2026-11-05-outside-levers, RESULT b26e4bc3ad22).
- **Images and bitstream, unchanged.** The compiler evaluation's devmk images (dev compiler + movc flag) on C.5a
  (caplifive_vp2c_1fc2cd5aa).
- **Kit** t3-smoke/bkit23-all2-b1. It is bkit20-all-b1 with three changes:
  - module 2f81e8a5842010b3 (wt-fa perf/retire-sweep-vp2 2188cab26d40), params `fault_ahead_max=64
    retire_exit_sweep=3 image_rootfs=1 step_budget_ns=80000000`;
  - launcher 0bde5fd8e0c9c374, the exact image grant (perf/exact-image-grant, `-DCAPSTONE_VEXEC_EXACT_IMAGE`);
  - firmware `-DCAPSTONE_BOOT_QUANTUM=8000000 -DCAPSTONE_SUPERVISOR_QUANTUM=8000000`, a 320 ms quantum.
- **Reference: out-all-b1.** It is the same board, the same bitstream and the same images. The only difference is
  the module, the launcher, the params and the quantum.

**QEMU gate, already met.** modrs/qeg-{gcc,imagick,nab,lbm,xz,mcf} runs this module and this launcher with every
module switch on: six of six, rc 0, specdiff PASS (outside-levers PREREG, Addendum 2). The firmware quantum cannot be
gated on QEMU. On silicon it ran nab on bitstream 3 with X0 holding.

**Order** (`all2-b1.sh`):
1. Wait for all-b1.sh (pid 4077108) to end with DONE in its own run. That leaves board 1 back on bitstream 3.
2. Recheck the QEMU gate, then flash C.5a (sealed sha256) and read it back.
3. **gcc alone, the canary.** rootfs and the exact grant have never run together on silicon. `all2-grade.py --canary`
   must read Y0 holds, or the run stops after gcc.
4. The other five stages: imagick, nab, lbm, xz, mcf.
5. Grade, then flash bitstream 3 back.

## Predictions (`all2-grade.py`, every arm against the same arm of out-all-b1)

| # | Prediction | Point estimate |
|---|---|---|
| Y0 | rc 0, specdiff PASS, outputs = twin, control passes, bench line fresh; the four parameters read back; QNT0 007A1200; instret within 0.1 % | |
| Y1 | Run time below the reference, every arm. | |
| Y2 | gcc: launch ≤ 4 s (16.83) and run ≤ 30 s (41.1). | launch about 2.2 s, run about 26 s. The memfd copy goes; the devmk grant already pins only 2,220 pages. |
| Y3 | imagick: launch ≤ 4.7 s over both commands (6.66), run at least 2 s lower, each arm. | −3 to −6 s |
| Y4 | nab: steps ≤ 4,500 (11,279), run at least 25 s lower, each arm. | about 3,350 steps; −39 s: 7,900 fewer quantum exits at 3.56 ms outside and about 1.4 ms inside |
| Y5 | lbm: steps ≤ 8,000 (23,316), run at least 40 s lower, each arm. | about 6,800 steps; about −110 s: 16,500 fewer exits, sweep 3 on about 105 k retired pages, about 4,000 fewer faults |
| Y6 | xz, mcf: run ≤ 0.98 × the reference. | xz about −6 %, mcf about −4 % |
| Y7 | Inside cycles 0.97-1.005 × the reference, every arm. | −1 to −2 % from fewer resumes |

**Reported.** Run time against the 3 Nov baseline, and against Plain test, beside the all-b1 and 3 Nov ratios.

**What would refute what.**
- **Y0 fails on gcc (the canary).** rootfs and the exact grant together change what the program runs or reads. The run
  stops, and no other stage runs on this kit.
- **Y1 fails on a stage with Y0 holding.** The switches do not add on that stage. The cause is read from the stage's
  own launch, steps and teardown.
- **Y4 or Y5 fails on steps.** The quantum did not reach C.5a's firmware.

**Grader controls** (perf/all2-b1, 5 Nov):
- out-all-b1's own runs given as the run read rc 1. Y0 fails on the params and QNT0, and Y1, Y2, Y3, Y4 and Y5 fail.
- A mock with the point estimates reads rc 0.
- An empty folder reads rc 2.
- `--canary` reads rc 0 on the mock's gcc and rc 1 on all-b1's.
- Y6 and Y7 fire both ways on xz and mcf mocks built from out-fa-b1: 0.97 × run holds; 0.99 × run and cycles × 1.01
  fail.

**Dry run** (perf/all2-b1/dryrun, every board call stubbed, the real kit, bitstreams, QEMU gate and pre-registration
check):
- A-ok and I2-wait (waiting on a live pid): DONE.
- B-canary: stops after gcc, then flashes back.
- C-notdone, and C2-olddone (a DONE that belongs to an earlier run): STOP.
- D-qgate, E-flashfail, G-noprereg, H-busy, I-wait (DONE of another pid): STOP.
- F-defaults, the real grader: the canary reads rc 2 on no data and stops after gcc.

**Limits.**
- N = 1 per arm. Test inputs only, which overweight launch and teardown.
- The outside-levers RESULT's safety notes apply. The quantum masks interrupts for 320 ms, against 80 ms today.
  `image_rootfs` is safe by the module's own one-inode-per-file criterion. A claim-auditor run on 5 Nov tests that
  claim. If it refutes it, the run is stopped before the flash, and an addendum records why.
