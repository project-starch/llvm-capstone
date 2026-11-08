# Pre-registration (DRAFT): the native route's first board boots on the new RTL

DRAFT. Every value marked PLACEHOLDER is filled in, and this file committed and pushed, BEFORE the first boot. The
launch is gated on that push. The kit refuses to run while `board.env` names a PLACEHOLDER.

## What is being asked

Does the native Linux-task route run on silicon? That route is the external collaborator's adapter built into the
kernel, Linux 6.4.14 with `linux-6.4/`, and the direct and exchange syscall arms. The new bitstream adds what C.5a lacks:
CSRUNV 4/5, the paged table, CPTAG/CLDRAW/CSTRAW and the checker.

## Platform

| Part | Value |
|---|---|
| Bitstream | PLACEHOLDER name, PLACEHOLDER sha256 (the rtl lane's sealed build; a hash with its lint numbers and a completed synthesis) |
| Firmware | PLACEHOLDER: which monitor and wrapper the native route boots on the new RTL. On QEMU it is plain OpenSBI `fw_dynamic`, and the model does the context switch. |
| Kernel | Linux 6.4.14 with this branch's `linux-6.4/` series (lane commit 252699a9cafc) and the board's `.config` plus the series' options. The kit relinks it on the bake's `rootfs.cpio` with the adapter parameters appended (CMDLINE_EXTEND). The per-boot Image hash is recorded in the kit's SEQ. |
| Adapter parameters | PLACEHOLDER. Proposed, the QEMU gate's own set (route trace off): `capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.native_fast=1 capstone_vm.dom_binfmt=1 capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 capstone_vm.supervisor_slots=1`. If the bitstream lacks transport or raw moves, use C.5a's model instead (`capstone_vm.c5a_model=1` in place of transport and unpinned). |
| Probes | The QEMU-qualified musl-IO probes (native01 build), byte-identical between the twin and the board stage. Whether their ABI holds on silicon has not been checked (see the open questions). |
| Kit | `bkit36-native64`, a `cp -rL` of `bkit30-unify`. It runs one rung, `vbatch`, whose controller is `ctl.sh` running `native-smoke.sh`. |

## Boot 1 (Tier 1): the native smoke, which RETURNS a code

`native-smoke.sh` runs these steps in this order. Each step prints `@@NS-VERDICT <step> PASS|FAIL <code>`, and the
run ends with `@@NS-RESULT <first failing code>` and `RESULT vbatch retval=<same>`. A step that never returns leaves
no `@@NS-RESULT`; `classify.py` then names the open step.

| Code | Step | Pass line (the same as `run-musl-io.py io`) |
|---|---|---|
| 4 / 5 | the adapter is present / the route is on | `@@NS-PARAM native_task=Y`, `dom_binfmt=Y`, `native_dispatch=Y` |
| 10 | plain control (`busybox true`, no capability code) | rc 0 |
| 20 / 21 | exchange arm, and through a pipe | `MUSL_IO checks=170 failures=0` / `checks=4` |
| 30 / 31 | direct arm, and through a pipe | `MUSL_IO checks=176 failures=0` / `checks=4` |
| 40 | `probe.dom` (128 KiB stdin) | `native integration checks=51 failures=0` |
| 50 | `gcc-coverage.dom` | `gcc coverage checks=20 failures=0` |
| 60 | `entry-driver blocked.dom`: starts a SECOND live native task ("PASS second native task ready") | `native entry failures=0` |
| 70 | `task-driver compute.dom blocked.dom`: two live native tasks | `native task failures=0` |

Steps 60 and 70 are last because they are the only steps expected to wait. They print straight to the console, because
a driver killed by its alarm loses block-buffered output written to a file. The board judges them on rc, and
`classify.py` also requires the pass line inside the step's span.

**Predicted readings, written before the boot:**

- **The full model works on the new RTL** (transport, the paged table and raw moves): every step PASS, `@@NS-RESULT 0`.
  The QEMU twin of this exact stage on the 6.4 QEMU kernel reads that (`twin/q64-full`, all 9 PASS).
- **The bitstream has no context transport**, or the C.5a parameters are used: steps 10-50 PASS. Steps 60 and 70 FAIL
  with rc 142: each driver ends on its own 30 s alarm, because the second native task waits for the one slot. The 6.4
  twin run of this stage on C.5a's model reads exactly that, `@@NS-RESULT 60` (`twin/q64-c5a`). The 6.1 gate's C.5a
  probe stopped at the same point. Its last line before `Alarm clock` is "PASS first native task ready", which is
  entry-driver's string (task-driver's binary does not contain it). The probe-1 note attributes that line to
  task-driver.
- Code 4 or 5 is a kernel or command-line fault, not an RTL one. Re-check the kit's SEQ line `native kernel ...`, and
  that `@@NS-CMDLINE` carries the parameters.
- A wedge in steps 20-60 (no `@@NS-RESULT`) names the first native step that fails on silicon. Compare it with the twin
  before writing any attribution.

## Boot 2: three short SPEC runs on the native direct arm (after boot 1 reads 0, or 70 with the cause understood)

Order: shortest and safest first, one benchmark per step, each returning its rc and `@@CNT` counters.

| # | Benchmark | QEMU reference |
|---|---|---|
| 1 | 519.lbm_r test | The collaborator's 6.1 platform: 10/10 SPEC test entries PASS, lbm and nab among them (`spec-final-qualification.json`). The checked-emu measurement: 7.95e9 instructions, rc 0. |
| 2 | 544.nab_r test | the same qualification |
| 3 | 502.gcc_r train01 | rc 0 on QEMU with the paged table (`out-gtr-dir-off`). **Flat-16 does not hold it:** rc 139 after 11.26e9 instructions (C.5a probe 2). |

Predicted: rc 0, and outputs that match the QEMU twin by hash (specdiff on the host). gcc train01 completes ONLY if the
bitstream's paged table is in use. On a flat table the prediction is probe 2's fault (`capstone_task: fault rc=-14
kind=2 cause=30`).

Staging: SPEC's own commands, from `make-stage.py`. Executables are stripped (`--strip`) to fit the 26,000,000 B
payload cap: gcc's unstripped `.dom` is 24 MB. Stage sizes are recorded before the bake.

## Gates before the launch

1. `preflight-board-run.sh` PASS.
2. The kit's deep dry run, kept beside this file: `BOARD_STUB=2` on the cpio seed and a recorded twin transcript. It
   must reach `BOARD_RUN_DONE` and produce all three verdicts: PASS on the full twin, FAIL 70 on the C.5a twin, and NO
   RESULT on a transcript cut inside a step.
3. Read every row of the new bitstream's open ISSUES entries against these images.
4. Check the bitstream sha256 locally. After the flash, read the resident name back. A known-good fallback stays
   registered.

## Open questions this draft cannot settle

- The firmware for the native route on silicon.
- Whether the QEMU-built probes, and the native SPEC images, carry the silicon ABI's requirements: delin sites, glue,
  NONLIN type numbering (the rtl lane's frame-layout review, difference 1).
- The adapter parameters for the new bitstream.
- On this kernel the stock process-ABI controls cannot run, so the boot has no published known-good control. The
  plain-control step is the only control.
