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
| Kernel | Linux 6.4.14 with this branch's `linux-6.4/` series (lane commit d0a371f130e4). The `.config` is the board's stock `.config` plus exactly `native-config.delta`: RISCV_CAPSTONE_TASK, VIRT_CPU_ACCOUNTING_GEN in place of TICK_CPU_ACCOUNTING, and CONTEXT_TRACKING_USER(_FORCE). SMP=y, NR_CPUS=64 and TREE_RCU stay stock. The board's device tree has one hart, so there is one possible CPU, which the adapter requires. The kit relinks the kernel on each bake's `rootfs.cpio` with the adapter parameters appended (CMDLINE_EXTEND), refuses any other config delta, and records the Image hash in SEQ. |
| Adapter parameters | PLACEHOLDER. Proposed for all three boots: the twin's SPEC set (`twin-run.sh spec`). That is run-musl-io.py io's set plus `dom_null`, `dom_local_tmp`, `native_locks`, `native_stat`, `native_seek`, `native_fd`, `native_close`, `native_path` and `native_clock`, as native01's gcc train01 runs used; `checked_copy_emu` is dropped because the new RTL has the checker. If the bitstream lacks transport or raw moves, use C.5a's model instead (`capstone_vm.c5a_model=1` in place of transport and unpinned). |
| Probes | The QEMU-qualified musl-IO probes (native01 build), byte-identical between the twin and the board stage. Whether their ABI holds on silicon has not been checked (see the open questions). |
| Kit | `bkit36-native64`, a `cp -rL` of `bkit30-unify`. Its rungs are `b0-stats b0-hello vbatch`, as in every kit. The board controls load the process-ABI control module rebuilt against this kernel tree (`adapter/control/capstone.ko`): its `.text`, `.rodata`, `.data`, `.init.text` and `.exit.text` are byte-identical to fpga-pkg-127's `capstone.ko`, and its vermagic is `6.4.14 SMP riscv`. The image's stock `capstone.ko`, rebuilt the same way, is identical to the stock build in the same five sections. `vbatch` runs `ctl.sh`: `native-smoke.sh`, then on boots 2 and 3 `spec.sh`. |
| Control | b0-stats and b0-hello first, before any native code. **A boot whose control rung does not pass is VOID.** |

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
  The QEMU twin of this exact stage on the minimal-config 6.4 QEMU kernel reads that (Image `31296c3b`, `twin/q64f-full`,
  all 9 PASS).
- **The bitstream has no context transport**, or the C.5a parameters are used: steps 10-50 PASS. Steps 60 and 70 FAIL
  with rc 142: each driver ends on its own 30 s alarm, because the second native task waits for the one slot. The 6.4
  twin run of this stage on C.5a's model reads exactly that, `@@NS-RESULT 60` (`twin/q64f-c5a`). The 6.1 gate's C.5a
  probe stopped at the same point. Its last line before `Alarm clock` is "PASS first native task ready", which is
  entry-driver's string (task-driver's binary does not contain it). The probe-1 note attributes that line to
  task-driver.
- Code 4 or 5 is a kernel or command-line fault, not an RTL one. Re-check the kit's SEQ line `native kernel ...`, and
  that `@@NS-CMDLINE` carries the parameters.
- A wedge in steps 20-60 (no `@@NS-RESULT`) names the first native step that fails on silicon. Compare it with the twin
  before writing any attribution.

## Boots 2 and 3: three SPEC runs, native AND Plain on the same kernel

These boots use the same kernel recipe and the same parameters as boot 1. Each starts with the control rungs and the
smoke, and `spec.sh` runs only if the smoke reads 0, 60 or 70, the codes that leave every single-task step passed.
Every executable is stripped; the native images keep `.capstone_virtual`, `.capstone_application` and
`.capstone_domreq`. Stage sizes are recorded before the bake.

**Boot 2** (stage `make-spec-stage.sh a`), in this order:

| # | Command | Arm |
|---|---|---|
| 0 | 519.lbm_r test | Plain |
| 1 | 519.lbm_r test | native |
| 2 | 544.nab_r test | Plain |
| 3 | 544.nab_r test | native |
| 4 | 502.gcc_r train01 | native, last |

Payload estimate: 23,675,287 of 26,000,000 B.

**Boot 3** (stage `make-spec-stage.sh b`): 502.gcc_r train01, Plain. Payload estimate: 21,265,056 B.

The two gcc arms cannot share a payload. gzipped, the native image is 5.72 MB and the Plain one 3.85 MB, against about
10 MB free. Boot 3 uses the same kernel recipe, so Plain gcc runs on the same kernel as native gcc.

**Predicted readings.** The QEMU twin of these exact stages, on the minimal-config kernel with the SPEC parameters
(`twin/q64f-spec-a`, `-b`):

| Command | Twin reading |
|---|---|
| all 6 | rc 0 |
| Plain and native outputs | byte-identical |
| `hkrdenq.out` | equals SPEC's reference |
| `train01...s` (`c786c07f51d56232`) | equals SPEC's reference, and the unstripped native image's QEMU run |
| `lbm.out` | differs from SPEC's reference bytes (specdiff on the host decides) |

Twin wall times, which are not board predictions: lbm 6.72 s Plain / 7.90 s native; nab 6.26 / 7.97; gcc train01
11.84 / 17.75.

On the board, the prediction is that every command reads rc 0 and every output hash equals the twin's
(`spec-classify.py --twin`); a difference goes to specdiff. gcc train01 native completes ONLY if the bitstream's paged
table is in use. On a flat table, the prediction is C.5a probe 2's fault (`capstone_task: fault rc=-14 kind=2 cause=30`).

The native-to-Plain ratio is read only within one kernel. Both arms run on this kernel, so the accounting options cost
both of them. Neither arm is compared with the stock kernel's module or Plain runs.

## Gates before the launch

1. `preflight-board-run.sh` PASS.
2. The kit's deep dry run, `BOARD_STUB=2`, does the real relink on the cpio seed and replays recorded twin transcripts.
   It must reach `BOARD_RUN_DONE` and produce every verdict:
   - PASS on the full twin;
   - FAIL 60 on the C.5a twin;
   - NO RESULT on a transcript cut inside a step;
   - SPEC PASS on the spec-a twin.

   `classify.py` and `spec-classify.py` were also run directly on doctored transcripts: MALFORMED when a pass line is
   removed, SPEC FAIL when a hash is changed, SPEC NO RESULT when the transcript is cut.
3. Read every row of the new bitstream's open ISSUES entries against these images.
4. Check the bitstream sha256 locally. After the flash, read the resident name back. A known-good fallback stays
   registered.

## Open questions this draft cannot settle

- The firmware for the native route on silicon.
- Whether the QEMU-built probes, and the native SPEC images, carry the silicon ABI's requirements: delin sites, glue,
  NONLIN type numbering (the rtl lane's frame-layout review, difference 1).
- The adapter parameters for the new bitstream.
- Whether the control rungs pass under the native route's firmware. The control module loads by vermagic and is
  byte-identical where it matters, but the process ABI needs the monitor's support.
- More than one possible CPU, the negative control for the adapter's refusal, cannot be run on QEMU. QEMU itself
  refuses: "x-capstone-u-mode requires exactly one possible hart". The refusal is in `capstone_vm.c`'s load check.
