# Outside-time levers on board 2: pre-registration

Written and pushed before the boots (5 Nov, 01:30). Board 2, bitstream 3 (caplifive_vp2c_e7af48457), after its C.5a
session (c5a-spec-b2.sh) ends with DONE and the board is back on bitstream 3. Script `xp-b2.sh`, grader `xp-grade.py`.

## Why

Three read-only analyses of the 4 Nov board runs (fault-ahead on, bitstream 3) found that most of what is left of
the outside time is not per exit:
- **Teardown.** Between a command's end and the module's release line, xz spends about 100 s per command on
  130,628 pinned pages: about 0.77 ms per page. Each page is swept by `clear_page_tags`, a load and a store of one
  word per 16-byte granule. The L1 D-cache is the write-through `wt_dcache` with 16-byte lines and no write-allocate
  (rtl: capstone_cv64a6_imafdc_sv39_config_pkg.sv:76, wt_dcache_missunit.sv:477-483). Every load is a full miss plus
  a serialised 1-byte tag read, and a plain store clears the granule's tag with no fill.
- **Launch.** `launch_ns` grows with the image's file size: gcc 21.9 s for 15.6 MB, imagick 5.4 s, lbm 1.3 s. The
  module's `one_inode_per_file` list omits the initramfs ("rootfs"), so CV_IMAGE is refused and the launcher copies
  the image into a sealed memfd. The QEMU twin runs from ext2 and launches in milliseconds.
- **Quantum exits.** The hardware quantum is 80 ms (CAPSTONE_BOOT_QUANTUM / CAPSTONE_SUPERVISOR_QUANTUM =
  2,000,000). nab's 12,826 steps are about 96 % quantum exits. A fitted cost of 5-7 ms per quantum exit would explain
  nab's outside excess over Plain. The board kernel is HZ_PERIODIC at HZ=100, and interrupts are masked for the
  whole quantum.
- **Step budget.** CV_STEP's 5 ms budget stops each fault-ahead window after its first 16-page batch on the board
  (faults ≈ pages / 17 on xz and lbm; QEMU reaches 64-page windows).

## The change

Module 2f81e8a5842010b3: wt-fa perf/retire-sweep-vp2 2188cab26d40, on fault-ahead a6fba52f75c3. Three 0644
parameters, all default off or unchanged:
- `retire_exit_sweep`:
  - 0: unchanged;
  - 1: store-only sweep for an exiting task's teardown;
  - 2: MEASUREMENT ONLY, no sweep;
  - 3: store-only sweep for every retire. The vp2 launcher munmaps right after CV_RETIRE.
  A must-fire line reports each teardown pass: `teardown pages= ns= sweep= exiting=`.
- `image_rootfs`: also accept "rootfs" and "ramfs" in `one_inode_per_file`.
- `step_budget_ns`: CV_STEP's budget, default 5 ms.

Firmware lever (kit q8 only): `-DCAPSTONE_BOOT_QUANTUM=8000000 -DCAPSTONE_SUPERVISOR_QUANTUM=8000000`. That is
320 ms; the RTL csupquantum is 32 bits (csr_regfile.sv:380).

QEMU gate:
- v1: imagick and lbm with sweep 1, PASS.
- v2: imagick and lbm, all levers, PASS.
- v3 (this module, sweep 3): six benchmarks, running.
- No board run until v3's gcc, imagick, nab and lbm PASS: xp-b2.sh checks it before its first board call. Addendum 1 records the gate.

## Kits and runs (one chain per kit, Temporal arm read)

Every kit is bkit12-perf-b3-b2 with module 2f81e8a5842010b3, plus:

| Kit | Params | Firmware | Stage |
|---|---|---|---|
| bkit22-ctl-b2 | fault_ahead_max=64 | stock | gcc |
| bkit22-sweep-b2 | + retire_exit_sweep=3 | stock | gcc |
| bkit22-nosweep-b2 | + retire_exit_sweep=2 | stock | gcc |
| bkit22-root-b2 | + image_rootfs=1 | stock | gcc, imagick |
| bkit22-q8-b2 | fault_ahead_max=64 | quantum 8,000,000 | nab |
| bkit22-allmod-b2 | + retire_exit_sweep=3, image_rootfs=1, step_budget_ns=80000000 | stock | lbm |

The reference for run time, launch, steps and faults is out-fa-b1 (board 1, bitstream 3, fault-ahead 64, module
6e854c46c654c843, the same default paths). It is cross-board; the bound on the board offset is about 0.5 %.

## Predictions (`xp-grade.py`)

| # | Prediction |
|---|---|
| X0 | Every run: rc 0, specdiff PASS, outputs equal the twin's, the control passes, every kit parameter is read back as set, and QNT0 reads 001E8480 (q8: 007A1200). |
| X1 | ctl, gcc: the teardown costs 0.4-1.2 ms per page. This is the instrument's control: the xz-derived 0.77 ms. |
| X2 | sweep, gcc: at most 0.35 × X1's ms per page. No line fill and no tag read; two posted writes per granule. |
| X3 | nosweep, gcc: at most 0.15 × X1, reported. This bounds unpin plus dirty. If it reads near X1, the cost is not the sweep. |
| X4 | root, gcc: launch at most 5 s, against 21.9 s. Killed if launch stays at or above 18 s. |
| X4b | root, imagick: launch over both commands at most 2.9 s, against 6.7 s. |
| X5 | q8, nab: outside = run − cycles/25 MHz. At most 160 s reads H-exit: the cost is per quantum exit, and a longer quantum is the fix. At least 185 s reads H-time: the cost is per tick or per second, and a tickless kernel is the candidate. In between reads partial. Reported. Reference 198.4 s. |
| X6 | allmod, lbm: run time at most 0.985 × out-fa-b1's 1,963 s. The sweep covers about 105 k CV_RETIRE pages × about 0.5 ms, plus about 4.5 k fewer faults. |
| X6b | allmod, lbm: faults at most 2,500, against 6,211. QEMU reads 1,646. |

**Grader controls.**
- out-fa-b1's runs given as every kit read rc 1: X0 fails on the params and QNT0, there is no teardown line, and launch is unchanged.
- A mock with the predicted values reads rc 0.
- No data reads rc 2.

**Dry run.** perf/xp-b2/dryrun, branches:
- A-ok;
- B-noprereg;
- C-notdone;
- D-wait;
- E-busy;
- F-readback;
- G2-exists;
- H-busychain;
- F-defaults: real paths, every board call stubbed;
- I-qgate: the QEMU gate not passed (the script checks modrs/q3-{gcc,imagick,nab,lbm} before any board call).

**After it.** dt-b2.sh, the stage A session, is relaunched to wait on this script's pid; it is otherwise unchanged.

**Limits.**
- N = 1 per run.
- Test inputs only, which overweight launch and teardown against train and ref.
- `retire_exit_sweep=2` leaves tags in returned frames; it is a timing measurement only.

## Addendum 1 (5 Nov, 02:10): the QEMU gate of this module

The gate is kit spec/kit-rsq3. It runs QEMU module 4e6f76009598f053, the same tree 2188cab26d40, with every lever on:
`fault_ahead_max=64`, `retire_exit_sweep=3`, `image_rootfs=1`, `step_budget_ns=80000000`. It covers the six SPEC test
stages in their Spatial arm, in parallel slots.

**Result: six of six PASS.**
- Every command has rc 0 and specdiff PASS, and outputs are equal: gcc 1/1, imagick 2/2, nab 1/1, lbm 1/1,
  xz 12/12, mcf 2/2.
- Every kit parameter reads back as set (`image_rootfs=Y`).
- The teardown line fires with `sweep=3 exiting=1` (xz: 130,704 pages per command).

**What this shows, and what it does not.** Mode 3 zeroes one word of every 16-byte granule of every retired page,
including the CV_RETIRE pages that lbm and mcf retire during the run. No benchmark output changed, so no program read
a retired page after its retire. The rootfs lever cannot fire on the twin, which runs from ext2. Timings are not read
on QEMU.

The gate that xp-b2.sh checks (gcc, imagick, nab and lbm) is therefore met.

## Addendum 2 (5 Nov, 03:05): the exact image grant, one more run

**The change.** capstone-vexec 0bde5fd8e0c9c374 is built from wt-fa perf/exact-image-grant with
`-DCAPSTONE_VEXEC_EXACT_IMAGE`. The default build stays byte-identical (978c09d61f774315).
- load() grants the image at the smallest exactly representable length, page-rounded, at a granule-aligned base,
  instead of the next power of two.
- For gcc's baseline image that is 0x1068000 instead of 0x2000000, so 596 pinned pages instead of 4,588.
- This padding is the whole of the eval RESULT's gcc page drop (Corrections, 20700808adbf).

**The run.** Kit bkit22-eg-b2 is bkit22-ctl-b2 with that launcher. It runs gcc right after ctl's gcc: same board,
same session, one variable. The script's copy of spec-board.sh takes SB_VEXEC; the default hash is unchanged.

**Its own QEMU gate.** Kit spec/kit-egq is kit-rsq3 with the QEMU-built launcher a7596c8eaad129f3, on all six
stages. xp-b2.sh runs eg only once modrs/qeg-{gcc,imagick,nab,lbm,xz,mcf} are all done with rc 0 and specdiff PASS.
Otherwise it logs SKIP and goes on.

**X7** (eg against ctl, gcc), all three must hold:
- peak pages at most 2,700, against about 6,150;
- launch at least 3 s lower: 3,992 fewer pins at 1.1-1.9 ms;
- run time at least 5 s lower, including about 3.2 s of teardown.

Grader controls:
- ctl's run given as eg reads FAILS.
- A mock at the predicted values reads holds.

Dry-run branches J-egskip (the gate not passed) and K-egrun (each chain gets its own launcher hash) are added.
