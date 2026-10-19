# Tier 3 board backend smoke (2026-10-18)

The first real use of `tools/boardvm.py` and `tools/board/board-run.sh`: the corpus's virtual positive controls in one
board boot, through the same three steps as a QEMU boot (`smoke.py`).

**The image.** `controls.dom` 491655742d378fcd: postgres c-repros shared/controls.c, built `-O0` with the v9 kit's SDK.
- Gates: delin-census 0, qemu-only-csr-census 0.
- vp2-silicon-gate: 0 refused, one attributed site, `libc_exit_fini`'s LDC (unreachable: the runtime overrides the
  function, C-82, and .fini_array is empty, checked by the gate). See `silicon-gate-attributions.tsv`.

**The kits:**
- **QEMU:** 3b8040d5ff6e (809d1fc107e64515) with the v7-bundle5 flat images and the compiler lane's flat adapter
  65519da197423250.
- **Board:** module ce3a542d00acce86 and launcher b27afefc09c6cae6 (Tier 1's), monitor capstone-sbi vp2/fetch-retry
  007d81d with the S-23 retry and the CPMX diagnostic armed, and the Tier 1 base set for the b0-stats control
  (`base.manifest`, `board.env` with the scratch path as `<t3-smoke>`). The firmware is built inside the recipe.

**The QEMU prediction** (run 2026-10-18 07:21, `smoke.py --qemu-kit`):
- control-bounds-malloc: fault, cause 28;
- control-uaf-malloc: fault, cause 25;
- control-clean: exit 75 (controls.c's refusal of an unknown control name, a deterministic exit).

**Pre-registered for the board boot:**
- b0-stats 0, then the batch rung with the same three readings and `completed=True`.
- **What would show a backend defect** (to be fixed before any corpus boot):
  - NO DATA, a missing step, or two batch start markers;
  - a step read from outside the batch segment;
  - a firmware or kernel hash absent from the recipe's record.
- **What would show a silicon divergence:** a different cause or exit, named with its fault line.

## Result, first smoke boot (2026-10-18 07:23-07:42): VOID, a backend defect found and fixed

- **The recipe ran:** a bake of 10 files (checked present, then absent, by hash), firmware 86b21d9f7ebd674e
  (17.5 MB, under the 26 MB cap), and the driver.
- **The board control failed:** b0-stats read **901**, "B0: a module without the process ABI is already loaded and
  cannot be unloaded". The image still carried the stock /capstone.ko.
  - **Cause, in the recipe:** board-run.sh sourced board.env without exporting it, so BAKE_REPLACE_KO never reached
    bake-manifest.sh, a child process. The replacement silently did not happen.
  - bake-manifest.sh did not check the replaced module by hash, which would have caught it at bake time. The vp2
    Tier 1 bake did check it.
- **The batch then hung in its first step** (control-bounds-malloc: 13 minutes of heartbeats, where QEMU faults it
  with cause 28 in seconds). This is with the classic module loaded beside capstone_vm, a configuration no Tier 1 boot
  had. The run was ended by SIGINT.
- **The fix:** board-run.sh sources board.env under `set -a`. bake-manifest.sh checks the initramfs's /capstone.ko by
  hash, against the replacement after the bake and against the stock module after the restore.
- **The rerun is pending,** on the next free board slot. Until it passes, the backend is not validated on silicon.

## Pre-registered (2026-10-18, before the second smoke boot): the same smoke on the fixed recipe

- **The kit:** `<t3-smoke>/bkit2`, a fresh copy.
  - It differs from the first boot's kit only in the two fixed scripts and in BOARD_BASE_MANIFEST naming its own
    copy, which has the same five lines.
  - board.env, the module, the launcher, the monitor and the defines are unchanged.
- **The new must-fire check:** the recipe's stub mode now prints what a CHILD process sees of board.env.
  - The dry run reads `BAKE_REPLACE_KO=/tmp/capstone/r55-table/fpga-pkg-127/module/capstone.ko`.
  - The same kit with `set -a` removed reads `BAKE_REPLACE_KO=UNSET`. The first boot's stub printed only commands, so
    it could not show this.
- **The run:** `smoke.py --board-kit <t3-smoke>/bkit2 --image <t3-smoke>/bin/controls.dom --raw <t3-smoke>/board-run2`.
- **Predicted:**
  - The bake prints `cpio /capstone.ko: the replaced module` (f2a2645cc570dcc6, the module the Tier 1 bakes put
    there), and after the restore, `the stock module`.
  - b0-stats 0.
  - The batch rung reads QEMU's three readings: fault cause 28, fault cause 25, exit 75, with `completed=True`.
- **If control-bounds-malloc hangs again** with the process-ABI module in place, the hang is not the recipe's. It is a
  silicon reading of `capstone-exec`'s fault path in a corpus image, and it is recorded with the batch rung's last
  lines.

## Result, second smoke boot (2026-10-18 08:13-08:20): control-bounds-malloc hangs again, with the recipe fixed

- **The recipe did what was pre-registered.**
  - The bake printed `cpio /capstone.ko: the replaced module` (f2a2645cc570dcc6) after baking and `the stock module`
    after the restore.
  - Firmware ccff4dcfc354591a, 17,467,912 B; b0-stats 0.
- **The batch rung entered control-bounds-malloc and never left it.**
  - The heartbeat kept printing (`@@HB 1..3`), so Linux was alive. The step printed nothing, because its output goes
    to a file until it ends.
  - The monitor's S-23 retry line (FRTY), which fires about once per launcher process in every Tier 1 boot, never
    appeared. The process never reached its first retire.
  - QEMU faults this control with cause 28 within seconds.
- **The run was ended by SIGINT after about 3 minutes in the step.** The first smoke ran 13 minutes without the step
  ending. The driver released the board, and no board.txt was written: an interrupted driver never reaches that step.
  This record comes from the scoped stream.
- **This is not the recipe's defect,** as pre-registered. Two facts narrow it:
  - The adapter (launcher b27afefc, module ce3a542d) is Tier 1's own, and Tier 1 runs contract.dom through it on
    every boot.
  - A live Linux rules out the RTL lane's ghost-store wedge (a faulting STC leaving a commit-queue entry that never
    drains), which stops the whole core.

## Pre-registered (2026-10-18, before the third smoke boot): the diagnostic smoke, every step made to return

- **The run:** `smoke-diag.py --board-kit <t3-smoke>/bkit3 --controls <t3-smoke>/bin/controls.dom --contract
  <stage>/vm/contract.dom --raw <t3-smoke>/board-diag`.
- **The steps,** in one boot through the same BoardBatch backend:
  1. **contract-ok:** Tier 1's contract.dom (125705371837ea92), `ok argument`. It is the matched arm: the same gate
     script, launcher, module and batch environment, with a different image.
  2. **The three controls,** each bounded.
     - At 150 s a watcher prints every launcher thread's `/proc` state and wchan (R spinning, S sleeping,
       D uninterruptible), then sends SIGTERM, and SIGKILL 10 s later.
     - Each step then prints the module's lines from the ring buffer (dmesg). The console shows only KERN_EMERG
       during rungs.
- **bkit3** differs from bkit2 in three ways:
  - the recipe reads the core's apertures after the last rung when nothing wedged (driver 79eaf36851e12d05,
    `BAKED_FINAL_APERTURES`, tested offline in four arms with a negative control);
  - the bake's restore removes the directories that held only its files;
  - boardvm's docstring no longer claims a preflight.
  - Module, launcher, monitor and defines are unchanged.
- **Predicted:**
  - **The bake:** the replaced and stock module checks as before, and no `test-domains/vbatch` directory left in
    the shared overlay afterwards.
  - b0-stats 0.
  - **contract-ok:** `BOUNDED-EXIT 0` and `VIRTUAL_APPLICATION_OK`. If it fails, the batch environment itself is
    the variable.
  - **control-bounds-malloc, if the hang repeats:**
    - `BOUNDED-EXIT 143`, or 137 after the SIGKILL;
    - the BOUND-STATE lines name where the launcher is;
    - whether `CONTROL bounds-malloc mark` printed says whether the image reached main().
  - **control-bounds-malloc, if it returns:** QEMU's cause 28.
  - **uaf-malloc and clean,** if reached: cause 25, and exit 75.
  - **The resting apertures** (the RTL lane's control, if nothing wedges): 193 = 0 with no valid bit in 221, 225
    bit 3 = 0, 227 bit 7 = 0, 228 = 0xb8 or 0x9f (index 56 or 31). A non-zero 193 with fewer valid bits than the
    count is a ghost entry counted in this boot.

## Result, the diagnostic boot (2026-10-18 08:40-08:57): Tier 1's own contract.dom hangs under the batch gate too

- **The bake:** the replaced and stock module checks both printed. Firmware 29cfc290d4e01078; its monitor sources
  and defines equal r12's. b0-stats 0.
- **Every step hit the 150 s bound.** In each, the launcher was in state R with wchan 0: running, not sleeping or
  blocked in the kernel. SIGTERM ended it with 143.
  - **contract-ok:** the launcher printed the image's five grants (slots 0-4). In every Tier 1 boot the next line
    is slot 5, the domain's first runtime grant, and the step ends with VIRTUAL_APPLICATION_OK in seconds.
  - **The three controls:** none printed its `CONTROL ... mark`.
  - The monitor's S-23 retry (FRTY) never fired. It fires about once per launcher process in every Tier 1 boot.
- **So the variable is the batch environment, not the corpus image.** The known-good image fails the same way.
- **The resting apertures (BAKED_FINAL_APERTURES) are not readable as a snapshot.** The pre-registered
  `193 = 0` was an instrument error.
  - The RTL lane quoted the source: the board's LED pulse stretcher holds each bit high for about 2^20 cycles
    (ariane_xilinx.sv:956-966).
  - So on a running core every dynamic aperture reads as the OR over that window. This read gave 193 = 7 for a
    4-deep queue, and 221 = 0xfe, a combination no single cycle can hold.
  - **What stays valid at rest:** 228 = 0xeb, index 107, the switcher's rest value after an ESCAPE. The four contexts
    escaped and were never resumed: the kernel side never reached the module's resume path. And the latched trap log.
  - The driver's comment now says this. A commit-queue control needs a parked core.

## Pre-registered (2026-10-18, before the fourth smoke boot): Tier 1's rung, then the batch gate, in one boot

- **The rungs:** b0-stats, b0-hello, vm-contract, vbatch.
  - **vm-contract** is Tier 1's own rung: vmrun.sh with vm-contract.sh in /test-domains/vm. Its contract.dom
    (125705371837ea92), launcher (b27afefc) and module (ce3a542d, the r9/r10 module) are byte-identical to the batch's.
  - **vbatch** is the diagnostic batch: contract-ok, then the three controls.
- **bkit4** differs from bkit3 in four ways:
  - `BOARD_RUNGS`, so the recipe runs any rung list containing vbatch, with an oracle of 0 for each rung;
  - the batch controller hands every other rung to vmrun.sh (`--other-ctl`, boardvm `other_ctl`);
  - the Tier 1 files are added to the base set, pinned as copies in the kit;
  - the driver comment (fd38816a90c64c22).
- **The bound's probe** now prints the launcher and its children twice, 2 s apart: state, utime, stime, wchan and
  the current syscall. A utime that grows means spinning in user space; a stime that grows means spinning in the
  kernel.
- **What the boot separates:**
  - **vm-contract 0 and the batch's contract-ok 0:** a virtual context needs a warm-up (a classic domain run, or
    Tier 1's rung) before the first one. The controls then give their first silicon readings: cause 28, cause 25,
    exit 75.
  - **vm-contract 0 and contract-ok hangs:** the batch gate's environment is the variable (its cwd, image path or
    preamble), independent of the warm-up.
  - **vm-contract hangs:** the base set or the recipe's image differs from the Tier 1 bake. The batch's reading is
    then collateral.

## Result, the fourth smoke boot (2026-10-18 09:01-09:10): the backend reads the controls exactly as QEMU does, once a warm-up runs first

**Firmware** dca331cb55bb2363; the bake's module checks both printed. The rungs read:
- **b0-stats 0, b0-hello 0.**
- **vm-contract** (Tier 1's own rung) reads identity: VIRTUAL_APPLICATION_OK, then the stale access denied with 25.
  One FRTY, as in every Tier 1 boot.
- **The batch, every step returned well inside its bound:**
  - **contract-ok:** all its grants (slots 0-8) and its stats, then `VM_CONTRACT_FAIL:environment`, exit 1. That is
    expected: the batch gate does not export `CAPSTONE_VM_CONTRACT=environment`, which Tier 1's script does. The
    step's point was that it no longer spins.
  - **The three controls** equal the QEMU smoke's readings in cause and code-relative pc, compared by script on the
    two smoke outputs:

    | step | board | QEMU |
    |---|---|---|
    | control-bounds-malloc | mark, cause 28 at code offset 0x104d8 | the same |
    | control-uaf-malloc | mark, cause 25 at 0x10524 | the same |
    | control-clean | exit 75 | the same |

    The image is controls.dom 491655742d378fcd, the sha256 in the launcher's own fault line.
- **The resting apertures** read 228 = index 107, as before. The dynamic fields are the stretcher's OR and are not
  recorded as a reading.

**What the four smoke boots separate:**
- In boots 1-3, no classic domain and no Tier 1 rung ran before the batch. The first virtual context then spun in R
  after the image's five grants, with no S-23 retry and the switcher resting at an ESCAPE. Every later context in
  those boots did the same.
- In boot 4 the batch ran after b0-hello and vm-contract, and every step returned.
- So on vp2 the first virtual context after a cold boot needs a warm-up run before it. Which of b0-hello (a classic
  domain) and vm-contract (a first virtual context) is the warm-up is not separated yet.
- Every Tier 1/2 boot ran both before any virtual step, which is why this was never seen there.
- **The Tier 3 recipe carries the validated sequence:** `BOARD_RUNGS="b0-stats b0-hello vm-contract vbatch"`, with
  Tier 1's files in the base set (bkit4).

## Pre-registered (2026-10-18, before the fifth smoke boot): which rung is the warm-up

- **The one change from boot 4:** `BOARD_RUNGS="b0-stats b0-hello vbatch"`. vm-contract does not run. The kit is
  bkit5, a copy of bkit4: base set, recipe, driver, module, launcher, monitor and the batch's four steps are all
  unchanged.
- **The batch returns as in boot 4** (contract-ok exit 1 with VM_CONTRACT_FAIL:environment, then cause 28, cause 25,
  exit 75): a classic domain run (b0-hello) is the warm-up.
- **The batch spins as in boots 2-3** (state R at the bound, no mark): b0-hello is not enough. Then what made boot 4
  work is a first virtual context run by Tier 1's script, and the batch gate is the variable only for the FIRST
  context of a boot.

## Result, the fifth smoke boot (2026-10-18 09:13-09:21): b0-hello alone is the warm-up

- **The rungs:** b0-stats 0, b0-hello 0, then the batch. vm-contract did not run. Firmware 535688fd0d434ae9.
- **Every batch step returned:**
  - contract-ok: VM_CONTRACT_FAIL:environment, exit 1, as in boot 4;
  - control-bounds-malloc: cause 28 at 0x104d8;
  - control-uaf-malloc: cause 25 at 0x10524;
  - control-clean: exit 75.
  - The three controls equal QEMU's by script, for the second time.
- **So a classic domain run is the warm-up** the first virtual context after a cold boot needs. Filed as ISSUES S-26.
- **The S-23 retry (FRTY) count, by rung, compared by script:**
  - Tier 1's vm-contract rung: 1, over its two launcher processes (boot 4 and r12);
  - r12's vm-suite: 3, over about 17 processes;
  - the batch: 0, over four processes, in boots 4 and 5.
  - So the retry is not once per process; it depends on the step.

## Pre-registered (2026-10-18): the Tier 3 runner pilot, one corpus arm end to end (not a record)

- **What:** `pilot.sh <t3-smoke>/bkit6 <scratch>` runs `run-virtual-cases.py --board-kit` on memcached
  plain-temporal-repros, arm virtual-malloc: its controls and three cases, fixed and buggy.
  - The SDK is the v9 kit's.
  - The bundle goes to scratch, not under the corpus's results/.
  - bkit6 is bkit3 with `BOARD_RUNGS="b0-stats b0-hello vbatch"`. The warm-up is S-26's; the scripts are dev's.
- **What it answers:**
  - **The path:** the runner's build, plan, BoardBatch boot, observe and verdicts writer, on silicon.
  - **The payload:** each image's size and gzip size, the stage's, and the firmware against the 26,000,000-byte cap.
    This is the first measured point of the boot plan.
  - **Observe on a real case:** whether it yields the same fields as QEMU, cell by cell (`compare-pilot.py`, tested
    two-sided: SAME on itself, DIFF on a changed offset).
- **The prediction:** the compiler lane's QEMU record of the same images on the reference pin, capstone-qemu
  b290f5656600 with svadu=false, matched by image sha256.
  - Until it exists, the reference is the v6-kit QEMU bundle of this corpus. Its images differ, so only verdict,
    cause and symbol+offset are comparable: CAUGHT, cause 25 at `mct_read_probe+56`, `mct_write_probe+88` and
    `mct_read_probe+56`.
- **A wedge is S-25's,** if its signature is a commit count with no valid bits. It is recorded as that, not as the
  runner's failure.

## Result, the runner pilot (2026-10-18 09:26-09:34): the whole path works on silicon, and the three cells read as QEMU's

- **The path:** `run-virtual-cases.py --board-kit` built the images with the v9 SDK, staged them, baked, booted and
  observed, then wrote a verdicts-v1 bundle. inputs.json records machine `board`, firmware 8c10d8afb64ba949, the
  kernel, launcher b27afefc, module ce3a542d and monitor 007d81d.
- **The rungs:** b0-stats 0 and b0-hello 0 (S-26's warm-up), then the batch.
- **The controls:** bounds-malloc and uaf-malloc both fault, as the configuration expects.
- **The cases:** all three fixed arms exit 0, and all three buggy arms fault with cause 25.
  - The verdicts: 00 CAUGHT at `mct_read_probe+56`, 01 CAUGHT at `mct_write_probe+88`, 02 CAUGHT at
    `mct_read_probe+56`, each reached and attributed by probe.
  - Against the v6-kit QEMU bundle of this corpus (different images), verdict, cause, symbol+offset, reached,
    completed and attribution are equal in all three cells, compared by script.
  - **Against the reference pin** (the compiler lane's record on capstone-qemu b290f5656600, `cpu.txt` showing
    svadu=false, the smoke kit with only the QEMU binary swapped), `compare-pilot.py` reads **3 of 3 cells the same
    by image**: the three image hashes match, and verdict, cause, symbol+offset, reached, completed, attribution and
    both controls are equal.
- **The payload, the boot plan's first measured point:**
  - Each case image is about 197 KB, about 76.6 KB gzipped; controls.dom 75.1 KB gzipped.
  - The stage was 1,657,992 B, 686,071 B gzipped.
  - The firmware stayed at 17,467,912 B, the clean size's 2 MiB step, well under the 26,000,000-byte cap.
- **The time:** bake, firmware and boot to the end of the batch took 8 minutes for 9 steps.

## Tier 3's 26 arms on the board: which runner, and what each still needs (2026-10-18)

| runner | arms | board path |
|---|---|---|
| `tools/run-virtual-cases.py` (case.c corpora and the prebuilt ffmpeg pool / memcached allocator ports) | 13: ffmpeg carved x2, plain-heap, plain-temporal, pool x2, subobject; memcached allocator x2, plain-heap, plain-temporal; wireshark plain-heap, plain-temporal | `--board-kit`, validated by the pilot |
| `postgres/c-repros/shared/run-arm.py`, `mmgr-repros/shared/run-defects.py`, `sql-repros/shared/run-arm.py` (virtualvm.Batch) | 5 | `--board-kit` (lane 0d542dc4f97b), the same BoardBatch drop-in |
| `sqlite/engine-repros/shared/run-virtual.py`, `wireshark/wmem-repros/shared/run-defects.py` (appvm.run_app) | 4 | the appvm batch session (`tools/board/appvm-session.py`). Its QEMU control (`session-control.sh`) passes: wmem virtual-malloc collected, booted once on the v6-fix kit and read back equals the direct appvm run of the same builds in 23 of 23 cells by image, with the share linked and the marker read through /mnt/host |
| `cpython/pymalloc-repros/runners/virtual/run-virtual.py` (appvm.run_app, per-case fixtures in the share, no stdlib) | 2 | the appvm batch session, as wmem: each fixture path is unique per case number and its content a function of that number |
| `cpython/allocator-boundary-repros/runners/virtual/run-virtual.py` (appvm.run_app with a staged stdlib) | 2 | **needs a trimmed stdlib**. The full CPython 3.13 Lib is 287 MB raw and 86 MB gzipped, against about 8.5 MB of headroom under the cap. The static import closure (modulefinder over the 84 trigger and control scripts and the runner's launch.py, plus the interpreter's startup modules) is 328 files: 7.6 MB of source, 1.9 MB gzipped. modulefinder misses dynamic imports, so a trimmed stdlib must first equal the full one on QEMU, every reading identical, before it boots. |

- **Every board arm records under `vp2-<arm>`.** The QEMU bundle stays the virtual arm's current record.
- **The sql runner's payload is not measured yet.** It stages the PostgreSQL share and an initdb'd cluster. The
  recipe's 26 MB cap check stops it before any boot if it does not fit.
- **The order on the board,** after the pinned bundles and the arena-pin module:
  - the 13 run-virtual-cases arms first: the pilot's path, a 9-step boot is about 8 minutes;
  - then the postgres three;
  - then the two appvm corpora.
  - Each boot runs b0-stats and b0-hello first (S-26's warm-up).
  - A wedge with R-57's signature (a commit count with no valid bits) is S-25's residual, recorded as that.

## The campaign's SDK, frozen (2026-10-18)

The v9 SDK's `sdk.json` named a live LLVM build directory (`llvm-lane-c27`), which the compiler lane rebuilt
incrementally at 08:06 today. clang is a thin driver over `lib/*.so` (BUILD_SHARED_LIBS), so a rebuild can change
codegen with the driver's hash unchanged. Over a 26-boot campaign that would split the records across two compilers
without anything showing it. So every Tier 3 arm, on the board and in its QEMU prediction, is compiled from a frozen,
read-only copy: `<mnt-hdd>/capstone/t3-freeze`.
- **What it holds:** llvm-lane-c27's bin and lib, plus the v9 SDK, libc, musl and link script, with `sdk.json` pointing
  at the copies. Each copy was compared equal to its source.
- **Identity:**
  - `MANIFEST.SHA256` sha256/16 44528aa3d25debd5, over 6685 files, 119 of them `lib/*.so`.
  - `libLLVMCapstoneCodeGen.so` 43e1781e3f1a5e5d. The compiler lane identifies it as the post-08:06 library, the one
    the pilot and its recent records used.
  - clang-22 a5be1a4d621513eb, the bundles' `compiler` field. That is the driver alone, so cite the manifest.
- **Self-contained, checked three ways:**
  - `ldd`: all 103 of clang's libraries resolve inside the copy.
  - strace of a full corpus build: the only opens naming the live tree are ENOENT probes for system libraries,
    through the RUNPATH's absolute fallback.
  - The pilot's four images rebuild byte-identical from it.
- **The plain case.c arms' images**, built from it with no boot, for the hash-matched predictions:
  `<mnt-hdd>/capstone/t3-images/<tag>.images.SHA256SUMS`. There are 7 arms and 92 images. controls.dom is
  491655742d378fcd in all of them.

## Tier 3 proper: the campaign's rules, pre-registered once (2026-10-18), and each arm's prediction before its boot

**The configuration.**
- Board kit `bkit7`:
  - module 3005ed52 with arena_pin OFF (boot 5: the pin removes no wedge, and pin-off is 4c's configuration);
  - launcher b27afefc;
  - monitor 007d81d with the vp2 defines;
  - rungs b0-stats, b0-hello (S-26's warm-up), then vbatch.
- SDK `t3-freeze`.
- One arm per boot, through `tier3-seq.sh` with the arm's line from `t3-images/ARMS.spec`. The board records `vp2-<arm>`.

**The prediction for each arm:** the compiler lane's QEMU run of the same images. Its kit is `t3-smoke/qkit7`:
- capstone-qemu b290f5656600, binary 7a4f3a64776f0c88;
- behind a wrapper (5f731f99ec6a19b2) that strips exact bounds and appends svadu=false, logging every -cpu string;
- module 5bfc38f8, the QEMU-kernel build of 3005ed52's source, with no params.

That run's images must equal the board's staged list by sha256. The run's cells, as `tier3-cells.py` lines, are
committed under `records/<tag>.prediction.txt` before that arm boots.

**The comparison:** `compare-pilot.py <prediction bundle> <board bundle>`, cell by cell, by image.
- **Pass:** every cell and both controls SAME: verdict, reason, reached, completed, cause, symbol+offset, attribution.
- **A wedge:** the batch ends early, and the cells after it read NO-READING:infra.
  - The driver's apertures are compared with `board-tier12/boot3/compare-wedge-apertures.py`.
  - R-57's signature (a commit count with no valid bit) is read as S-25's residual on this bitstream, not as the
    case's verdict.
  - The unreached cells are re-run in a later boot, from the next case on (`--only`).
- **Any DIFF** is a finding. It is read from the case's own transcript before anything is said about it.

**Arm mpt** (memcached/plain-temporal, 3 cases), the first boot, prediction in `records/mpt.prediction.txt`:
- CAUGHT 3, each cause 25 at its labelled probe (mct_read_probe+56, mct_write_probe+88, mct_read_probe+56);
- both controls fault.

This is the pilot's reading on the same image bytes, now on the campaign's module and frozen SDK.

**Amendment (2026-10-18, before the second arm): the SDK is `t3-freeze-nb`, the frozen tree with `-fno-builtin`.**
- **Why:** the frozen v9 SDK compiled applications without `-fno-builtin`. Its `sdk.json` codegen lacks the flag,
  which the plan pins for Tiers 1-3 and which Run 5 showed clears mmgr case 00's tag loss.
- **What changed:** `t3-freeze-nb` is the same frozen tree with `-fno-builtin` appended to the codegen. The compiler
  lane checked that the two `sdk.json` files differ in that key alone, by that one item.
- **mpt** has four byte-identical images either way, so its pre-registration and boot stand for this SDK.
- **The other arms' images** are in `t3-images-nb/` (the prebuilt trees rebuilt there too). Against v9 they differ in
  0 to 2 images for the plain arms, and in nearly all for fso and the four prebuilt arms.
- **Caveat 1, from the compiler lane's measurement:** `-fno-builtin` is a PARTIAL mitigation for C-81. It diverts
  explicit `memcpy` calls, but not `__builtin_memcpy` and not a copy the compiler synthesises. Its scanner reads
  `copy32_unknown_builtin` as losing tags at -O0, -O1 and -O2, with the flag on. So a clean cell here is no evidence
  about C-81 exposure.
- **Caveat 2:** Tier 1/2's application images (contract, sqlite3, mruby) were v9 without the flag. A cross-tier
  comparison of application behaviour compares two build configurations.

### Arm mpt, result (2026-10-18 12:50-12:58): 3 of 3 cells the same as the prediction; the campaign's first silicon record

`records/mpt.board.txt`. memcached/plain-temporal-repros, recorded vp2-malloc. Firmware 55cd2bc4d221b627
(17,467,912 B, the clean base's 2 MiB step); module 3005ed52 with the pin off.
- **CAUGHT 3**, each with cause 25 at the predicted probe. Both controls fault.
- `compare-pilot.py` against the compiler lane's qkit7 prediction reads 3 of 3 SAME, by image sha256.
- The boot completed: VIRTUAL_STAGED_DONE, and the batch rung returned 0.
- Its images are byte-identical on t3-freeze and t3-freeze-nb, so this reading stands under the amendment.

### Arms wm0 and wm1 (wireshark/wmem-repros), pre-registered (2026-10-18): through the appvm batch session, predicted on qkit7 by the board lane

- **The mechanism:** `tier3-appvm-arm.sh` runs `run-defects.py` through `tools/board/appvm-session.py`: collect, one
  boot, read. The prediction ran it with `--virtual-kit qkit7`, and the board runs it with `--board-kit bkit7`. The
  hosted builds are `t3-images-nb/wmem/{cases,controls}-{OFF,ON}`, made on t3-freeze-nb.
- **The image check:** with `EXPECT_IMAGES` set to the prediction's list, the board's collected images must equal the
  prediction's by sha256 before the boot, or the script stops. Dry-run both ways: another arm's list stops it, and the
  right list passes.
- **These two predictions are the board lane's run on the same qkit7**, an amendment for the appvm arms. The lock was
  free, and the compiler lane was told and can re-run them as a cross-check.
- **wm0 (virtual-malloc, recorded vp2-malloc), `records/wm0.prediction.txt`:**
  - MISSED 23, every case completing;
  - the controls: bounds-chunk complete, uaf-chunk complete, uaf-jumbo fault.
  - This equals the corpus record and the v6-fix run.
- **wm1 (virtual-nested-pools, recorded vp2-nested-pools), `records/wm1.prediction.txt`:** NO-READING:control-failed 23.
  - The cases themselves fault 25 at `wm_probe`.
  - But the uaf-jumbo control faults with cause 29 during setup, before its ready mark, so the runner voids every
    cell.
  - This is the v6-fix run's reading exactly: wmem patch 0001's jumbo-block derive is refused by Q-15 with exact
    bounds off. The exact-bounds rerun there returned 22 of 23 cells to the record.
  - Silicon's bounds are not exact, so silicon is predicted to refuse the same derive. A jumbo control that completes
    on the board would be a finding about Q-15 on silicon.
- **The order:** wm0, then wm1, one boot each.

### Arms mph, fph, fpt, pre-registered (2026-10-18): the compiler lane's nb predictions, each on the board's bytes

Each record was written by `register-prediction.sh`, which first checked that the run's staged images equal
`t3-images-nb/<tag>.images.SHA256SUMS`, and that its module is 5bfc38f8.
- **mph** (memcached/plain-heap, vp2-malloc): CAUGHT 9. `records/mph.prediction.txt`.
- **fph** (ffmpeg/plain-heap, vp2-malloc): CAUGHT 25. `records/fph.prediction.txt`.
- **fpt** (ffmpeg/plain-temporal, vp2-malloc): CAUGHT 13. `records/fpt.prediction.txt`.
Each tally is its corpus record's. These three boot after wm1, through `tier3-seq.sh` with `--sdk t3-freeze-nb/sdk`
and `t3-images-nb/ARMS.spec`, one boot each, in that order.
- **fca** (ffmpeg/carved, virtual-malloc, recorded vp2-malloc): MISSED 13, its record. `records/fca.prediction.txt`.
- **fcs** (ffmpeg/carved, virtual-nested-pools, recorded vp2-nested-pools): CAUGHT 13, its record.
  `records/fcs.prediction.txt`.
  - All four carve controls read as expected: live-carve complete; uaf-carve, uaf-malloc and bounds-malloc fault.
  - Cases 00 and 10 read NO-READING control-failed on v6-fix, a NULL from the module's add(). Here they read CAUGHT.
- **Provenance:** fcs and fso compile port sources from the lane worktree and `ffmpeg-buffer-pool`, both outside the
  freeze. No commit since bbc46be9d596 touches `capstone/ports` or `capstone/runtime`, and the image-hash gate
  catches any drift regardless.
- **The order after wm1:** mph, fph, fpt, fca, fcs.
- **wph** (wireshark/plain-heap, vp2-malloc): CAUGHT 12, its record. `records/wph.prediction.txt`.
- **wpt** (wireshark/plain-temporal, vp2-malloc): CAUGHT 10, its record. `records/wpt.prediction.txt`.
- **The order after wm1:** mph, fph, fpt, fca, fcs, wph, wpt.

### Arm wm0, result (2026-10-18 13:02-13:13): the core wedges in the uaf-jumbo control, a signature that is neither S-25's kit-application face nor R-49 as fixed

`records/wm0.board.txt`. The board's collected images equalled the prediction's 26, then the boot ran 2 of 26 steps:
session-share 0, then uaf-chunk 0 (as predicted, it completes). Then `virtual-malloc-01-uaf-jumbo` began and the core
went silent: no heartbeat, and the driver read the wedge apertures at its idle limit.
- **The apertures:**
  - store unit WAIT_STORE_READY (194 = 3);
  - commit count 4 with valid bits 1101 (193 = 4, 221 = 0xd0), so one counted slot is invalid;
  - the data request valid and unacknowledged, with write_en and metadata_en (226 = 0x80, 227 = 0x05): a
    capability store outstanding;
  - a domain switch in progress (225 = 0x88: tbe, domsw), not in M-mode (224 = 0x1e), switcher index 28 (228 = 0x9c).
- **Against S-25's kit-application wedge (r10):** it differs in 8 of 15 apertures (`compare-wedge-apertures.py`).
  r10 has no valid bit at all, privM, and an idle store unit.
- **Against R-49** (a switcher write pushed over a full commit queue, the switch never finishing): the shape matches
  (WAIT_STORE_READY, count 4, a starving data request with write_en). But R-49 is fixed on silicon, and its fix
  192a5e624 is an ancestor of bf3db3175, checked with `git merge-base`. So a matching shape is not that mechanism.
- **Read so far, not a mechanism:** the invalid counted slot is R-57's ghost. A switch that then starves is what a
  full commit queue does to the switcher. The control is a use-after-free of a jumbo block, which on QEMU faults 25 at
  the probe. The attribution is handed to the RTL lane, with the apertures as recorded.
- **The cells:** NO-READING:control-failed 23, from a boot that did not complete. That is not a verdict on any case.
- **wm1 did not run** (the loop stops after a boot that does not complete). Its prediction has the same uaf-jumbo
  control. It goes last in the campaign, as the one known wedge risk.

### Arms fso, pb12, pb9, pb10, pgc, pgm0, pgm1, pre-registered (2026-10-18): the compiler lane's nb predictions

Each was registered by `register-prediction.sh`: the board's bytes and module 5bfc38f8 were checked. Every tally is its
corpus record's.
- **fso** (ffmpeg/subobject, vp2-malloc): MISSED 9. On v6-fix it was control-failed, from the pool code's tag loss, and
  `-fno-builtin` clears it.
- **pb12** (memcached/allocator, stock, vp2-malloc): CAUGHT 1, MISSED 8.
- **pb9** (ffmpeg/pool, stock, vp2-malloc): MISSED 4.
- **pb10** (ffmpeg/pool, Sublet, vp2-nested-pools): CAUGHT 4. Its pool-return control flips polarity against pb9.
- **pgc** (postgres/c-repros, vp2-malloc): CAUGHT 5.
- **pgm0** (postgres/mmgr, vp2-malloc): MISSED 5.
- **pgm1** (postgres/mmgr, vp2-pg-pools): CAUGHT 5. Its uaf-chunk control flips polarity against pgm0.
- **pb13** (memcached/allocator, Sublet) is not registered: its QEMU run is VOID (exit 75). The 11 images match, but its
  chunk-free-90 control and every fixed build fault 29 at address 0, so nothing can be scored. That is a prediction-side
  question first.
- **The order after the current seven:** fso, pb12, pb9, pb10 through `tier3-seq.sh`; then pgc, pgm0, pgm1 through
  their own runners with `--board-kit bkit7`.

### Arm mph, result (2026-10-18 13:13-13:23): 9 of 9 cells the same as the prediction

`records/mph.board.txt`. memcached/plain-heap, vp2-malloc, firmware 1b353d22aa552e5e. CAUGHT 9, and both controls
fault. compare-pilot.py reads 9 of 9 SAME by image.

### Arms sq0, sq1 (sqlite/engine-repros) and pym0, pym1 (cpython/pymalloc-repros), pre-registered (2026-10-18): the board lane's qkit7 predictions through the appvm session

Built on t3-freeze-nb (sqlite with its two extension sources, all 33 cases; pymalloc's hosted replay, 20 cases and its
control). Each was predicted with `tier3-appvm-arm.sh --virtual-kit qkit7` and registered by `register-prediction.sh`.
- **pym0** (vp2-malloc): MISSED 20, its record. 2.1 MB gzipped, one boot.
- **pym1** (vp2-nested-pools): CAUGHT 20, its record. 2.1 MB gzipped, one boot.
- **sq0** (vp2-malloc, memsys5): CAUGHT 5, MISSED 26, NO-READING:inconclusive 1, NO-READING:unattributed 1.
- **sq1** (vp2-nested-pools, memsys5 with Sublet): NO-READING:control-failed 33.
  - Both controls die with cause 29 at address 0, during the Sublet heap's setup.
  - This is v6-fix's own reading without exact bounds: 29 at the memsys5 Sublet derive, Q-15, a (iii) derive site.
    There, exact bounds returned every built cell to the record.
  - On silicon, whose bounds are inexact, the same refusal is predicted.
- **The sqlite payload:** about 28.6 MB gzipped per arm, against roughly 8.5 MB of headroom. So each sqlite arm runs as
  several boots, selected with the runner's `--only`, and each boot is compared with the prediction's cells for the
  same cases.
- **The sqlite risk:** these are the workloads whose relatives wedged Tier 2 (S-25). A wedge reads as the pre-registered
  rule says.
- **The order after the chain now running:** pym0, pym1, then sq1 (one subset boot first: a Q-15 check on silicon), then
  sq0's subsets, then wm1 last.

### Arms fph and fpt, results (2026-10-18 13:23-13:45): 25 of 25 and 13 of 13 cells the same

`records/fph.board.txt` and `records/fpt.board.txt`: CAUGHT 25 and CAUGHT 13 on silicon, by image, as predicted.
`t3-status.py` prints the campaign's state from `records/` alone.

**The wmem arms' position on bf3db3175 (2026-10-18).** The runner runs uaf-jumbo, the control wm0 wedged in, before any
case in every boot, on both arms. So neither wmem arm can be measured on this bitstream until the RTL lane attributes
that wedge, and either a software rule avoids it or the next bitstream fixes it. This is the current position, not a
result pending wm1. wm1 stays last in the queue, where a wedge costs nothing else.

### Arms fca and fcs, results (2026-10-18 13:45-14:05): the first matched pair on silicon. The same 13 carved cases are MISSED under malloc and CAUGHT under the Sublet carve, every cell as predicted

`records/fca.board.txt`, `records/fcs.board.txt`; the bundles are in `ffmpeg/carved-repros/results/2026-10-18-vp2/`.
- **fca** (vp2-malloc): MISSED 13, and both generic controls fault in the same boot. So the detector fires on this
  image, and the zero is the configuration's.
- **fcs** (vp2-nested-pools, `-DFFC_SUBLET_CARVE`): CAUGHT 13. The carve controls read as declared: live-carve
  completes, uaf-carve faults.
- **The arm is the only variable:** the same corpus, the same cases, and both readings equal the QEMU prediction cell
  by cell, by image.
- **No carve derive was refused on silicon.** The carve's children are representable under Q-15's rule, which bf3db3175
  implements as the DYN unit's vctx-gated round-trip refusal (`capstone_dyn_unit.anvil` CDERIVE, `ex_stage.sv`
  dyn_child_repr_q).

### Arms wph and wpt, results (2026-10-18 14:05-14:25): 12 of 12 and 10 of 10 cells the same

`records/wph.board.txt`, `records/wpt.board.txt`: CAUGHT 12 and CAUGHT 10 on silicon, as predicted. The bundles are in
`wireshark/plain-{heap,temporal}-repros/results/2026-10-18-vp2/vp2-malloc`.

**The first seven case arms are through:** mph, fph, fpt, fca, fcs, wph and wpt, with mpt before them. That is 8 arms
and 98 cells on silicon, every cell equal to the QEMU prediction on the same image bytes.
- **A slip in the queue, recorded so it is not repeated.** The chained waiter for fso and the pool and postgres arms
  checked `pgrep -f "tier3-seq[.]sh"`, and its own command line contained `tier3-seq.sh` (the script it was about to
  run). It matched itself and stopped before booting anything; this is CLAUDE.md's pgrep rule. No board time was lost:
  the appvm queue (pym0, pym1, sq1a) took the board next. The re-queued waiter checks only patterns its own command
  line cannot contain.

### Arm pym0, result (2026-10-18 14:25-14:36): a silicon-only DIFF. Every pymalloc program faults in its first raw allocation with cause 6 at address 0x12; QEMU completes the same bytes

`records/pym0.board.txt`. The board's collected images equalled the prediction's 21. The boot completed all 21 steps,
and compare-pilot.py reads 0 of 20 cells the same.
- **Every program faults the same way, the uaf-block control included:** `cause=6` (store address misaligned) at code
  offset 0x159c8, address 0x12, checkpoint `last=0x39`. So the control reads "none" and every cell is control-failed.
- **The site** (defect-00, `llvm-objdump` with the frozen tools): `pym_raw_malloc+792`, `sd a4, 0x10(a5)`. That is
  backing.c `b->size = n`, where `b = metadata + used` is built from the `metadata` capability global plus the `used`
  integer, stored to a stack slot by `stc`, and reloaded by `ldc` just before.
  - So on silicon `b`'s cursor is 2 at the first `pym_raw_malloc` call. That call is main.c's scratch calloc, right
    after `pym_backing_init` stored `metadata = aligned_alloc(16384, 16 MiB)`.
- **Not a mechanism:** a cursor of 2 is also boot 4's signature (a destination cursor of 2), which was the Q-26 wait
  rewrite. This module does not carry that rewrite, and a shape is not a cause. The diagnosis is handed to the compiler
  lane, which owns the port. pym1, the same backing code under the Sublet arm, is the next data point.

### Arm pym1, result (2026-10-18 14:36-14:47): the same silicon-only fault under the Sublet arm, 0 of 20

`records/pym1.board.txt`. Every program faults `cause=6 address=0x12 last=0x39` at code offset 0x15e5c of its own image,
pym_raw_malloc's `b->size` store, as in pym0. So the fault does not depend on the arm.
- The launcher's region grants match QEMU's through slot 12, the last before the fault: the same order, sizes,
  registers and permissions, including the 16 MiB metadata request and the 64 MiB arena.
- The compiler lane has a lead. aligned_alloc(16384, …) promises only 16 KiB alignment, while Q-15's granule for
  16 MiB and 64 MiB is 32 KiB and 128 KiB. The earlier silicon run used level0, which over-aligned those regions. The
  lane is building a probe that prints both regions' bases, the reloaded `metadata` and `b`, and a 4 KiB control,
  predicted on QEMU first. Not a mechanism yet: the lead does not predict cause 6 at a cursor of 2.

**wm0's wedge, attributed (the RTL lane, from bf3db3175's source, 2026-10-18): a third face of R-57.**
- 225 has dom_switch_busy set, so a switcher walk is in progress. 228 is index 28, which is x3, early in the register
  block. 226/227 show the walk's 16-byte capability WRITE outstanding: a SAVE step.
- 194 shows the store unit in WAIT_STORE_READY. 221 shows valid 1101 with no data request: the entry at the read
  pointer is the invalid one.
- So the ghost sits at the queue's HEAD, the queue never drains, and the switcher's next write is correctly refused
  forever. That is R-49's fix doing its job on top of R-57's stuck queue.
- The bit decode was checked here against the driver's own aperture labels. The fix on virtual-profile-2-c 2114e11bd
  covers it.
- **Not closed:** which store produced the ghost. The discriminator is the compiler lane's: does wmem's jumbo path make
  a capability store a fresh page's first access?
  - If yes, the wmem software rule on this bitstream is a pre-touch in the jumbo allocator.
  - If no, there is a second producer, with S-24's shape (a present, writable page with D clear) as the next arm.

### Arm sq1, subset a, result (2026-10-18 14:47-15:00): Q-15's refusal confirmed on silicon, 6 of 6 cells the same

`records/sq1a.board.txt`: sqlite/engine-repros vp2-nested-pools, cases 00-05 with both controls, compared against the
prediction cut to those cases (`subset-bundle.py`).
- **Both memsys5 controls fault with cause 29 at `capstone_cap_derive+156`,** the Sublet heap's CDERIVE. So every cell
  is control-failed, as predicted, and as the DYN unit's vctx-gated round-trip refusal says it must.
- **The fault line's address differs and is not scored:** silicon reports a non-zero address (the child's cursor) where
  QEMU reports 0. compare-pilot compares cause and symbol+offset, which are equal.
- **The other five sq1 subsets are not booted.** Every boot runs these two controls first on the same image set, and
  their refusal is now read on silicon, so each subset could only repeat control-failed. That is five boots of no
  information. sq1's record is therefore 6 of 33 cells measured, with the cause of all 33 confirmed. The arm becomes
  measurable once the Sublet memsys5 heap derives representable children (Q-15's consequence, the compiler lane's
  work).

### Arm fso, result (2026-10-18 15:01-15:12): pymalloc's silicon-only fault, in a second port

`records/fso.board.txt`. ffmpeg/subobject vp2-malloc reads NO-READING:control-failed 9, against a predicted MISSED 9.
Both generic controls pass in the same boot (28 and 25).
- **Every case's FIXED run faults:** `cause=6 address=0x12` at `sd a4, 0x10(a5)` after the same `stc`/`ldc`
  spill-reload. That is metadata-allocator.c's `b = (struct block *)(arena + used); b->size = size`.
- **The same idiom as pymalloc's backing.c:** a static capability global set once by an init function from a large
  allocation, an adjacent static `size_t used`, and `b = global + used` reading cursor 2 at the first allocation.
- **A lead, not a mechanism (the compiler lane's probe tests it):** the init's capability store may be a fresh .bss
  page's first touch, which is S-25's trigger. A pre-touch arm will tell. The RTL lane is asked whether R-57 also
  predicts the reloaded value 2.

### Arm pb12, result (2026-10-18 15:12-15:22): the same silicon-only fault, in a third port

`records/pb12.board.txt`. memcached/allocator, stock (vp2-malloc), reads control-failed 9 against a predicted CAUGHT 1,
MISSED 8. The board has 19 faults of `cause=6 address=0x12` (the chunk-free-90 control and every case) and QEMU none.
Both generic controls pass. That is three ports: pymalloc, ffmpeg's pool/metadata allocator and memcached's
allocator.
- **The RTL lane's reading, from bf3db3175's source: R-57 is excluded.** R-57's ghost stops the commit queue from
  ever draining, so the very next store of any kind would stall forever. These programs run on and fault at the
  next instruction.
  - CORRECTED (2026-10-18, the RTL lane's retraction): this section first gave two more legs, that "a lost STC
    leaves zero on fresh .bss" and that a ghost never writes. The first is VOID: it assumed the reloaded granule was
    the .bss global, but the reload is of a STACK spill slot (below). The exclusion rests on the one mechanical leg
    above.
- **The directed simulation (the RTL lane):** drop only the spill `stc` from the ports' exact sequence, with the
  stack slot seeded with 2. The reload then reads a non-capability, and `sd n, 0x10(b)` faults cause 6 at address
  0x12, value for value the ports' reading.
  - So the defect is a capability spill store that did not land. The reload returns the slot's previous contents,
    and a 2 in live stack data needs no explanation.
  - With the store present, the sequence is correct in simulation, with and without the intervening load.
  - The RTL lane's next hypothesis, from source: a committed store dropped by a flush (`store_unit.sv:283` gates the
    push on `!flush_i`, and `:351` returns the unit to IDLE on a flush). In a virtual context flushes are frequent,
    and nothing wedges.
- **The compiler lane's probe** reads the granule back raw (both halves with plain `ld`, plus the tag) before and after
  the init, with and without a plain pre-touch. Pre-registered go/no-go: "the pre-touch changes the reading".

### Arm pb9, result (2026-10-18 15:22-15:30): 4 of 4 cells the same, with no cause-6 fault

`records/pb9.board.txt`. ffmpeg/pool, stock (vp2-malloc): MISSED 4 on silicon, as predicted, with both generic controls
faulting. Its prebuilt CMake build of the pool sources reaches no cause-6 fault, so the silicon-only fault is not
common to everything built from these sources. It is the contrast case for the probe.
- Checked with `llvm-nm`: pb9's images define `av_malloc` but not `ff2_memory_init`. The stock build's allocator
  does not use the static arena, whereas fso's images define both. So the fault tracks the static-arena idiom:
  consistent with it, and not yet a mechanism.

### Arm pb10, result (2026-10-18 15:30-15:42): a kernel panic on a load refused under a covering region, S-23's data face (the RTL lane's attribution)

`records/pb10.board.txt`. ffmpeg/pool, Sublet (vp2-nested-pools).
- **What ran:** case 00 read CAUGHT, as predicted. Case 01's FIXED run returned 0. In its BUGGY run the monitor printed
  its region dump (29 regions), then `MTVL:BC2C0825 CPMX:00000002`. Linux took `Oops - load access fault` at
  `mod_node_page_state+0x40` (cause 5, badaddr = mtval = 0xbc2c0825) and panicked. Cases 02 and 03 did not run.
- **The covering region:** decoding the dump, region 21 `[0xac100000, 0xc0000000)`, RWX, is loaded in CPMP slot 7 and
  covers 0xbc2c0825. So a kernel data load was refused under a covering executable entry.
- **The attribution (the RTL lane, from bf3db3175):** S-23's data face.
  - The fetch check (`pmp_data_if.sv:239-246`) and the data check (`:300-310`) are one comparator over one decoded
    array. Both go live when the escape returns the privilege to the caller (`csr_regfile.sv:2124-2127`), while the
    RESTORE walk is still writing the CPMP ids.
  - The fetch side is guarded by kill_s1 and flush_if for the whole busy window. The LSU keeps a dom-switch exemption
    from that flush, R-50's subject, which 429c60b32 narrowed but did not remove. So the data side can show this where
    the fetch side rarely does.
  - The monitor's fetch-retry (VP2_FETCH_RETRY) cannot cover a data access.
- **Caveat:** CPMX is the monitor's own print from swap_cpmp and cannot name the entry the comparator used. Region 21
  covering mtval is this lane's decode of the dump.
- **The post-panic apertures are not a reading:** a commit count of 7 in a 4-deep queue, and 0xff fields.
- **A candidate workaround, not adopted:** a data-side retry in the monitor. The RTL lane's conditions: S or U origin
  only; a loaded entry covering mtval for the access type; the same bounded no-progress guard; inject as today when
  nothing covers it; never for AMOs or LR/SC.
  - Its prediction: if the window is the mechanism, a data retry succeeds on its first attempt.

### The pymalloc probe, pre-registered (2026-10-18): three arms, ppA, ppB and ppC, each one boot, before the sqlite subsets

These are the compiler lane's images, verified against the hashes it sent. Case 00 is the probe in each hosted build;
cases 01-19 are the production images, as a built-in control. All three run `pym_backing_init` and
`pym_allocator_init` exactly once, in production order, and print nothing until every value is captured. Then they
print, and fall through into the real path. So the original cause-6 fault, if it recurs, is the control that the
probe did not perturb the bug.
- **ppA (faithful):** the global's granule as two plain loads, `used` as a plain load, and the capability via `lcc`
  (type, base, end, cursor). Read right after the init's store (v1) and after allocator_init (v2).
- **ppB (pre-touch):** ppA, plus one plain byte store into the globals' page before the init.
- **ppC (pre-read):** ppA, plus the granule and `used` read with plain loads BEFORE the init: the previous contents.
- **The prediction:** the QEMU twins on qkit7 (`records/pp{A,B,C}.prediction.txt`, whose probe lines are copied in).
  Every value is right: raw lo equals the 16 MiB region, its end minus base is 16 MiB, `used` is 0, ok v1=1 v2=1, the
  before-init values are zeros, the regions are granule-aligned, and MISSED 20.
- **How the silicon reading names the class** (the compiler lane's table, with the RTL lane's go/no-go):
  - ppC's before-init lo, hi or `used` non-zero, especially 2: stale contents before the init (the stale-refill class).
  - raw lo right but the capability's cursor wrong: the capability load or decode.
  - raw lo right in v1 but wrong in v2: forwarded, then lost.
  - `used` = 2: the other global is the corrupted one.
  - ppB differing from ppA: the first-touch class. ppB the same as ppA: not first-touch.
  - The regions not granule-aligned on silicon: the representability lead is live.
  - All ok AND no fault in all three: the probe perturbed the trigger. That is a null result, not a pass.

### Arms pgc, pgm0 and pgm1, results (2026-10-18 15:41-16:08): 15 of 15 cells the same, and a second matched pair on silicon

`records/pg{c,m0,m1}.board.txt`.
- **pgc** (postgres/c-repros, vp2-malloc): CAUGHT 5.
- **pgm0** (postgres/mmgr, vp2-malloc): MISSED 5.
- **pgm1** (vp2-pg-pools): CAUGHT 5. The same five memory-context cases are missed under malloc and caught under the
  pools configuration, and each arm's own controls fire with the polarity it declares.

### Probe arm D, pre-registered (2026-10-18): a fence after the capability spill

`pym-probe-hosted-D` (the compiler lane's image, ebc820195bd37f93, verified). pym_raw_malloc is exactly production's plus
one `fence rw, rw`, at 0x259ac between the spill `stc` and the reload `ldc`. The compiler lane checked this over the
whole function. D carries ppA's readouts and falls through to the real path.
- **The prediction:** its QEMU twin reads MISSED 20 with every probe value right (`records/ppD.prediction.txt`).
- **The reading on silicon, if a lost spill store is the mechanism:** ppA faults at pym_raw_malloc and ppD does not. The
  fence would then be a working software rule for -O0 corpus code on this bitstream.
- **If ppD still faults:** the fence does not order the lost store, which is information for the RTL lane's next arm
  (an escape or switcher flush).

**A provenance fact for the frozen SDK (the compiler lane, 2026-10-18).** t3-freeze's runtime archive
(libapplication-runtime.a d10e4e75f12f601a) was built from the v9 kit's tree at f6dfd19d9396. That tree's vm-abi.h makes
virtual ABI 6 current, and every Tier 3 image carries the magic `CPONVVM6`. dev's vm-abi.h defines only ABIs 2, 3 and
5, and of all remote refs only the compiler lane's branch carries ABI 6. So the campaign's runtime revision lives on that
branch, and an SDK built from dev makes ABI-5 images, which the kit's launcher refuses (`virtual ABI: Exec format
error`, exit 126).

### ppA on silicon (2026-10-18 16:08-16:19): the global is right; the fault recurs in the real path

The probe's readings on silicon:
- `metadata`'s raw low word equals the 16 MiB region, and the capability has the right base and end with its cursor at
  the base.
- `used` is 0. ok v1=1, v2=1.
- Both regions are granule-aligned (mod 32K = 0, mod 128K = 0), so the representability lead is dead on silicon too.
- Then the original fault, `cause=6 address=0x12`, recurs at the same pc in the real path.

So the init's store to the global landed, and the lost store is elsewhere. With the RTL lane's directed simulation,
that is the stack spill.

### The pymalloc probe, results (2026-10-18 16:08-16:51): the global is intact, the stack spill store is lost, and a fence after it does not help

`records/pp{A,B,C,D}.board.txt`. The images equalled the predictions' 21 in every arm.
- **ppA, ppB and ppC read identically.**
  - The global's granule is right: the raw low word is the 16 MiB region; the capability's base, end and cursor are
    right; `used` is 0; both regions are granule-aligned.
  - ppC reads zeros before the init.
  - Then the real path faults `cause=6 address=0x12` at pym_raw_malloc's `sd a4,0x10(a5)`, as in production.
  - So the init's store landed, the plain pre-touch changes nothing, and the lost store is the stack spill, which the
    RTL lane's directed simulation reproduces value for value.
- **ppD, the same function plus one `fence rw, rw` between the spill `stc` and the reload `ldc`: still faults.** The
  fault is at pc offset 0x159cc, production's plus 4, so the fence was in the executed code. All 21 programs fault the
  same way.
  - A fence drains every pending store before the reload, and the reload still reads the slot's old contents. So the
    capability store never reaches memory, or lands elsewhere. A fence is NOT a software rule for it.
- **Open, with the RTL and compiler lanes:** the RTL lane's escape and switcher-flush arms, and a fresh-stack-page
  first-touch arm. -O0 frames spill to s0-relative slots, and a capability store as a fresh page's first access is
  S-25's trigger, though R-57 as analysed cannot lose a store without a wedge. A software arm E, pre-touching the
  stack, would split those.

### RETRACTION (2026-10-18 17:20): "the capability spill store is lost; the reload returns the slot's previous contents" is refuted for the probe arms

This README stated, in the pb12 section's correction and the probe results above (lane commits 896b1ccc6b2c and
8852ebcaebdd), that the defect is a capability spill store that did not land, so the reload returns the stack slot's
previous contents, and that "a 2 in live stack data needs no explanation". A claim-auditor refuted that for ppA-ppD:
- **The slot's last contents are known.** In every probe image, `b`'s slot (pym_raw_malloc's `s0-0x70`, `DW_OP_fbreg
  -112`) is [main_sp-0xc0, main_sp-0xb0), exactly the probe's locals `au`/`am`. pym_probe_addrs writes them with plain
  `sd`, and nothing writes there again before the `stc`.
  - Their values are printed on the board, e.g. ppA's `&used=0x3f8dae3d20`.
  - A dropped `stc`, or a reload of the last contents, would put a5 = &used and fault at &used+0x10, which is
    8-aligned. The board reads `address=0x12` in all four arms.
- **So where the 2 comes from is NOT established.** The directed simulation seeded the slot with 2 to match the board,
  so it shows only that a non-capability 2 in a5 gives this fault. It does not explain the probe arms.
- **This holds if the faulting call is the first pym_raw_malloc, which nothing measures.** If a later call faults,
  "first block" must come out instead.
- **What stands:** the four ports' readings, the shared code sequence (the auditor disassembled every distinct
  faulting image and pc: always `sd a4,0x10(a5)` after the same `stc`/`ldc` of `s0-0x70`), image identity with the QEMU
  twins, the global being right at init, and ppD's fence not preventing it. "Never reaches memory" is withdrawn with
  the mechanism.
- **Also corrected:** "QEMU runs the same bytes clean" means no cause-6 fault and no address 0x12. The twins do fault,
  with the arms' own detections and controls.
- **The open hypotheses (the auditor's):**
  - (a) the reload returns a version of the line older than its last plain stores;
  - (b) a value not from this slot at all;
  - (c) b is already 2 at the `stc`;
  - (d) a later call faults, not the first.
- **The settling arms, one batched boot (asked of the compiler lane):**
  - a marker after `pym_raw_calloc`, pinning which call faults;
  - a sentinel plain `sd` into `s0-0x70` right before `b = …`: sentinel+0x10 means a dropped store, still 0x12 means
    not the slot;
  - a readout of the slot halves, the global and `used` with plain loads when b's cursor is not metadata + used;
  - ppA as the control.
  - The RTL lane adds a 64-byte dump around the slot, to catch a store landing at the wrong offset.
- **What would have caught it:** the frame-layout check the auditor did. The probe's own locals overlap the spill slot,
  so the "previous contents" the mechanism predicts were on the board all along.

**All Tier 3 boots are UNGATED by the driver's C-gates.** The recipe runs the baked-rung driver with PREFLIGHT=0, as
every vp2 Tier 1/2 boot does, and the driver prints "Any verdict from this boot is UNGATED -- say so when reporting
it". The substitute checks are the campaign's own:
- every image equal by sha256 to its prediction's (register-prediction.sh, EXPECT_IMAGES);
- every fault line's sha256 equal to a staged image's;
- the firmware cap;
- the recipe's monitor and driver pins.

### Arm sq0 subsets a and b, results (2026-10-18 16:51-17:16): a is 6 of 6; b is 1 of 6, a fourth port with silicon-only faults including the value 2

- **sq0a** (cases 00-05): 6 of 6 cells the same as predicted.
- **sq0b** (cases 06-11): 1 of 6. QEMU completes all six on the same bytes. On silicon, all at -O0 SQLite code:
  - 06 and 10 fault `cause=24 address=0x2`, an untagged capability with cursor 2 dereferenced, at idxLargestIndex+232
    and sqlite3AuthCheck+112;
  - 07 and 08 fault 28 (bounds) at sqlite3VdbeMakeReady+1776 and sqlite3VdbeAddOp3+504;
  - 09 faults 29 at freeSpace+48.
- The value 2 as an untagged capability is the allocators' value and boot 4's (cause 24 at contract.dom's memset, a
  destination cursor of 2). This is an observation, not a mechanism: the open question is the same one as the
  allocators', recorded above.

### The settling batch, pre-registered (2026-10-18, before its boots): ppA2, ppE, ppE1, ppE2, ppE3, ppE4, scored against a named candidate

**The candidate** (the RTL lane, from bf3db3175's source and its simulation arm va-rdclob, fb3f87b6c; not yet shown on
silicon):
- Every RESUME of a virtual context writes the literal integer 2 into the CONTEXT's x(rd), where rd is the trusted
  caller's CSRUNV destination register.
- `commit_stage.sv:610-624` builds one request for entry and resumes, with `ra_data` the bad-frame refusal's kind 2.
  The switcher's finish() writes it for modes 3 and 4, which it does not exempt.
- The module's resuming call site (capstone_vm.c:1600, `.text+0x40c8` in 3005ed52) has rd = x15 = a5, by the compiler
  lane's decode.
- **Supporting, from this lane's auditor:** every 2-fault read directly has a5 = 2.
  - The allocators' `sd a4,0x10(a5)` is the first store into the fresh metadata region. It page-faults, and is replayed
    after the module resumes the context.
  - SQLite cases 06 and 10: a5 goes from a valid capability to 2 with no instruction writing it. There the resume is
    not at the fault point, which suggests an asynchronous escape. N=1 each.
  - On silicon, a cause-24 fault's tval is rs1's value (`ex_stage.sv:1176-1185`, `cva6.sv:1522`), so "address 0x2"
    reads "a5 = 2".

**Predictions under the candidate, written before the boots:**
- **ppA2** (ppA re-run, the control): as ppA. The global is right, then the fault recurs at 0x12.
- **ppE** (stack pre-touch): STILL faults at 0x12. The fresh page is the metadata mapping, not the stack.
- **ppE1** (a marker after the first pym_raw_calloc): no marker. The first call faults.
- **ppE2** (a sentinel in b's slot): STILL faults at 0x12, not sentinel+0x10.
- **ppE3** (a plain-load compare of b): NO mismatch, and the fault anyway. The slot holds the right b; a5 is overwritten
  after the reload, by the resume.
- **ppE4** (metadata a LINEAR grant): the first fault at 0x12, NOT 0x11. The 2 is the resume's literal, not a type
  field. Scored on the FIRST fault only: E4 breaks linearity by construction, so its QEMU twin faults later too.

**A reading that refutes it:** E or E2 not faulting, E3 printing a mismatch, E4 reading 0x11, or the marker printing.

### Arm sq0 subsets c and d, results (2026-10-18 17:16-17:42): 2 of 6 and 1 of 6, the same register-2 shape

- **sq0c:** `cause=24 address=0x2` twice and `cause=29 address=0x2` once.
- **sq0d:** cause 24 and 29 at 0x2, twice each, and one cause 13 at an address made of ASCII bytes (a string read as a
  pointer).
- The candidate's source citation was checked here at bf3db3175:
  - `commit_stage.sv` builds the CSRUNV request with `set_ra = rd` and `ra_data` the literal 2, with `runv_mode` 3 or 4
    on a resume;
  - `capstone_dom_switcher.anvil:584` writes it whenever `set_ra != 0`, unless the request is a good mode-1 entry.
- S-27 records all this in ISSUES.

### Arm sq0, subsets e and f, and the arm in full (2026-10-18 17:42-18:05): 18 of 33 cells the same

- sq0e reads 5 of 6 and sq0f 3 of 3.
- Over the six subsets, sq0 (sqlite/engine-repros vp2-malloc) reads 18 of 33 cells equal to its prediction. Of the
  other 15, 14 carry S-27's register-2 fault, and case 22 differs only in its fault cause, on both modules. (This line
  first said all 15 carried S-27; see the RETRACTION in the sq0b-e workaround section below.)
- **The bundles:** each subset ran on its own firmware (its own stage), so the arm is filed as six verdicts-v1 bundles,
  `sqlite/engine-repros/results/2026-10-18-vp2/vp2-malloc-sq0{a..f}`, and sq1's single subset as
  `vp2-nested-pools-sq1a`. Each carries its own platform record. Combining them is the fold's decision, and corpus.json
  is untouched.

### Arm wm1, result (2026-10-18 18:05-18:16): a wedge in the first control, the same face as wm0

`records/wm1.board.txt`. wmem's nested-pools arm ran session-share, then wedged in `virtual-nested-pools-00-uaf-chunk`,
its first control.
- Its apertures equal wm0's in 14 of 15 switch fields: count 4, WAIT_STORE_READY, a capability write unacknowledged,
  the switcher at index 28, not in M-mode.
- Only 221's valid bits differ (0111 here, 1101 in wm0), so the invalid entry sits in another slot. The same R-57 face.
- **Note, for after the S-27 workaround:** with every resume clobbering the context's a5 on this module, a re-run of
  wm0 and wm1 on the patched module will show whether these wedges were downstream of S-27. They stay attributed as
  recorded until then.

**Amendment to ppE4's pre-registration (before its boot).** Its QEMU twin (this lane's, agreeing with the compiler
lane's) reads 19 MISSED and 1 NO-READING:setup-fault: case 00 faults `cause=24 address=0x40` in memset+0x2c, in setup,
before the `sd` it was built to discriminate. That is E4's deliberately broken linear `metadata` going through memset.
So ppE4 cannot separate 0x11 from 0x12. Its prediction is now the QEMU twin's own reading
(`records/ppE4.prediction.txt`), and a difference on silicon is what gets read. ppE3 is unchanged: no mismatch, and the
fault anyway.

### ppE3 on silicon (2026-10-18 18:39-18:50): no mismatch, and the fault moves to the next use of a5, which reads short of the 8 written into it

`records/ppE3.board.txt`.
- **No MISMATCH line:** the plain-load compare found the slot holding the right `b`. The global's readouts are right.
- **The fault moved:** `cause=28 address=0x3f870e40b0` (bounds) at pc offset 0x15cb4, and that address is exactly
  `&used` (the probe printed `&used=0x3f870e40b0`). From E3's disassembly:
  - `li a5, 0x8` at 0x25c64 sets the size of `used`. Nothing writes a5 again before 0x25cac.
  - `sd a2, 0x10(a4)` at 0x25c90 is `b->size = n`. In E3's register allocation `b` is in a4, not a5. It is the first
    store into the fresh metadata page, so it is expected to page-fault and be resumed by the module. That is
    inferred: this boot printed no escape lines and read no stats.
  - `add a5, a4, a5` then `shrink a2, a4, a5` (0x25cac, 0x25cb0) build bounds of a5's length over `used`. The 8-byte
    `ld a4, 0(a2)` faults on bounds at the window's base, `&used`.
  - **That gives a5 < 8 at that point**, where 8 was written and nothing writes a5 in between. A length of 2, the
    resume's literal, fits, but this reading does not fix the value. The exact 2 is measured by the allocators'
    0x12 faults (tval = 2 + 0x10).
- **So moving `b` out of a5 moved the symptom to the next use of a5, whose value changed with no instruction writing
  it.**
  - The register is clobbered, not the stack slot, since the slot was right. That is what the candidate (every resume
    writes 2 into x(rd), and the module's resume rd is x15) predicts, and what the "lost spill store" reading cannot
    produce.
  - E3 does not separate the rd write from other register-clobbering mechanisms; the source and the simulation arm do.
- **RETRACTION (2026-10-18, found by a claim-auditor before S-27's status line was written):**
  - This section first said a5's value "8 becomes 2" and "its value is 2". A bounds fault at the window's base gives
    only a5 < 8.
  - The text above replaces that wording. The first version is on dev at 4303f2993d57, and on the lane at
    7dfa3c19999d, whose commit subjects repeat it.

### ppE4 on silicon (2026-10-18 18:50-19:01): the first fault at 0x12, not 0x11; the amended prediction differs in all 20 cells

`records/ppE4.board.txt`.
- **The reading:** all 21 virtual steps (the control and the 20 cases) fault the same way:
  - `cause=6 address=0x12` at pym_raw_malloc+792;
  - that is `sd a4, 0x10(a5)` at 0x259c8 in image bcb51ed62c66d12e, right after `ldc a5, 0(a0)` reloads `b`.
  - Every cell scores NO-READING:control-failed.
- **The twin, by contrast:** its control completes, and case 00 faults `cause=24 address=0x40` at memset+0x2c.
- **Both faults come after the same 14 CAPSTONE_VM_ADD_OK grants** (slots 0-13, with the 8 MB, 16 MB, 64 MB and 16 MB
  linear grants), so silicon REACHED the `sd` the arm was built to discriminate. The value it read is 0x12: the
  resume's literal 2, not a type field (0x11).
- **The amendment above was wrong.** It said ppE4 "cannot separate 0x11 from 0x12", reasoning from the twin's setup
  fault. It did separate them, and the reading is the original pre-registration's (0x12).
  - The operative prediction was the amended one (the twin's reading), and silicon differs from it in every cell.
  - The 0x12 reading is therefore scored as an observation, not as a pre-registered confirmation.
- **Not settled:** whether the twin's memset comes after this `sd` in program order (calloc's memset of the new block,
  backing.c:90-92), or whether silicon passes a memset that QEMU faults in. A backtrace on the twin's fault would settle
  it. The reading does not depend on it.
- **Not booted:** ppA2, ppE, ppE1 and ppE2 of the settling batch. E3's reading and the paired boot on the S-27
  workaround module (next) answer what they were built for. ppE's QEMU twin is filed as `records/ppE.prediction.txt`.

### The S-27 paired boot, pre-registered (2026-10-18, before its boots): the workaround module against 3005ed52's records

**The module:** d71f207a3af5546b, from the compiler lane's e5775346cc8e (origin/module-resume-rd-x0).
- The entry/resume CSRUNV is issued with rd = x0, so nothing is written into the context's x(rd).
- The event kind is read back from frame +16. On bf3db3175 every ESCAPE writes it there: the EVENT walk saves CSR id 70
  at +16 (`capstone_dom_switcher.anvil:466-467`). A REFUSE writes {2, 29} at +16 and +24 itself (`:424-429`).
- The kit gates, re-run here by stage-module.sh: qemu-only-csr-census 0 of 4,762 instructions; vp2-silicon-gate as
  trusted_S: 0 refused, 0 candidates, 4 conditional.

**Boot 1, Tier 1** (`board-tier12/boot3/run-tier12-wa1.sh`):
- Firmware ad5e84f761dd0f7c (Image e4b6e7535a0a2a77; monitor 007d81d with the S-23 fetch retry; wrapper 1269c77).
- The stage is ap1's with one line changed, the module, and the bake's cpio check finds d71f207a by hash.
- Reference: ap1 (firmware 83e5c5c6d6f89471, module 3005ed52).
- **Predicted:**
  - b0-stats, b0-hello and vm-contract as their oracles (0, 0, 0);
  - vm-contract's identity branch: VIRTUAL_APPLICATION_OK, then the stale access denied with 25 (exit 139);
  - the banner once, and 0 "wrote no event record" lines.
- **Readings against the module on silicon:** any refusal line (the +16 record not written), vm-contract off its oracle,
  or a wedge.

**Boot 2, pym0** (`tier3-pair-wa1.sh`):
- It runs only if boot 1 reads as predicted.
- Kit bkit7-fix6 is bkit7 with the module swapped (`diff -r`: the module, and board.env's manifest path, whose content is
  unchanged).
- The same runner arguments and the same 21 images (`t3-images-nb/pym0.images.SHA256SUMS`) as pym0 on bkit7.
- **Predicted:** the QEMU verdict.
  - Control uaf-block completes and the 20 cells read MISSED, equal to `records/pym0.prediction.txt` (this lane's twin
    on 5bfc38f8).
  - The same holds against the compiler lane's twin on the new module's QEMU build, 28882779c95b94ad (`t3-pym0-fix6`,
    20 of 20 identical to the record).
- **Reference on 3005ed52:** `records/pym0.board.txt`, 0 of 20, every cell control-failed by cause 6 at 0x12.
- **The pair is two-sided if pym0 reads 20 of 20 here.**
- **Readings that are not that, named in advance:**
  - the 0x12 fault again: another resume site writes x(rd);
  - a different fault: a second defect, unmasked;
  - an S-mode oops at a collect site: pym0 collected, and this kit has no stopgap;
  - cause 0xcafe0001 on a fault line: a refused CSRUNV.
- **CV_STATS is not read on the board in boot 2.** The batch session gives each step only the runner's environment, and
  adding CAPSTONE_VM_STATS would make the pair differ in two things. pym0's collection count comes from the compiler
  lane's QEMU census.

Both boots are UNGATED (PREFLIGHT=0); image identity by sha256 is the substitute. The dry run is `dryrun-pair-wa1.txt`.

### fso, pb12 and pb10 on the workaround kit, pre-registered (2026-10-18, before their boots), behind the pair

`tier3-fix6-seq.sh` starts when the pair ends. It runs only if pym0 read 20 of 20 on bkit7-fix6. Otherwise it HOLDS
and the pair's RESULTS go to the runtime lane first. Each arm is its ARMS.spec line under the tag `<arm>-fix6`, through
`tier3-seq.sh` on bkit7-fix6. Each is scored against the same prediction bundle as its 3005ed52 run (the compiler
lane's nb record, `t3-<arm>/bundle`). The compiler lane's census reads 0 collections on all three (qkit7-fix6), and its
gate reads them identical per case on the two modules' QEMU builds.
- **fso-fix6** (ffmpeg/subobject, vp2-malloc): MISSED 9 with both controls as predicted. On 3005ed52 it read 0 of 9,
  control-failed by S-27's 0x12 fault at the allocator's `sd`.
- **pb12-fix6** (memcached/allocator, vp2-malloc): CAUGHT 1, MISSED 8. On 3005ed52 it read 0 of 9, the same S-27 face.
- **pb10-fix6** (ffmpeg/pool, Sublet, vp2-nested-pools): CAUGHT 4, last in the boot order.
  - On 3005ed52 it read 1 of 4: case 01's BUGGY run ended in a kernel panic on a load refused under covering CPMP region
    21 (S-23's data face, the RTL lane's attribution).
  - The module change does not touch that path, so S-27's workaround is NOT predicted to change it.
  - A recurrence is S-23's data face at N=2. A clean 4 of 4 is one boot without the window, not a fix.
- Each record names kit bkit7-fix6 and module d71f207a3af5546b. The bundles are filed under
  `results/2026-10-18-vp2-rdx0/<arm>`, apart from 3005ed52's.
- UNGATED (PREFLIGHT=0), as every Tier 3 boot. Dry run: `dryrun-fix6-seq.txt`.

### pym1, sq0b-e, wm0 and wm1 on the workaround kit, pre-registered (2026-10-18, before their boots), after fso/pb12/pb10

`tier3-fix6-appvm.sh` runs after `FIX6SEQ DONE` (a HOLD or STOP there holds it too).
- Each arm is one boot through `tier3-appvm-arm.sh` on bkit7-fix6, with its 3005ed52 run's runner arguments.
- Its collected images must equal its `t3-images-nb` list before the boot. pym1's and wm0's arguments were also checked
  by a collect-only run.
- Each is scored against its 3005ed52 run's prediction, under the tag `<arm>-fix6`. The bundles are filed under
  `results/2026-10-18-vp2-rdx0/` (sq0's as `vp2-malloc-sq0<x>`).
- The compiler lane's census reads 0 collections on every one of them (qkit7-fix6).

**Predicted** (the 3005ed52 reading in brackets):
- **pym1-fix6** (vp2-nested-pools): CAUGHT 20. [0 of 20, S-27's 0x12 fault in every step.]
- **sq0b-fix6 to sq0e-fix6** (vp2-malloc, six cases each): the sq0 prediction cut to each subset (subset-bundle.py).
  [1, 2, 1 and 5 of 6; the rest carried S-27's register-2 faults.]
- **wm0-fix6** (vp2-malloc): MISSED 23. [A wedge in the uaf-jumbo control, R-57's face per the RTL lane.]
- **wm1-fix6** (vp2-nested-pools): NO-READING:control-failed 23, the QEMU twin's own reading. [A wedge in its first
  control, wm0's face.]

**What would re-attribute the wedges:** if wm0 and wm1 now reach their predictions, the wedges were downstream of S-27,
and R-57's attribution for them is withdrawn. If either wedges again with the same apertures, R-57's attribution stands
at N=2.

UNGATED (PREFLIGHT=0). Dry run: `dryrun-fix6-appvm.txt`.

### The S-27 paired boot, results (2026-10-18 19:14-20:12): both boots read as pre-registered; pym0 reaches its QEMU verdict on d71f207a

**Boot 1, Tier 1** (`board-tier12/results/board-vp2-tier12-wa1.result-lines.txt`):
- b0-stats, b0-hello and vm-contract read as their oracles, and vm-contract takes the identity branch.
- "REFUSED-CSRUNV (must-fire) 0; banner 1".
- Every line equals ap1's on 3005ed52, including the FRTY address and vm-contract's fault offset.

**Boot 2, pym0 on bkit7-fix6** (`records/pym0-fix6.board.txt`):
- 21 of 21 steps ran, and the boot completed. Control uaf-block completed, and the 20 cases read MISSED.
- That is 20 of 20 cells the same as `records/pym0.prediction.txt`, and 20 of 20 against the compiler lane's twin on
  the new module's QEMU build (28882779c95b94ad).
- The session shows 0 domain fault lines, the banner once and 0 refusal lines.
- Each program now runs for minutes, where on 3005ed52 each faulted within seconds; the boot took 52 minutes against 11.

**The pair:**
- The 21 staged `.dom` files are sha256-identical, file by file, between this boot and pym0's on 3005ed52. The runner
  (fdc271ee3c107951), launcher (b27afefc09c6cae6) and monitor (007d81d) are the same.
- The firmware and kernel hashes differ, by the bake of the other module into the initramfs. The kits differ in the
  module alone (`diff -r`).
- **On module 3005ed52e860c7af: 0 of 20, every step control-failed by cause 6 at 0x12. On module d71f207a3af5546b: 20
  of 20 at the QEMU verdict.**
- The module change is the resume CSRUNV's rd = x0, together with the sentinel store and the fence before it (now
  unconditional; on 3005ed52 it was a parameter, off) and the kind read from frame +16. The pair confirms the
  workaround.
- That the rd write is the mechanism rests on the RTL source (bf3db3175's finish() and commit_stage's set_ra / ra_data)
  and the RTL lane's simulation arm. On silicon it is not yet separated from the fence or the sentinel.
  - The cheap discriminator is 3005ed52 with `fence_before_entry=1` on pym0's images: 21 of 21 at 0x12 if the rd write
    is the cause.
  - (This sentence first credited E3 as well. See E3's retraction above.)
- How S-27's status is worded in ISSUES waits for a claim-auditor pass.

**A slip, caught before it was reported:** a live read of the driver log at 19:45 showed `cause=25` at code offset
0x140f0. It looked like a fault in case 05 (disassembled there: an `ldc` in insert_to_freepool). It was the previous
boot's console, replayed on connect: Tier 1's vm-contract, whose identity branch is exactly that stale access, at that
offset. Scoping the read to after the session's first marker removed it. The board-run skill already says to scope a
live scan to the run's own load.

### The S-27 fence discriminator, pre-registered (2026-10-18, before its boot): 3005ed52 with fence_before_entry=1 on pym0

**Why:** the claim-auditor found that d71f207a's resume site differs from 3005ed52's in four ways, so the pair does not
separate rd = x0 from the fence on silicon. The other three are rd x15 -> x0, the kind read from +16, and a ~0 sentinel
at +16. On 3005ed52 the fence exists behind the `fence_before_entry` parameter, off in bkit7.

**The arm:** kit bkit7-fence is bkit7 with `adapter/module/params` = `fence_before_entry=1` (`diff -r`: the params file,
and board.env's manifest path, whose content is unchanged). It runs pym0's 21 images with the same runner arguments, in
one boot after the fix6 appvm chain (`tier3-fence-arm.sh`). The batch refuses a boot whose module did not read the
parameter back.

**Predicted (the RTL lane's mechanism):**
- 21 of 21 steps fault `cause=6 address=0x12` at pym_raw_malloc+792, as pym0 read on 3005ed52;
- 20 of 20 the same against that board bundle; 0 of 20 against the QEMU prediction;
- "@@PARAM fence_before_entry=Y" once.

**Readings:**
- That reading rules the fence OUT as what cures S-27. It does not separate rd = x0 from the sentinel or the +16 read,
  though the mechanism names rd. The RTL lane's va-rdclob, which has no fence, carries that.
- 0 faults would mean the fence alone suffices, and the rd account is reopened.

UNGATED (PREFLIGHT=0). Dry run: `dryrun-fence-arm.txt`.

### Amendment (2026-10-18 20:40, before fso-fix6 ends): fso-fix6's batch budget is too short; fso, pb12 and pb10 re-run on a scaled budget

- **The budget:** `run-virtual-cases.py` gives a board batch 300 + 60 s per step (:235/:554). The kit recipe passes
  that as the driver's per-rung wait (BAKED_TIMEOUT). For fso's 20 steps that is 1500 s.
- **Why it runs short:** on 3005ed52 every step faulted within seconds. On d71f207a they run to completion, about 1.9
  min a step here (9 steps in the rung's first 17 min).
- **So fso-fix6's batch rung (from about 20:19:40) is cut at about 20:44:40, with about 13 of 20 steps read.** The
  steps it reads are valid readings. The rest score NO-READING:infra, a budget void, not a module reading.
- **Consequences:** tier3-seq.sh stops after a boot that does not complete, so pb12-fix6 and pb10-fix6 do not run.
  The appvm chain is unaffected: its sessions budget at least 300 s a step (pym1 6600 s, sq0b 2700 s, wm0 15900 s).
- **The re-run** (`tier3-fix6b-seq.sh`): fso, pb12 and pb10 under tags `<arm>-fix6b` on kit bkit7-fix6t, after the
  fence arm.
  - bkit7-fix6t is bkit7-fix6 with `BOARD_STEP_S=300` in board.env. board-run.sh scales the per-step part, giving
    300 + 300 s per step. The recipe's stub mode reads 1500 vs 6300 for 20 steps.
  - The module, launcher, runner (hash unchanged) and images are unchanged.
  - The predictions are unchanged (fso MISSED 9; pb12 CAUGHT 1, MISSED 8; pb10 CAUGHT 4).
- **fso-fix6's partial bundle is not filed in the corpus.** Its record keeps the steps it read, marked as a budget
  void. Dry run: `dryrun-fix6b-seq.txt`.

### fso-fix6, result (2026-10-18 20:12-20:45): 6 of 6 cells read as predicted before the batch budget cut the boot; a budget void for the other 3

`records/fso-fix6.board.txt`. As the amendment above said, the batch rung ran out of its 1500 s wait.
- **The six cases that ran** (00-05, buggy and fixed) read MISSED, as predicted. On 3005ed52 the same arm read 0 of 9,
  every cell control-failed by S-27's 0x12 fault.
- **The controls fire as positive controls should:** bounds-malloc faults 28 and uaf-malloc faults 25.
  - Those two are the boot's only fault lines. There is no 0x12 fault.
- **Cases 06-08 never ran.** They score NO-READING:infra, a budget void and not a module reading. The re-run on
  bkit7-fix6t (fso-fix6b) reads them.
- **compare-pilot.py** crashed on a cell with no board image. It now reports such a cell as NOT RUN and counts it
  against the total. The record was rewritten with the fixed scorer.
- **The partial bundle is not filed in the corpus.**
- **The sequence stopped after fso**, so pb12-fix6 and pb10-fix6 did not run. Both are in the fix6b re-run.

### pym1-fix6, result (2026-10-18 20:45-20:58): CAUGHT 20, 20 of 20 cells as predicted (0 of 20 on 3005ed52)

`records/pym1-fix6.board.txt`. The images are pym1's list, gate-checked, on kit bkit7-fix6 (module d71f207a3af5546b).
- **The cells:** every case reads CAUGHT, and control uaf-block faults as predicted. That makes 21 fault lines, the
  detections themselves. Nothing faults at 0x12.
- **On 3005ed52** the same images read 0 of 20, every step control-failed by S-27's 0x12 fault.
- **Timing:** the boot took 13 minutes, because a caught program ends at its detection.
- **The bundle:** `cpython/pymalloc-repros/results/2026-10-18-vp2-rdx0/vp2-nested-pools`. UNGATED (PREFLIGHT=0).

### postgres/sql-repros vp2-malloc on the workaround module, pre-registered (2026-10-18, before its boot): sql0-fix6, last in the queue

**The arm** (`tier3-sql-arm.sh`, after the fix6b sequence):
- The compiler lane's lt0 image, `t3-new6/postgres-lt0/image/postgres.dom` (81ee24a0a1e957c9, CPONVVM6, -fno-builtin
  through the SDK hook). The gate is checked before the boot.
- The native16 fixture `t3-new6/tmp/pgfixture/cluster`, nothing pre-installed. run-arm.py records the tree as
  173aa5b35c1b57b9; the builder's own hash is 1287b5822d166345.
- sql-repros' run-arm.py, `--arm virtual-malloc --board-kit bkit7-fix6s`, in one boot of 21 single-user postgres
  sessions: 5 preflights, 2 controls, the cases, and their control.sql sessions.
- The runner (292e7440305d3ad3) and judge (2ed6701045dd5689) hashes equal the QEMU record's.

**The kit:** bkit7-fix6s is bkit7-fix6 with `BOARD_TIMEOUT_MULT=4`, which board-run.sh multiplies into the runner's
budget.
- The runner gives 600 + 300 s a session, 6900 s here. The recipe's stub mode reads 27600.
- The reason: on vp2 a step takes about 18 times its QEMU time. pym0's twin ran 21 steps in 3.3 min, and silicon took
  about 2.1 min a step.
- The QEMU record took about 47 s a session (16.5 min for the boot). So about 13-15 min a session is expected here,
  and the boot runs 4-5 hours. It goes last because a hung session would hold the board to the budget. The idle limit
  still ends a wedged core.

**Predicted** (the QEMU record `t3-sql-virtual-malloc/bundle`, qkit7, module 5bfc38f8):
- CAUGHT 02 and 03 (cause 28 in memcpy, attributed by control.sql);
- MISSED 01 and 04-09;
- both controls (bounds-malloc, uaf-malloc) fault;
- case 03's CREATE EXTENSION ltree runs without pre-install.
- The compiler lane's gate reads every bundle the same on the two modules' QEMU builds, so the 5bfc38f8 record
  stands for d71f207a.

**VOID** if the bundle's fixture record is not 173aa5b35c1b57b9, or its image not 81ee24a0a1e957c9.

UNGATED (PREFLIGHT=0). Dry run: `dryrun-sql-arm.txt`.

**Amendment to the fence discriminator's wording (2026-10-18 21:25, before its boot; the prediction is unchanged):**
- In 3005ed52's source, `fence_before_entry` makes `run()` emit `fence rw,rw` before the CSRUNV, and nothing else
  (e5775346cc8e^ `capstone_vm.c:616-625`). There is no `sfence.vma`.
- So the arm separates MEMORY ORDERING from the rd = x0 change and says nothing about the TLB.
- On the TLB side, the board kernel brings up one hart ("smp: Brought up 1 node, 1 CPU", pym0-fix6's raw console).
  That is what closes the module's resume-boundary TLB gap on vp2 today (the runtime lane's point).

### Queue change (2026-10-18 21:35, before the boots concerned): sql0 moves to the next module; sq0a and sq0f join the d71f207a set

- **The runtime lane's ruling:**
  - The d71f207a chain finishes as pre-registered (sq0c-e, wm0, wm1, the fence arm, the fix6b trio), so every bundle
    of that set names one module.
  - Then the next module revision gets its Tier 1 boot.
  - Then sql0, cpython allocator-boundary and pg-pools run on the NEW module.
- **sql0-fix6's chain is stopped.** It was still waiting, so no boot was spent. Its pre-registration above is
  superseded before any boot; the arm is re-registered on the new module.
- **The fold maps an arm to ONE bundle,** so sq0's six subsets must carry one module to merge. sq0a and sq0f ran only
  on 3005ed52, so they re-run as sq0a-fix6 and sq0f-fix6 (`tier3-fix6c.sh`, after the fix6b trio). The
  `tier3-fix6-appvm.sh` invocation is unchanged, started through a trigger file.
  - **Predicted:** sq0's prediction cut to each subset. They matched it on 3005ed52 (6 of 6, 3 of 3) with no S-27
    fault, so the same reading is expected.
- **Not mergeable as asked:** firmware and kernel differ across subset boots by construction. Each bakes its own
  stage. The 3005ed52 subsets agree in the 11 other platform fields. How to merge is the runtime lane's ruling.
- Dry run: the wrapper started the appvm script on its trigger with arms sq0a and sq0f, which were scored 6 of 6 and
  3 of 3 on stubbed readings.

### RETRACTION (partial, 2026-10-18 21:50): case 22 of sq0 never carried S-27; 14 of the 15 non-matching 3005ed52 cells did, not 15

Withdrawn: the sq0 summary's "the other 15 carry S-27's register-2 fault" (dev 4303f2993d57).
- Case 22 was never S-27. On 3005ed52 it faulted exactly as it does on d71f207a (cause 13, the same pc, the same ASCII
  tval). Its faulting instruction has no a5 operand and its tval is not 2.
- It was counted among the 15 because it was a field DIFF. Its DIFF is the cause-priority one below.
- **Corrected count: 14 of the 15 carried S-27.**
  - 9 had tval 2: 06, 10, 12, 14, 15, 19, 20, 21, 23.
  - 5 faulted 28/29 at real addresses: 07, 08, 09, 17, 26. These are now confirmed S-27-downstream: each reads as
    predicted on the workaround module.
- Cases 19 and 20 then show the cause-priority DIFF on d71f207a, with their verdicts as predicted.
- The evidence is sq0d-fix6 (`records/sq0d-fix6.board.txt`, below). On the workaround module, case 22 faults as it did on
  3005ed52.

### sq0b-e on the workaround module, results (2026-10-18 20:58-21:48): 21 of 24 cells as predicted in every field, all 24 in verdict

`records/sq0{b,c,d,e}-fix6.board.txt`. Kit bkit7-fix6, module d71f207a3af5546b. Each subset's images are gate-checked.
The bundles are under `sqlite/engine-repros/results/2026-10-18-vp2-rdx0/vp2-malloc-sq0<x>`, and the merged arm bundle
follows once sq0a and sq0f have run.

| subset | d71f207a (every field) | d71f207a (verdict) | 3005ed52 (every field) |
|---|---|---|---|
| sq0b | 6 / 6 | 6 / 6 | 1 / 6 |
| sq0c | 6 / 6 | 6 / 6 | 2 / 6 |
| sq0d | 3 / 6 | 6 / 6 | 1 / 6 |
| sq0e | 6 / 6 | 6 / 6 | 5 / 6 |

**sq0d's three field DIFFs differ in the fault CAUSE alone.** Each has the same verdict, the same faulting
symbol+offset, the same attribution and the same tval on both sides:
- cases 19 and 20 (rtree stale node, `nodeRelease+144`): silicon cause 4 (load misaligned), QEMU cause 24, tval 0x17;
- case 22 (spellfix stale vtab, `editDist3FindLang+244`, `lw a0, 0(a0)`): silicon cause 13 (load page fault), QEMU
  cause 24, tval 0x74746973272c276e (ASCII: the freed vtab's memory now holds a string).
- So for a load through an untagged register holding a bad address, silicon reports the address exception and QEMU the
  capability one.
- **Traced (ISSUES Q-31):** this is exception ORDER, not a missing check.
  - Inside a context the LSU's tag check is live: the context runs at raw M, and ppE3's 28 comes from the same block.
  - A misaligned exception wins the merge (`load_store_unit.sv:1113-1114`), and a non-canonical address page-faults in
    the MMU (`cva6_mmu.sv:533-534`, `:630-631`). QEMU tests the tag first.
  - No verdict can flip, since an aligned, canonical address still meets the tag check.
  - (A first rtl-oracle pass concluded that the check is not live in a context. It had missed the raw-M privilege and
    is superseded.)

UNGATED (PREFLIGHT=0). The count of the 3005ed52 cells that carried S-27 is corrected in the RETRACTION just above.

### wm0-fix6, result (2026-10-18 21:48-21:59): the same wedge as on 3005ed52, so it is not downstream of S-27, and R-57's attribution stands at N=2

`records/wm0-fix6.board.txt`. Kit bkit7-fix6, module d71f207a3af5546b, wm0's 26 images gate-checked.
- **The boot ran 2 of 26 steps,** exactly as on 3005ed52. Control uaf-chunk completed, then the core wedged in control
  uaf-jumbo.
- **The wedge apertures equal wm0's on 3005ed52 in 14 of 15 switch fields:**
  - the trap log 0x88 and its mepc 0x35a88 and tval 0x73;
  - WAIT_STORE_READY, a store-buffer commit count of 4, an unacknowledged capability write, and the switcher at
    index 28;
  - the commit pc at the same code offset (…96f54).
- **Only sw=221's commit-queue valid bits differ:** 1011 here, 1101 on 3005ed52, and 0111 in wm1. The invalid entry
  sits in another slot each time.
- **As pre-registered,** the same apertures again keep the RTL lane's R-57 attribution, now at N=2. Removing S-27 did
  not remove this wedge.
- **The bundle is not filed:** 21 of its cells never ran, and the 3005ed52 wm0 bundle was not filed either.
- wm1-fix6 follows. UNGATED (PREFLIGHT=0).

### wm1-fix6, result (2026-10-18 21:59-22:11): the same wedge as on 3005ed52; neither wmem wedge is downstream of S-27

`records/wm1-fix6.board.txt`. Kit bkit7-fix6, module d71f207a3af5546b, wm1's 26 images gate-checked.
- **The boot ran 1 of 26 steps,** and the core wedged in the first control, as on 3005ed52.
- **The wedge apertures equal wm1's on 3005ed52 in every field but two:**
  - sw=221's commit-queue valid bits: 1011 here, 0111 there;
  - the commit pc, which sits at the same code offset (…8988) in a different load.
- **Together with wm0-fix6,** both wmem arms wedge identically with and without S-27's workaround. R-57's attribution
  (the RTL lane's) holds at N=2 per arm, and wmem's silicon arms wait for a bitstream with R-57's fix.
- **The bundle is not filed:** 25 of 26 steps never ran, the same rule as wm0-fix6.
  - The 3005ed52 wm1 bundle was filed (`2026-10-18-vp2/vp2-nested-pools`). It is likewise a wedge in the first control,
    and the fold should treat it as no reading.
- UNGATED (PREFLIGHT=0).

### The S-27 fence discriminator, result (2026-10-18 22:11-22:22): the fence is ruled out, as predicted

`records/pym0-fence.board.txt`. Kit bkit7-fence: module 3005ed52e860c7af with `fence_before_entry=1`.
- **The run:** pym0's 21 images, gate-checked equal to the list. The batch read the parameter back once:
  "@@PARAM fence_before_entry=Y".
- **The reading:** all 21 steps fault `cause=6 address=0x12` (20 at pc offset …159c8, 1 at …1610c), so every cell is
  control-failed.
- **Against pym0 on 3005ed52 without the fence:** 20 of 20 cells the same. Against the QEMU prediction: 0 of 20.
- **So `fence rw,rw` before every CSRUNV does not cure S-27,** and the cure in d71f207a is in its other changes: rd =
  x0, the sentinel, the +16 kind read.
  - On silicon rd = x0 is still not separated from the sentinel and the +16 read. The RTL lane's va-rdclob, which has
    neither, carries that.
- ISSUES S-27's status is updated to match.
- The bundle (`cpython/pymalloc-repros/results/2026-10-18-vp2-fence/vp2-malloc`) is a discriminator record, not a fold
  input. UNGATED (PREFLIGHT=0).

**Two notes added after the fence result (2026-10-18 22:35):**
- **The fence arm is silent about translation.** `fence rw,rw` orders memory accesses and invalidates no TLB entry;
  only `sfence.vma` does. So "a fence before entry changed nothing" says nothing about a stale-translation window (the
  RTL lane's va-satpstale candidate for S-23).
- **pb10's MTVL is a PHYSICAL address,** which settles the VA-versus-PA question raised against the S-23 data-face
  decode.
  - A data-side CPMP refusal from S-mode under capmode reports `tval = cpmp_paddr_x`, which is `lsu_paddr_i`
    (`pmp_data_if.sv:135`, `:304-313` at bf3db3175). Only the standard PMP path reports the virtual address
    (`:283-288`).
  - 0xbc2c0825 cannot be a kernel virtual address, so the refusal came from the CPMP path.
  - So region 21 [0xAC100000, 0xC0000000) did cover the refused physical address in the dump. That is the same
    comparison the CPMP makes (`:146-147`).
  - **Consequence for the candidates:** a stale translation alone cannot produce this refusal. Under a steady table
    the page is covered, whichever table translated to it. The refusal still needs the CPMP table to be in flux at the
    access, which is S-23's window.
  - A stale translation could still be a contributing step, but it is not sufficient here.

### Tree A flash plan, pre-registered (2026-10-18 22:45): NOT AUTHORISED until the lead says so in this lane's own session

Tree A is capstone-ariane `virtual-profile-2-c` 8522d72fe, carrying the fixes for R-57, Q-24, S-27 and S-28. It is
not sealed yet: synthesis is in implementation, and a congestion failure is a live outcome. A flash needs the lead's
own word in this session; a relayed one does not count. Nothing below runs before that.

**Where the flash falls:** at a chain boundary, never inside a bundle.
- The d71f207a set finishes first: fso/pb12/pb10-fix6b, then sq0a/sq0f-fix6. That way every bundle of the set, and the
  merged sq0 bundle, names one bitstream.
- The flash goes after `FIX6APPVM DONE` in `t3-run-fix6c.out`, and before the module revision's Tier 1 boot if that
  has not started. If tree A seals while a chain runs, the chain finishes its current arm and the next one is held.
- Everything is pushed (lane and dev) before the flash.

**Before the flash:**
- `GET /api/bitstreams` must list `caplifive_vp2_bf3db3175.bit`, so the flash-back path exists.
- Register tree A's sealed `.bit` under the name the synth lane gives, with its sha256 recorded here before the
  upload.
- Then the HOW-TO-LAUNCH-ON-FPGA sequence: power on, settle ~30 s, `POST /api/flash-bitstream`, wait for a FRESH
  `flash_state=done`.
- Read back `nv_bitstream_name`. It must equal tree A's name, or the session stops.

**The session, each step gating the next, in vp2's order (dev 987bf254ee40):**
1. **The canary,** two passes: S-16, floor and s1gap. Predicted identical to d93a8d66b's readings in each pass, and
   pass 1 identical to pass 2, so 9 of 9.
2. **The arm set:** each arm equal to the board vector the RTL lane simulates for that exact image on tree A's RTL.
   The vectors are needed from the RTL lane before the flash. vp2's 25 arms are re-simulated, plus any arm tree A's
   fixes add (S-27's va-rdclob, R-57's capability-store pair, S-28).
3. **Corpus set 7** on the same firmware payload: 33 of 33 records identical.
4. **Tier 1** with the module revision current at that time (d71f207a, or its successor once its own Tier 1 has passed
   on bf3db3175).
   - Predicted the same readings as on bf3db3175: the S-27 workaround is harmless on fixed RTL.
   - Firmware: the same monitor (007d81d with the S-23 fetch retry; tree A does not fix S-23), with the DTS checked
     against the new bitstream by the preflight.
5. **wm0, then wm1,** as R-57's positive controls on the then-current kit. Predicted to RUN instead of wedging, reading
   their QEMU predictions (wm0 MISSED 23; wm1 its twin's control-failed 23). Each wedged with the same apertures on
   both modules on bf3db3175.
6. **pym0 on the OLD module 3005ed52 (kit bkit7),** as S-27's control on the fixed RTL.
   - The RTL now exempts resume modes 3/4 from the rd write, so even the module's rd = x15 is not clobbered.
   - Predicted 20 of 20 at the QEMU verdict (control complete, MISSED 20), against 0 of 20 on bf3db3175 with the same
     images. That is the RTL fix two-sided on silicon.

**Stops:** any canary mismatch, a readback naming another bitstream, or an arm vector mismatch stops the session
before the Linux steps. A failed canary means flashing back to bf3db3175 needs the lead's word again.

### fso, pb12 and pb10 re-run on the scaled budget (2026-10-18 22:22-23:27): fso 9 of 9, pb12 9 of 9, and pb10 repeats its S-23 panic exactly

Kit bkit7-fix6t (BOARD_STEP_S=300), module d71f207a3af5546b.
- **fso-fix6b** (ffmpeg/subobject, vp2-malloc): MISSED 9, 9 of 9 as predicted. On 3005ed52 it read 0 of 9. The boot
  took 43 minutes inside a 6300 s budget.
  - Bundle: `ffmpeg/subobject-repros/results/2026-10-18-vp2-rdx0/vp2-malloc`. The budget-void fso-fix6 bundle was
    never filed, so this one takes the arm's place.
- **pb12-fix6b** (memcached/allocator, vp2-malloc): CAUGHT 1, MISSED 8, 9 of 9 as predicted. On 3005ed52 it read 0 of
  9.
  - Bundle: `memcached/allocator-repros/results/2026-10-18-vp2-rdx0/vp2-malloc`.
- **pb10-fix6b** (ffmpeg/pool, Sublet): the same as on 3005ed52, and not a budget void.
  - Case 00 reads CAUGHT. Case 01's BUGGY run panics the kernel, so cases 02-03 never run: 1 of 4.
  - The panic matches pb10's on 3005ed52 line for line: `MTVL:BC2C0825`, `CPMX:00000002`, `INJS:00000002`, and all 48
    lines of the CPMP dump identical.
  - The kernel side matches too: `Oops - load access fault` at `mod_node_page_state+0x40`, called from
    `unpin_user_page+0x2a`.
- **What this means for S-23's data face:**
  - MTVL is the CPMP's physical address (`pmp_data_if.sv:135`, `:304-313`), and region 21 covers it in both dumps.
  - The refusal now repeats at the same physical address, from the same caller and in the same step, on two boots and
    two modules. That is a deterministic sequence, not the scatter a random timing window would give.
  - The RTL lane's window reading (the CPMP table in flux while the RESTORE walk writes it) still fits only if the
    window is hit identically each time. The record says so, and the RTL lane has the two captures.
- **The pb10-fix6b bundle is not filed:** 3 of its 4 cells never ran.
- UNGATED (PREFLIGHT=0).

### sq0a and sq0f on the workaround module, and sq0 merged (2026-10-18 23:27-23:50): 33 of 33 cells as predicted in verdict

- **sq0a-fix6:** 6 of 6. **sq0f-fix6:** 3 of 3. Kit bkit7-fix6, module d71f207a3af5546b, images gate-checked.
- **All six sq0 subsets now carry d71f207a.** They are merged by `merge-subset-bundles.py` into
  `sqlite/engine-repros/results/2026-10-18-vp2-rdx0/vp2-malloc`, the bundle the fold points to.
  - The merge holds 33 cells, tally CAUGHT 5, MISSED 26, NO-READING 2 (inconclusive 1, unattributed 1). That equals
    the QEMU prediction's tally.
  - The 11 shared platform fields are identical across the subsets. Firmware and kernel are recorded per subset.
  - The merged lines are the subsets' own, byte for byte.
- **Against the prediction:** 30 of 33 cells the same in every field, and 33 of 33 in verdict. The three field
  differences are sq0d's cases 19, 20 and 22, the fault cause only (Q-31: the RTL's trap order).
- **On 3005ed52** the same arm read 18 of 33, with 14 cells carrying S-27.
- **The d71f207a set is complete:** pym0, pym1, fso, pb12 and sq0 match their predictions. pb10 repeats its S-23
  data-face panic. wm0 and wm1 wedge as before (R-57). UNGATED (PREFLIGHT=0).

### The matched arms re-run on the workaround module, pre-registered (2026-10-18 23:55, before their boots)

The board is idle until the module revision's handover, or the lead's word on tree A. The nine run-virtual-cases arms
that matched their predictions on 3005ed52 re-run on d71f207a, so the fold can name one module for every arm.
- **How:** `tier3-fix6b-seq.sh` (dry-run before), kit bkit7-fix6t, tags `<arm>-fix6b`, the same predictions. The
  order is shortest first: mpt, pb9, mph, fpt, fca, wpt, wph, fcs, fph.
- **Predicted:** each reads its 3005ed52 reading again, the QEMU prediction:
  - mpt CAUGHT 3; pb9 MISSED 4; mph CAUGHT 9; fpt CAUGHT 13; fca MISSED 13; wpt CAUGHT 10; wph CAUGHT 12;
    fcs CAUGHT 13; fph CAUGHT 25.
  - None of them read an S-27 fault on 3005ed52.
- **What a difference would mean:** a cell that differs is a module regression, read at its first failure.
- **Interruption:** if the revision's Tier 1 boot or the tree A flash becomes due, the chain finishes its current arm
  and the rest are held (stop it between arms by its process group).
- UNGATED (PREFLIGHT=0).

### pgc, pgm0 and pgm1 on the workaround module, pre-registered (2026-10-18 23:55, before their boots), after the nine

`tier3-pg-fix6.sh` runs after the nine-arm re-run ends. It uses `tier3-pg-arm.sh` unchanged, with kit bkit7-fix6t
(module d71f207a3af5546b) and the t3-freeze-nb pg builds. The records are `<arm>-fix6b`, and the bundles go under
`results/2026-10-18-vp2-rdx0/`.
- **The predictions:** the compiler lane's QEMU records, `t3-<arm>/bundle`, made through the qkit7 wrapper with
  svadu=false applied:
  - pgc (postgres/c-repros, vp2-malloc): CAUGHT 5;
  - pgm0 (postgres/mmgr, vp2-malloc): MISSED 5;
  - pgm1 (postgres/mmgr, vp2-pg-pools): CAUGHT 5.
- On 3005ed52 all three read 15 of 15, the second matched pair (pgm0 against pgm1).
- **pgm1 and the Q-15 repair:** the runtime lane predicts the repair leaves pgm1 unchanged, since its recorded faults
  do not use the ≥4 KiB large-chunk paths. So this reading stays valid after the repair.
- UNGATED (PREFLIGHT=0). Dry run: `dryrun-pg-fix6.txt`.

**Amendment to the tree A flash plan's step 2 (2026-10-19 00:05, before any flash; still NOT authorised):** the
arm-set predictions are in.
- **The table:** dev 7ba0a7afc87a, `vp2-bare-2026-10-17/results/rtl-sim-vectors-8522d72fe.txt`. All 25 arms are
  IDENTICAL on tree A's RTL to bf3db3175's vectors, count and values.
- **So step 2 gates on the existing `rtl-sim-vectors.txt` values,** each arm by the sha256 of the `.bin` the board
  loads, exactly 25 arms.
- **The instrument's control:** the rebuilt extractor reproduces all 25 published bf3db3175 vectors exactly.
  - It fails four distinct ways on purpose: a missing count word, a hole in the values, a wrong recorder address, an
    empty trace.
  - Every arm ends at the cycle limit by design, so the gate keys on the recorder's count word and values, never on the
    harness verdict.
- **Two arms are vp1 images** not in that folder: q17-order and vb-trans-off. Their CSRUNV column reads "?".
- **The RTL lane's caveat, quoted:** "an identical vector is exactly what you get when no acknowledged RESUME with a
  non-zero rd ever happens, and S-27's fix deliberately leaves rd alone on a first entry. So this set does not
  exercise S-27 and must not be cited as evidence that it works."
  - The same holds for R-57 (these images run in M-mode with no translation), Q-24 and S-28.
  - S-27's RTL fix on silicon is therefore step 6's (pym0 on module 3005ed52), not step 2's.

### Module revision 39d8492d's Tier 1 boot, pre-registered (2026-10-19, before the boot): rev7

**The module:** board module 39d8492d5c18216b, from the compiler lane's lane module-resume-rd-x0 a0a256575831 (code
76cfc60b3626: five items on e5775346cc8e).
- **The items:**
  - (1) the collect stopgap: -ENOMEM, and the pr_emerg "collect skipped";
  - (2) frame tests;
  - (3) a resume sfence.vma of the resolved address;
  - (4) a satp refresh from +928 before every resume;
  - (5) a busy-frame refusal before the pin.
- **The gates, re-run by stage-rev7.sh:** qemu-only-csr-census and vp2-silicon-gate.
- **Firmware:** fw_payload 7a53759a4773c81b (Image dcfc2e15bf1bbd77) (monitor 007d81d with the S-23 fetch retry, as wa1).
- **The stage** is wa1's with exactly three files changed: the module (d71f207a -> 39d8492d), `vm/vm-contract.sh` and
  `vm/vm-suite-inner.sh` (both `board-tier12/rev7/`).

**The boot:** rungs b0-stats, b0-hello, vm-contract, vm-suite.
- vm-contract loads the module with `diag_emerg=1 node_flat_records=4096`. diag_emerg puts CV_EMIT_FIRST lines at
  KERN_EMERG, and the reduced capacity lets sparse-churn exhaust within a few thousand allocations.
- It reads both parameters and supervisor_slots back, then the four counters before `ok` and after `stale`.
- vm-suite runs sparse-churn LAST between two counter reads.
- **The risk:** sparse-churn has never run on vp2. It was held under S-25, since a first-touch capability store has
  wedged the core (R-57). A wedge there is an S-25/R-57 reading taken after every other reading, not a module
  verdict.

**Predicted** (the compiler lane's QEMU readings on b290f5656600 + svadu=false, and rev7-judge.py's checks):
- b0-stats 0, b0-hello 0, vm-contract 0 with the identity branch (VIRTUAL_APPLICATION_OK, then the stale access
  denied with 25), as ap1 and wa1.
- `@@PARAM` diag_emerg=Y, node_flat_records=4096, supervisor_slots=1. The banner reads "loaded, diag arena-pin-1,
  supervisor_slots=1, diag_emerg=1", once, with 0 refusal lines.
- **The counters across vm-contract** (deltas, since the totals run from insmod):
  - resume_sfences = the sum of `faults=` over vm-contract's two CAPSTONE_VM_STATS lines, and > 0;
  - resume_satp_refreshes = the sum of `steps - 1`;
  - resume_satp_changed and busy_frame_refusals are 0 at every read.
- **sparse-churn:** `VM_EXIT:sparse-churn:1`, `VM_CONTRACT_FAIL:recycling allocation`, and the console's "capstone_vm:
  collect skipped: collection is not available on bf3db3175" at least once.
  - On QEMU at 4096: collect skipped at 3,584 nodes, then exit 1.
- **vm-suite** retval 1, the count of its VM_EXIT lines.
- **Readings that are not that:** a counter delta off (the resume path not as built); collect skipped absent (the
  must-fire not reaching the console, or the stopgap not taken); a wedge in sparse-churn (S-25/R-57).

**The re-run chain on d71f207a is HELD** at an arm boundary for this boot: after fpt-fix6b, with fca, wpt, wph, fcs,
fph and then pgc, pgm0, pgm1 still to run. It resumes after this boot. UNGATED (PREFLIGHT=0). Dry run:
`rev7/dryrun-rev7.txt`.

### Module revision 39d8492d's Tier 1 boot, result (2026-10-19 00:31-00:42): Tier 1 passes; sparse-churn wedges with S-25's sparse-churn face, before the stopgap could fire

`board-tier12/results/board-vp2-tier12-rev7.result-lines.txt`. Firmware 7a53759a4773c81b.
- **As pre-registered:**
  - b0-stats, b0-hello and vm-contract read as their oracles, with vm-contract's identity branch. That is identical to
    ap1 (3005ed52) and wa1 (d71f207a).
  - `@@PARAM` diag_emerg=Y, node_flat_records=4096 and supervisor_slots=1. One banner, "loaded, diag arena-pin-1,
    supervisor_slots=1, diag_emerg=1", and 0 refusal lines.
  - The counters before vm-contract read (0, 0, 0, 0). resume_satp_changed and busy_frame_refusals read 0 at all
    three reads taken.
- **The counter deltas, and a slip in my own check:**
  - `ok` printed `CAPSTONE_VM_STATS ... steps=511 faults=426`. 426 is exactly QEMU's count for `ok`.
  - The deltas across vm-contract are resume_sfences 447 and resume_satp_refreshes 550.
  - `stale` faults with cause 25 and prints no STATS line. My judge's prediction, "the deltas equal the sums over
    vm-contract's TWO STATS lines", could therefore never be checked exactly, and the judge reported NO DATA on it.
  - What the reading supports: `ok`'s 426 resolved faults are inside the delta (447 = 426 + `stale`'s unprinted
    faults), and 550 >= `ok`'s 510 = steps - 1.
  - That is consistent with item 3 and item 4 running on every resume, but not the exact equality pre-registered. The
    exact check needs the counters read between `ok` and `stale`.
- **sparse-churn wedged the core** (`VM_BEGIN:sparse-churn`, no VM_EXIT, driver rc 1).
  - The apertures equal S-25's sparse-churn face from r8, r9 and r12 in every column that table records: count 1, no
    valid bits, store_state 3 (WAIT_STORE_READY), 224 = 0x25, 225 = 0xc4, and the switcher at index 56.
  - So it is S-25 again (N=4 for this face), now on module 39d8492d at node_flat_records=4096.
  - The wedge came before exhaustion. The stopgap's "collect skipped" was never reached, so its silicon positive
    control is still OPEN. The reduced capacity did not bring the must-fire ahead of the first-touch wedge.
- **Verdict:** module 39d8492d passes Tier 1 on silicon. Its stopgap is unverified on silicon until R-57/S-25's fix.
  UNGATED (PREFLIGHT=0).

### The re-run's first four arms, results (2026-10-18 23:51 to 2026-10-19 00:28), and sql0 re-registered on module 39d8492d (before its boot)

**Results** on kit bkit7-fix6t (module d71f207a3af5546b), each as predicted, the QEMU tally again:
- mpt-fix6b CAUGHT 3 (3 of 3);
- pb9-fix6b MISSED 4 (4 of 4);
- mph-fix6b CAUGHT 9 (9 of 9);
- fpt-fix6b CAUGHT 13 (13 of 13).
The bundles are under each corpus's `results/2026-10-18-vp2-rdx0/vp2-malloc`. The chain was held after fpt for the
revision's Tier 1 boot. A first hold left fca's next arm running in its own `timeout` process group; it was stopped,
the shared image's restore was finished by hand, and fca re-ran from the start.

**sql0 on module 39d8492d (`tier3-sql-arm-rev2.sh`)** is re-registered by the runtime lane's ruling: sql0 FIRST, on
the new module, ahead of the d71f207a chain.
- **The kit:** bkit7-rev2s is bkit7-rev2 (module 39d8492d) with bkit7-fix6s's recipe and `BOARD_TIMEOUT_MULT=4`. It
  differs from bkit7-fix6s in the module alone (`diff -r`, apart from the manifest path).
- **Image, fixture, runner and judge** as the d71f207a registration above, which was superseded before any boot:
  lt0 81ee24a0a1e957c9, fixture record 173aa5b35c1b57b9, runner 292e7440305d3ad3, judge 2ed6701045dd5689.
- **Predicted** as the QEMU record `t3-sql-virtual-malloc/bundle`: CAUGHT 02 and 03, MISSED 01 and 04-09, both
  controls fault. The compiler lane's gate reads the five-item revision 16 of 16 identical on QEMU, so the record
  stands.
- **VOID** unless the fixture record and the image match after the boot.
- **The record and bundle:** `records/sql0-rev2.board.txt` and `postgres/sql-repros/results/2026-10-19-vp2-rev2/`.
- **Order:** it starts on a trigger file once the d71f207a chain is held after fca-fix6b and the board is idle. The
  chain (wpt, wph, fcs, fph, then pgc/pgm0/pgm1) resumes after it. If tree A's flash becomes authorised and sealed
  while sql0 runs, sql0 finishes first.
- **Dry run:** the guard, pass and VOID branches, on a stubbed runner. UNGATED (PREFLIGHT=0).

**fca-fix6b, result (2026-10-19 00:46-00:56):** ffmpeg/carved vp2-malloc on kit bkit7-fix6t (module d71f207a3af5546b),
MISSED 13, 13 of 13 as predicted, its 3005ed52 reading again. Bundle
`ffmpeg/carved-repros/results/2026-10-18-vp2-rdx0/vp2-malloc`. The chain was then held for sql0-rev2: the holder
killed the chain's group and wpt's `timeout` group within 6 s of fca's END, before wpt built anything. UNGATED
(PREFLIGHT=0).

### sql0-rev2 stopped at the firmware cap (2026-10-19 00:57-00:58), and the payload rule for every new pre-registration

- **sql0-rev2 did not boot.** The kit recipe's DECLARED-bake cap stopped it after the bake and firmware build, before
  the board: `STOP: firmware 30050824 B over the cap 26000000`. No board time was spent.
  - The stage was 65,953,267 B in 1,668 files. bin/postgres is 21.7 MB, and the native16 fixture is ~41 MB: one 16.8
    MB WAL segment, and template1, template0 and postgres at 8 MB each.
  - The runtime lane ruled that the cap stays at 26,000,000 B. The fixture shrinks instead, from the compiler lane: a
    1 MB WAL segment, template0 only if a case needs it, GMT-only zoneinfo.
  - Its QEMU A/B must read every sql cell equal to the native16 record, carry the new fixture_record hash, and print
    the stage bytes and the estimated firmware size. sql0's re-registration on 39d8492d gates on that hash.
- **My miss:** the sql0 pre-registration never sized the stage against the cap. The recipe's gate caught it, which is
  that cap's first positive control.
- **From now on every new pre-registration carries the stage bytes and the estimated firmware size.**
- **Applied to pgc/pgm0/pgm1, before they run.** Measured from their 3005ed52 runs, the same images and builds:
  - pgc: stage 3,192,428 B in 15 files;
  - pgm0: stage 5,780,035 B;
  - pgm1: stage 5,972,411 B;
  - firmware 19,565,064 B each, 6.4 MB under the cap.
  - The d71f207a module is 4 KiB larger than 3005ed52's, which does not move the 2 MiB-padded firmware.

**Amendment to the tree A flash plan (2026-10-19 01:06, before any flash; still NOT authorised): the bitstream is sealed
and pulled.**
- **The seal:** synth lane, capstone-ariane virtual-profile-2-c 8522d72fe. Routed WNS -18.985 (vp2: -19.019), TNS
  -1,032,423.812, LUTLP-1 0, route 39 min with no congestion failure.
- **The file:** `caplifive_vp2c_8522d72fe.bit`, 11,443,722 B, sha256
  ba1339bcd1f6fbb4a8d9322990ae054e9494853553802d004bcd4ec2e30f5e7e.
  - Pulled from the synth host's work-fpga output and verified against that sha256 before it was named. It is kept
    read-only in the board lane's bitstream directory, beside `caplifive_vp2_bf3db3175.bit`.
- **At the flash:**
  - register it under that name;
  - confirm `caplifive_vp2_bf3db3175.bit` is still registered, as the flash-back path;
  - read back `nv_bitstream_name`; it must be `caplifive_vp2c_8522d72fe.bit`. The readback is recorded with the
    session.
- **Where the flash falls:** at the next arm boundary of the d71f207a re-run once authorised. An arm in flight finishes
  first.

### sql0 on module 39d8492d, second form, pre-registered (2026-10-19 01:46, before its boot): sql0-rev2b

The first form stopped at the firmware cap. This one changes the image and the fixture, both re-gated on QEMU:
- **The image:** `t3-new6/postgres-lt0-dl/image/postgres.dom`, 64010720f1a6b3bc, `llvm-objcopy --discard-locals` of
  81ee24a0.
  - It drops 216,837 `.L` labels. The single PT_LOAD segment, .capstone_virtual and .capstone_domreq are
    byte-identical, and so is every non-.L symbol (checked by this lane and the compiler lane independently).
  - All 6 sql fault_sites and the record's evidence symbol (memcpy) are non-.L and present.
- **The fixture:** `t3-new6/tmp/pgfixture-trim/cluster`, run-arm record 956173e054762e91. It is the native16 cluster
  without template0/template1 and with a 1 MiB WAL segment.
- **`--pg-share`:** the share the native16 record used.
- **The prediction:** `t3-sql-trim-vm/bundle`, the QEMU record on exactly these inputs (qkit7):
  - CAUGHT 02 and 03 (cause 28 at memcpy+200), MISSED 01 and 04-09, both controls fault;
  - 9 of 9 cells equal to the native16 record, evidence text included. The compiler lane's comparator and this lane's
    both read 0 differ, and both catch a planted difference.
- **The size:** the stage shrinks by 37,143,681 B raw against the first form's 65,953,267. The firmware grows in
  2 MiB steps, so the estimate means either 25,859,336 B (under the cap) or 27,956,488 B (over).
  - A measure-only bake decides it first, on kit bkit7-rev2m: bkit7-rev2s with a recipe stop after the firmware build
    and the cap check, so it never reaches the board.
  - The board boot on bkit7-rev2s follows only if the measured firmware is under 26,000,000 B. If it is over, the
    fixture is trimmed again (the runtime lane's ruling).
- **When:** after the postgres chain (`tier3-sql-arm-rev2b.sh`). VOID unless the bundle's fixture record is
  956173e054762e91 and its image 64010720f1a6b3bc.
- Records `records/sql0-rev2b.board.txt` and `postgres/sql-repros/results/2026-10-19-vp2-rev2/`.
- **Dry run:** under-cap pass, over-cap stop with no boot, and VOID on a wrong fixture. UNGATED (PREFLIGHT=0).

### The re-run's last four arms, pgc's dead boot, and sql0's measured firmware (2026-10-19 01:00-01:52)

- **wpt-fix6b CAUGHT 10, wph-fix6b CAUGHT 12, fcs-fix6b CAUGHT 13, fph-fix6b CAUGHT 25:** each N of N as predicted, on
  kit bkit7-fix6t (module d71f207a3af5546b).
  - The bundles are under `results/2026-10-18-vp2-rdx0/` in wireshark/plain-temporal-repros (vp2-malloc),
    wireshark/plain-heap-repros (vp2-malloc), ffmpeg/carved-repros (vp2-nested-pools) and ffmpeg/plain-heap-repros
    (vp2-malloc).
  - **All nine arms that matched on 3005ed52 now match on d71f207a too,** 102 cells in all.
- **pgc-fix6b did not boot (infra):** the watchdog logged "NO-BOOT 466s load_image issued, no boot banner and no
  download for 421s; JTAG transfer moved nothing", aborted the runner (driver rc 130), and the board was released. No
  board transcript, every cell NO-READING:infra, so it is not a reading.
  - The run is moved to the attic, and its record kept as `records/pgc-fix6b-noboot.board.txt`. Its all-infra bundle
    is not filed. pgc, pgm0 and pgm1 re-queue after sql0 (the board-run skill: infra, retry).
- **sql0-rev2b's measure-only bake read 25,856,520 B**, under 26,000,000: the stage was 28,809,586 B in 1,072 files.
  The board boot on bkit7-rev2s started at 01:52:19. UNGATED (PREFLIGHT=0).

### Two silent boots in a row (pgc-fix6b 01:42, sql0-rev2b 01:52), and a board-health control pre-registered (2026-10-19 02:05)

- **What happened:** both boots downloaded their firmware completely over JTAG ("19565064 bytes in 146.10s",
  "25856520 bytes in 193.07s"). After `continue` there was no UART output at all, until the watchdog aborted each at
  ~420 s.
  - fph-fix6b's boot at 01:30 ran normally.
  - So two different firmwares, back to back, are silent. That is the board-run skill's AMBIGUOUS row: a dead firmware
    or a dead UART relay. Neither is a reading. sql0-rev2b is VOID by its own gate (no bundle).
- **An instrument defect:** the watchdog labelled both "NO-BOOT: load_image issued, no boot banner and no download ...
  JTAG transfer moved nothing". The download DID complete.
  - Its download detection most likely missed the size line, which the console delivered split across two gdb events.
  - The label is wrong, though aborting a silent boot was still right. To be fixed in board-watchdog.sh before it is
    relied on to tell a dead transfer from a dead boot.
- **The control, before any further long boot:** wa1's firmware ad5e84f761dd0f7c (Tier 1 only: b0-stats, b0-hello,
  vm-contract), run by `run-tier12-wa1.sh` into a fresh directory. It booted and passed at 19:14 tonight.
  - **Predicted:** identical to wa1 (three rungs as their oracles, the identity branch, 0 refusal lines).
  - **If it is silent too,** the board or console is down. The queue stops and the board owner is told.
  - **If it boots,** the two silent boots were transient. sql0-rev2b and the postgres arms re-run.

**The control's result (2026-10-19 02:12-02:22): silent too. The board is unavailable from this side.**
- wa1's firmware ad5e84f761dd0f7c, which booted and passed at 19:14, downloaded completely ("19565064 bytes in
  146.43s") and then gave no UART after `continue`. The watchdog aborted it at 570 s, driver rc 130, no transcript.
- **That is four boots in a row, the same way:** pgc-fix6b 01:42, sql0-rev2b 01:52, pgc-fix6b again 02:02 (the
  postgres chain's retry), and this control.
  - They cover two kits and three firmwares, one of them known-good.
  - Each JTAG download completed at the normal ~130 KiB/s, and each was silent after `continue`.
  - `flash_state` still names `caplifive_vp2_bf3db3175.bit` in each.
  - The last good boot was fph-fix6b, 01:30-01:42.
- **No reading at all,** and the cause is not known from this side: a dead UART relay, a board-side fault, or a
  volatile bitstream change that `nv_bitstream_name` would not show.
- **Every queue is stopped.** No further board time is spent until the board owner checks the board and its console.
- **Owed after it is back:** sql0-rev2b, pgc/pgm0/pgm1, and the tree A flash if authorised. Before any of them, one
  short control boot, which must print the OpenSBI banner.

### The no-boot diagnostic session, pre-registered (2026-10-19 02:40, before it runs)

fpga1's read-only check of board #1's host found the UART path alive: the FT232R present and not re-enumerated, no
kernel event since 01:25, the console service up since 2026-09-03. Its journal shows every session since 01:30
downloading completely, with the core at PC 0x10504-0x10544 at attach (the bootrom), then `continue`. So the silence
is board-side.

**The session** (`fpga_driver/diag_noboot_fpga.py`, wa1's firmware ad5e84f761dd0f7c):
- the cold_boot sequence up to `continue`, plus `monitor verify_image` and the first 8 words at 0x80000000 against the
  file;
- the UART bytes for 12 s after `continue`;
- then `monitor halt`, never reset, and pc, mcause, mepc, mtval, mstatus, a0, ra and sp, with two more pc samples
  across resume/halt.
- The board is released (power off) at the end. A power cycle and the short control boot come only after these
  readings are recorded.

**What each reading means:**
- a verify mismatch: the image did not land in DDR;
- pc in the bootrom (0x10000-0x10fff): the core never left it after `continue`;
- a non-zero mcause or pc at the trap vector: an early fault;
- pc moving inside 0x80000000..: the firmware runs, and the console output is what is lost.

**The diagnostic's readings (2026-10-19 02:41-03:00; a first attempt at 02:40 timed out in `gdb_start`, so a 3-try
retry was added, as the production drivers' cold_boot has):**
- **Before `continue`:**
  - pc after `monitor reset halt` = 0x10530, the bootrom.
  - load_image: "downloaded 19565064 bytes in 147.00s".
  - **verify_image: "verified 19565064 bytes in 151.25s"**, slowly, because no work area is configured, but complete.
  - The words at 0x80000000 equal the file's (0x00050433 0x000584b3 0x00060933 0x53c000ef ...). **The image is in DDR,
    intact.**
- **After `continue`:** 0 UART bytes in 12 s.
- **`monitor halt` then timed out** (no gdb prompt in 30 s). The debug module could not halt the core, so pc and the
  trap CSRs could not be read.
- **What this establishes:** the image reaches DDR intact, and the core, released from the bootrom at 0x80000000,
  goes silent at once into a state the debugger cannot halt. That is the same on a known-good image that booted at
  19:14, and it survives the power cycle every session starts with.
- **What it does not establish:** why. It is board-side and needs the board owner: a full power removal, or a check
  that the FPGA configuration and its clocks are what they were. A reflash of bf3db3175 would be the lead's call.
- **No further board time is spent.** The queue stays stopped.

**After the board's reboot (2026-10-19 11:33-11:38): the control passes. The board is back.**
- The lead rebooted the board, relayed by the runtime lane. The pre-registered control then ran: wa1's firmware
  ad5e84f761dd0f7c, Tier 1.
- **It read identical to wa1:** b0-stats, b0-hello and vm-contract as their oracles, the identity branch, banner 1, 0
  refusal lines, and the same FRTY retry at 011110E8.
- So the outage from 01:42 was board-side, and it was cleared by the reboot. Its cause is unknown.
- **Resuming as planned:**
  - sql0-rev2b (on 39d8492d, the measure-only bake first), then pgc/pgm0/pgm1 on d71f207a.
  - The silent runs are in the attic: sql0-rev2b 01:52, pgc 02:02. pgc's second dead-boot record is kept as
    `records/pgc-fix6b-noboot2.board.txt`, and its all-infra bundle is not filed.

### sql0-rev2b, result (2026-10-19 11:40-11:50): VOID, a runner defect; and sql0-rev2c pre-registered (before its boot)

- **What ran:** the measure-only bake again read 25,856,520 B, under the cap. The boot completed and all 21 sessions
  ran, in 10 minutes.
- **What failed:** run-arm.py's own positive control: "the image did not answer a trivial SELECT". Every session's
  output was `cp: can't stat '/mnt/vm/pgcluster'`.
- **Why:** sql-repros' `run-arm.py` wrote the batch step with the QEMU kit's mount path, `/mnt/vm`. The board batch
  stages under `/test-domains/vbatch` (`boardvm.BoardBatch.put`). No postgres code ran, so this is not a reading. The
  chain's gate refused the bundle (fixture record none, image none), and nothing was filed.
- **A second defect would have failed the next boot the same way.** The kit's bake recreates staged directories
  0775. `cp -a` carries that mode into `/tmp/pgd`, and postgres refuses a data directory with group or other write
  access. Native postgres 17.5 on a copy of the trimmed cluster, with directories 0775 and files 0644, gave
  `FATAL: data directory ... has invalid permissions` (rc 1). After `chmod 700` on the directory alone it answered
  `SELECT 1` (rc 0).
- **The fix (runner sha256/16 1f099612d70a9576):**
  - the session copies the path that `batch.put("pgcluster")` returns, and `cd`s to that path's directory;
  - `chmod 700 /tmp/pgd` follows the chown.
  - The c-repros and mmgr runners carry no `/mnt/vm` and no data directory. They are unaffected, so the pg arms were
    released first (PGF BEGIN 11:58:33).
- **The QEMU step is unchanged:** the generated step equals the old literal byte for byte, apart from the added
  `&& chmod 700 /tmp/pgd`, which is a no-op there because the fixture is already 0700.
  - The QEMU check re-runs `run-sql-trim.sh`'s exact invocation with only the runner changed
    (`<mnt-hdd>/capstone/t3-sql-trim-vm-r2`).
  - It is PASS only if `compare-pilot.py` reads every cell and control the same as `t3-sql-trim-vm/bundle` and 21 of 21
    sessions ran. The result is written to that directory's `CHECK` as `PASS <runner hash>`.

**sql0-rev2c** (`tier3-sql-arm-rev2c.sh`): rev2b's script with the tag, the run directory and two gates changed. It
runs only if the runner is 1f099612d70a9576 and the QEMU CHECK reads PASS for it.
- **Inputs and kits:** rev2b's, unchanged: image 64010720f1a6b3bc, fixture record 956173e054762e91, kit bkit7-rev2s
  (module 39d8492d, BOARD_TIMEOUT_MULT=4), and the measure-only bake on bkit7-rev2m first.
- **Size:** rev2b's stage was 28,809,586 B in 1,072 files, with firmware 25,856,520 B. The fix adds about 48 B of gate
  text per session, about 1 KB in all, so the firmware is expected at 25,856,520 B. The measure-only bake decides
  before any boot.
- **Prediction:** rev2b's, i.e. `t3-sql-trim-vm/bundle`:
  - CAUGHT 02 and 03 (cause 28 at memcpy+200);
  - MISSED 01 and 04-09;
  - both controls fault.
- **When:** after the pg chain ends (PGF DONE, STOP or HOLD in `t3-run-pgf4.out`).
- **Records:** `records/sql0-rev2c.board.txt` and `postgres/sql-repros/results/2026-10-19-vp2-rev2/`.
- UNGATED (PREFLIGHT=0).

### pgc, pgm0 and pgm1 on the workaround module, results (2026-10-19 11:58-12:24): 15 of 15 cells as predicted

- **The arms,** as pre-registered (kit bkit7-fix6t, module d71f207a3af5546b; firmware 7a2ceb535ace2a18, 19,565,064 B):

  | Arm | Result | Boot |
  |---|---|---|
  | pgc-fix6b | CAUGHT 5, 5 of 5 cells the same | 11:58-12:07 |
  | pgm0-fix6b | MISSED 5, 5 of 5 | 12:07-12:15 |
  | pgm1-fix6b | CAUGHT 5, 5 of 5 | 12:15-12:24 |

- **The controls** fault in every boot.
- **Against the same arms on module 3005ed52** (`records/pgc.board.txt`, `pgm0`, `pgm1`): the result lines are
  identical, cell for cell, and so are the comparison lines. Only the boot times, the stage bytes and the firmware
  hash differ. So S-27 never reached these arms, as their 3005ed52 reading already implied.
- **Bundles:** `postgres/c-repros/results/2026-10-18-vp2-rdx0/vp2-malloc`, and `postgres/mmgr-repros/results/2026-10-18-vp2-rdx0/`
  `vp2-malloc` and `vp2-pg-pools`.
- **What follows:** sql0-rev2c woke on PGF DONE at 12:24 and stopped on its own gate, "no QEMU PASS for runner
  1f099612d70a9576". The QEMU check is still queued on the lock. It is re-launched when the check reads PASS.

### The tree A flash is AUTHORISED (2026-10-19, ~13:28): the lead's own word in this lane's session, "yes, authorize"

The lead answered this lane's direct question ("do you authorize flashing tree A (caplifive_vp2c_8522d72fe.bit)?") in
this session. That covers the flash of the sealed bitstream above (sha256 ba1339bcd1f6fbb4...) and the pre-registered
session steps 1-6. Per the plan, it does NOT cover a flash back to bf3db3175 on a canary failure: that needs the lead's
word again, and the chain stops instead.

### sql0-rev2c, result (2026-10-19 13:01-, module 39d8492d): VOID again, a third board-only defect (the binary's mode)

- **What happened:** the measure-only bake read 25,856,520 B, as predicted, with a stage of 28,810,594 B. That is
  1,008 B over rev2b's, exactly 21 x 48. The boot ran, and every session exited 1 with:
  `LOG: invalid binary "./bin/postgres": Permission denied` and `FATAL: ./bin/postgres: could not locate my own
  executable path`.
- **Why:** postgres runs as `nobody` and checks with access(X_OK) that its own binary is executable. The kit's
  recipe (board-run.sh, the manifest's mode column) gives 0755 only to `*.sh` and capstone-vexec, so `bin/postgres` is
  baked 0644. On QEMU the stage keeps the source mode, 0775. capstone-vexec loads the image itself, so the kernel's
  exec check never runs; postgres's own check does.
- **Reproduced natively:** native postgres 17.5 run through the dynamic loader, which needs no execute bit, as
  capstone-vexec does not, gives the board's exact two lines at 0644 and answers SELECT 1 at 0755. A first test with
  a plain exec was void, because the kernel refused the exec before postgres ran.
- **This lane's slip:** after rev2b, this lane judged the 0644 files harmless because "vexec reads bin/postgres as an
  image". That missed postgres's own check. rev2c's pre-registration implied the runner fix was sufficient, and it
  was not.
- **Also checked, and NOT a defect:** the board kernel (6.4.14) has `# CONFIG_SYSVIPC is not set`, where the QEMU
  kit's 6.1.0 has SysV IPC. But the postgres image defines its own shmget, shmat and shmctl (the virtual runtime's
  `runtime/virtual/mapping.c`; `llvm-nm`: T at 0x1284c), so no SysV call reaches the kernel.
- **The fix,** for sql0-rev2d: a kit whose recipe keeps a staged file's execute bit when the source has it. The rev2b
  stage has three executable files: capstone-vexec, ctl.sh and bin/postgres. The runner is unchanged.
- **When:** sql0 runs on tree A after the flash session, not before it. Its prediction is QEMU's on either bitstream,
  and holding the authorised flash for a ~5 h arm is the wrong trade. rev2c's VOID boot ends first; an arm in flight
  finishes.

### The tree A session, as it will run (scripts private to the board lane, /tmp/capstone/treeA-session)

- **The scripts** are vp2's session scripts (2026-10-17), copied with names, paths and the two plan differences
  changed. sha256/16:
  - chain-treeA.sh 65a49c93dfd1321b
  - preflight-treeA.sh fbe2580a56923d89
  - flash-bit.py fab3586158b13323
  - run-treeA.sh e82ca104ebd7fff4
  - run-treeA-c7.sh 7dc25c47202ffff6
  - infra-retry.py ecc3967ffccfcc9d
  - cmp-s1gap.py 58ca15c1983878dc
  - board-free-wait.sh a5ab0cfd5bd89f21
- **The order:** the board is free, then the preflight, the flash, the canary twice (9 comparisons), the 25 arms
  against their vectors, and corpus set 7.
  - **The preflight:** the sealed sha256, the WNS -18.985 inside the band (>= -19.596), set 7's firmware
    1d2bbbf37f63f435, and the DTS pairing.
  - **The flash:** it refuses unless `caplifive_vp2_bf3db3175.bit` is still in the server's store (the flash-back
    path), and the readback must name tree A.
- **The two differences from vp2's chain, both the plan's:**
  - a canary failure STOPS with tree A resident, and does not flash back;
  - fewer than 25 arms as pre-registered stops the chain before the Linux steps.
- **The DTS gate:** `preflight-board-run.sh` blocked tree A as an unknown name. Its 65536-node row now includes
  `*_vp2c_*`. Tree A changes four core files over bf3db3175 (commit_stage, csr_regfile, cva6, ex_stage) and no Anvil
  source, so the node pool is vp2's.
  - With the set 7 firmware: tree A GO, bf3db3175 GO, `caplifive_bogus_123.bit` BLOCK.
  - With a synthetic 1021-node copy of that firmware: tree A BLOCKs on "MISMATCH".
- **Dry runs,** with stubbed steps and the real preflight, each reaching its pre-registered end:
  - all pass: `CHAIN_TREEA_STEPS123_DONE`;
  - a canary failure: `CHAIN_TREEA_STOPPED_CANARY`;
  - 24 of 25 arms: `CHAIN_TREEA_STOPPED_ARMS`;
  - a flash failure: STOP on the readback;
  - the real board-free wait beside sql0-rev2c's live driver, with a 5 s bound: STOP, busy.
  - The first dry runs found two defects in the dry-run plumbing itself: the scripts read from the override
    directory, and a quoted inline stub. Both were fixed before these.
- **Steps 4-6** (Tier 1 on module 39d8492d, wm0/wm1, and pym0 on 3005ed52) need kits naming the new bitstream. They
  are prepared while steps 1-3 run, and pre-registered before they run.

### Tree A steps 4-6, pre-registered (2026-10-19 13:45, before they run): `tier3-treeA-linux.sh`

They run after steps 1-3 end with `CHAIN_TREEA_STEPS123_DONE`, and any other end holds them. Every boot is
UNGATED (PREFLIGHT=0).

**Step 4: Tier 1 on module 39d8492d** (`board-tier12/treeA/run-tier12-rev7A.sh`).
- **The run:** rev7's run with exactly one change, the bitstream. Same stage (stage.SHA256SUMS as rev7's), firmware
  7a53759a4773c81b, driver fd38816a90c64c22 and judge. rev7 on bf3db3175 is its matched pair.
- **Predicted** as rev7 read on bf3db3175:
  - b0-stats, b0-hello and vm-contract as their oracles (0), with the identity branch;
  - `@@PARAM` diag_emerg=Y, node_flat_records=4096, supervisor_slots=1;
  - one banner, 0 refusal lines;
  - resume_satp_changed and busy_frame_refusals 0 at every read.
- **sparse-churn, which wedged on bf3db3175 with S-25's face** (N=4; S-25 cites R-57). Under tree A's R-57 fix it is
  predicted to RUN to exhaustion: `VM_EXIT:sparse-churn:1`, `VM_CONTRACT_FAIL:recycling allocation`, and the
  console's "collect skipped" at least once (the QEMU reading at 4096 nodes: collect skipped at 3,584 nodes, then
  exit 1).
  - That reading is also module 39d8492d's stopgap control on silicon, open since rev7.
  - A wedge with rev7's apertures would mean tree A's R-57 fix does not cover this face.
- **The judge's exact-counter item** stays NO DATA, as on rev7: `stale` prints no STATS line. So the judge's overall
  line reads NO DATA, and the per-item lines are the reading.

**Steps 5 and 6** run only if step 4's first three rungs read their oracles. Each arm uses its earlier runner
arguments and images (`t3-images-nb/<arm>.images.SHA256SUMS`), is scored against the same prediction, and is recorded
as `<arm>-treeA` under `results/2026-10-19-vp2c-treeA/`.
- **Step 5: wm0, then wm1,** on kit bkit7-rev2sA. That is bkit7-rev2s (module 39d8492d, budget x4) with only
  BOARD_BITSTREAM and the kit's own manifest path changed (`diff -r`). They are R-57's positive controls, and each
  wedged on bf3db3175 on both modules (N=2 per arm).
  - Predicted to run and read their QEMU predictions: wm0 MISSED 23, and wm1 control-failed 23 (its twin's).
- **Step 6: pym0 on kit bkit7A.** That is bkit7 (module 3005ed52, resume rd = x15) with the same two changes.
  - Predicted 20 of 20 at the QEMU verdict, MISSED 20 with the controls complete. On bf3db3175 the same images read
    0 of 20. That is S-27's RTL fix two-sided on silicon.
- **Dry run** (stubbed Tier 1 runner, arm and scorer):
  - all three Tier 1 rungs as oracle: wm0, wm1 and pym0 run with the kits and predictions above, then TAL DONE;
  - steps 1-3 stopped: TAL HOLD;
  - vm-contract off its oracle: STOP before any arm;
  - a driver running: STOP.

**Amendment to step 4's sparse-churn reading (2026-10-19 13:55, before step 4 runs), from the RTL lane's note on what
R-57's fix changes.** The fix does not make a faulting capability store succeed. The store's translation still
faults; the fix makes the exception reach the CAP_WB writeback, instead of the instruction retiring clean with a ghost
entry left in the store buffer's commit queue. sparse-churn has three readable outcomes on tree A:
- **(a) It runs to exhaustion:** `VM_EXIT:sparse-churn:1`, the recycling-allocation failure, and "collect skipped" at
  least once, as QEMU reads. The faults are reported and resolved as QEMU's are. This is the pre-registered
  prediction, and it closes the stopgap control.
- **(b) A clean, reported fault ends it** (a domain fault line, VM_EXIT non-zero, no wedge, the next rung or the
  boot's end reached).
  - That is R-57's fix reaching the writeback, so it is NOT read as the fix failing.
  - It is still a DIFF from QEMU, whose run resolves the faults and reaches exhaustion, so it is recorded as a new
    reading to explain. The stopgap control stays open.
- **(c) A wedge with rev7's apertures:** tree A's R-57 fix does not cover this face.

Two of tree A's known gaps can show in any step, and neither is a tree A regression:
- **S-29:** tree A has no satp flush, so a failure that looks like a translation from the wrong address space is
  S-29 unfixed. The module's caller and its context share the process's satp, so none is expected.
- **S-23:** the monitor's fetch retry still covers it, so a fetch refused with a PHYSICAL address remains S-23.

The arms phase cannot detect any of tree A's four fixes, since its 25 vectors are identical. Its pass says the images
and the RTL agree with the base, and nothing about S-27, S-28, R-57 or Q-24.

### sql0 on tree A, pre-registered (2026-10-19 14:00, before its boot): sql0-treeA (`tier3-sql-arm-rev2d.sh`)

- **The arm:** rev2c with two changes. It runs on tree A, and the kit keeps a staged file's execute bit.
  - Kits bkit7-rev2xA (boot) and bkit7-rev2xmA (measure-only) are bkit7-rev2s and bkit7-rev2m with BOARD_BITSTREAM
    caplifive_vp2c_8522d72fe and one recipe line: a source file that is executable stages 0755, else 0644.
  - The runner (1f099612d70a9576, its QEMU PASS), image 64010720f1a6b3bc, fixture record 956173e054762e91, the size
    gate and the void gates are rev2c's.
  - It runs only after the tree A Linux steps end `TAL DONE`.
- **Size:** the stage is expected at 28,810,594 B, rev2c's; modes do not change a cpio's length. The firmware is
  expected at 25,856,520 B, and the measure-only bake decides before any boot.
- **Prediction:** QEMU's, `t3-sql-trim-vm/bundle`, as for rev2b and rev2c:
  - CAUGHT 02 and 03 (cause 28 at memcpy+200);
  - MISSED 01 and 04-09;
  - both controls fault.
  - A session failing on a board-only harness difference makes the arm VOID, not a DIFF, as rev2b and rev2c were.
- **Records:** `records/sql0-treeA.board.txt` and `postgres/sql-repros/results/2026-10-19-vp2c-treeA/`.
- **Dry run:** `dryrun-tier3-sql-arm-rev2d.txt`. UNGATED (PREFLIGHT=0).

### rev8 on tree A, pre-registered (2026-10-19 13:55, before its stage, build and boot): the stopgap's control without sparse-churn, and rev7's exact counter check

- **Why it runs whatever step 4 reads:**
  - It closes rev7's exact-counter check either way: the counters are read between `ok` and `stale`.
  - If step 4's sparse-churn does not reach "collect skipped" (outcomes b or c), rev8 is the stopgap control that
    does not depend on R-57.
- **The stage** (`board-tier12/rev8/`), changed from rev7's in three files: vm-contract.sh (node_flat_records=4096,
  plus a counter read after `ok`), vm-suite-inner.sh (dense-churn in place of sparse-churn), and contract-dense.dom
  added.
  - contract-dense.dom is the compiler lane's image b6a11ff3ba726646 (lane module-resume-rd-x0 8807df747411): v9's
    contract.dom plus a `dense-churn` mode. It retires one arena, then derives children inside one already-written
    4096-byte block.
  - It passes qemu-only-csr-census (0), and vp2-silicon-gate with one attributed site: libc_exit_fini's LDC at
    0x24f9c, checked by the gate against the image (fini_array_size=0). A wrong-address control FAILs.
- **When:** `rev8-prep.sh` stages and builds only after the tree A Linux steps end TAL DONE. Step 4 reads the stage
  against rev7's record, and steps 5-6 bake in the same buildroot. The firmware hash is pinned here before the boot.
- **Predicted:** the compiler lane's QEMU readings on the twin module a5dec6a32c49b3d3 (kit qkit7-rev2) at 4096. At
  256 the launcher's own image grant is refused, so that value is VOID.
  - b0-stats, b0-hello and vm-contract read their oracles (0), and vm-suite reads 1.
  - Across `ok`: resume_sfences rises by `ok`'s own STATS faults, and resume_satp_refreshes by its steps - 1. QEMU
    reads 426 of faults=426 and 546 of steps=547. The board's own STATS values are used, since silicon's `ok` read
    steps=511 on rev7.
  - resume_satp_changed and busy_frame_refusals read 0 at all five reads.
  - **dense-churn:**
    - "collect skipped" exactly once, with retired 1 and skips 1 (QEMU: available 256);
    - progress lines up to derived=3000;
    - `domain fault cause=30` (the table exhausted) at the loop's CDERIVE, image VA 0x24238 (pc & 0xffff == 0x4238);
    - VM_EXIT:dense-churn:139.
- **The judge,** `rev8-judge.py`: PASS on a synthetic transcript of the QEMU readings, FAIL on the same transcript
  with cause 28 planted, and NO DATA on rev7's real transcript.

**The order after the tree A Linux steps:** rev8's prep and boot, then the RTL lane's S-29 bare arms (pre-registered
on receipt), then sql0-treeA. sql0-treeA's first waiter, which woke on TAL DONE, was stopped by process group before it
did anything (its out file is empty). Its measure-only bake would have shared the buildroot with rev8's bake. It is
relaunched behind the S-29 batch.

### S-29 and S-23's fetch face on tree A, pre-registered (2026-10-19 14:05, before they run): seven bare arms

- **The arms:** the RTL lane's seven images (dev 8e1df1e05a3c, `capstone/tests/rtl-smoke/satpstale-bare-2026-10-19/`),
  matched by its `images/SHA256SUMS` (all seven OK), with board_rec 0x80017000 in each.
  - Their wrapper, prologue, linker script and instruction header are byte-identical to the 25 arms'. So they run
    through the same bare runner and infra-retry, one power cycle each.
- **Predicted:** each reads its tree A vector from that folder's `results/rtl-sim-vectors.txt`:

  | Arm | Predicted vector |
  |---|---|
  | sv-stale | 0xaaaa 0xaaaa 3 0xb 0xaaaa 0x55 |
  | sv-stale-f | 0xaaaa 0xaaaa 3 0xb 0xaaaa 0x55 |
  | sv-stale-b | 0xaaaa 0xbbbb 3 0xb 0xbbbb 0x55 |
  | sv-stale-bf | 0xaaaa 0xbbbb 3 0xb 0xaaaa 0x55 |
  | sv-stale-bn | 0xaaaa 0xcccc 3 0xb 0xaaaa 0x55 |
  | sv-fetch-c | 0xaaaa 0xc 0x80006000 0x80006000 0x55 |
  | sv-fetch | 0xaaaa 1 0x80006000 0x80016000 0x55 |

  - sv-fetch is cause 1, with a PHYSICAL last value different from the faulting virtual address before it. That is
    S-23's signature from a stale translation alone. On tree B it reads cause 12 with the two equal.
  - sv-stale-bf (doubly fenced) and sv-stale-bn (no probe access) read the same on both trees. A difference on
    either is about the image or the harness, not the window.
  - Every arm ends in a spin loop at the cycle limit; the vector is the recorder's contents.
- **When:** after the tree A Linux steps end, on TAL DONE or STOP, since bare arms do not need Linux. A HOLD holds
  them. They run while rev8's firmware builds, which never touches the board.
  - Order: the five va-satpstale arms, then the satpu2 pair last (sv-fetch-c, then sv-fetch).
- **Compared** by `compare-sv.py` (count and values as 64-bit integers). The batch fails unless the table gives
  exactly 7 arms.
- **Dry run:** `dryrun-tier3-sv-batch.txt`. These arms say nothing about S-27, S-28, R-57 or Q-24.
- **The order after the Linux steps** is now: this batch (with rev8's build alongside), rev8's boot, then sql0-treeA.

### Tree A on silicon: flashed, and the canary passes 9 of 9 (2026-10-19 13:41-14:29)

- **Preflight GO (13:41:41):**
  - the sealed sha256, and the WNS -18.985 inside the band;
  - set 7's firmware 1d2bbbf37f63f435, and the DTS pairing through the new vp2c row.
- **Flash (13:41-13:43):** resident before was `caplifive_vp2_bf3db3175.bit`.
  - The upload took HTTP 201, and bf3db3175 was confirmed in the store as the flash-back path.
  - The SPI write ran 13:42:01-13:43:32. The readback after the mandatory power cycle is
    `caplifive_vp2c_8522d72fe.bit`.
- **Canary, against d93a8d66b's records** (one power cycle per bare image, through infra-retry):

  | Comparison | Pass 1 | Pass 2 |
  |---|---|---|
  | S-16 set | 18 arms, 98 reference readings, 0 not identical | the same |
  | floor | 6 arms, 77 readings, 0 not identical | the same |
  | s1gap | 3 arms, 0 functional differences | the same |

  - Pass 1 against pass 2 is identical by the same three rules.
  - `CANARY PASS 9 of 9` at 14:29:14. The 25 arms began at once.
- **The 25 arms (14:29-14:50):** all 25 read their pre-registered vectors, `VP2ARMS DONE 25 as pre-registered`.
  Each is compared by count and values with the bf3db3175 vectors the RTL lane showed identical on tree A's RTL.
- **Corpus set 7 (14:50-14:58):** on firmware 1d2bbbf37f63f435, against d93a8d66b's record: 33 reference records, 33
  new, 0 differ.
- **So steps 1-3 end `CHAIN_TREEA_STEPS123_DONE` at 14:58.** As the RTL lane noted, none of these can see tree A's four
  fixes. They show the images and the RTL agree with the base, so tree A changed nothing they exercise.

### Tree A step 4 (14:59-15:10): Tier 1 as rev7; sparse-churn did not wedge before the boot ended in a kernel panic with S-23's data-face shape (audited; RETRACTION of two claims below)

**RETRACTION (2026-10-19, same day; lane b3588262cc83, not on dev).** The first version of this section claimed two
things that do not hold. The claim-auditor refuted both:
- **"`__mmiowb_state` read on every spinlock acquire BEFORE the refusal refutes a persistent deny."** Withdrawn. Reads
  before the refusal rule out only a deny that already existed at those times. A deny created at the moment of a
  node free is untouched by them. The same argument went to the RTL and runtime lanes, and was withdrawn there too.
- **"The wedge is gone; the program went further."** Withdrawn. Nothing measures how far sparse-churn ran on either
  bitstream. A pb10 per-CPU base "the same" as this boot's, and "the memory layout is identical", were assertions;
  pb10 booted a different Image, and its boot log is not available.

**What the transcript shows** (N=1, UNGATED, PREFLIGHT=0; the auditor checked each point against
`/tmp/capstone/tier1/run3rev7A/` and the running Image):
- b0-stats, b0-hello and vm-contract read their oracles, with the identity branch, as rev7 on bf3db3175.
- **The fetch retry, then the dump.** In sparse-churn the monitor first retried an S fetch at `vm_ioctl+0xab0`, the
  instruction after the module's CSRUNV (`FRTY:01113D98`; the module base is inferred from two FRTY pcs). It then
  printed the region and CPMP dump: slot 7 = region 21 [0xAC100000, 0xC0000000) RWX, CPHT 7.
- **The refusal.** `MTVL:BC2B9440 CPMX:00000002 INJS:00000002`, where both 2s are the constant `CAPSTONE_NO_CPMP_REGION`
  (`sbi_capstone.h:93`; INJS is not an mcause on this path). The dump and MTVL belong to the load, not to the FRTY: a
  retried fetch returns before the dump (`sbi_capstone.c:3617-3618`).
- **The kernel.** Linux took `Oops - load access fault` (cause 5) at `_raw_spin_lock_irqsave+0x56` from
  `__wake_up_common_lock+0x4e`, in interrupt, and panicked.
- **The localisation.** The refused load is `lhu` of CPU 0's `__mmiowb_state`, from the Oops's own code bytes, which
  equal the running Image. Through the linear map, a4 + `__mmiowb_state` = a5 = PA 0xBC2B9440 = MTVL. CPU 0's
  per-CPU unit is [PA 0xBC2B8000, 0xBC2C7000) in this kernel.
- **The difference from pb10.** pb10's PA (0xBC2C0825) and kernel caller (`mod_node_page_state`) differ, on
  bf3db3175 with a different Image. Its CPMP dump was recorded as 48 lines against 45 here.
- **The same PA was read again AFTER the refusal.** `die()` takes `spin_lock_irqsave(&die_lock)` before printing the
  Oops line, and every acquire touches `__mmiowb_state`. That read went through slot 7 with no second refusal, so
  the refusal did not survive the monitor's handling.
  - Whether that handling re-armed the CPMP revnode trackers is open: a namespace flush across S/M if `srevroot_q` or
    `sup_active_q` was set at the fault (`ex_stage.sv:1417`, `csr_regfile.sv:2876-2878`). So this does NOT yet refute
    a persistent deny. `srevroot_q` and `sup_active_q` at the fault are the discriminator.
- **What the per-CPU locality does not show.** Per-CPU data is the first thing interrupt entry and lock acquisition
  touch, so any "first kernel access to the window after event X" account predicts it. It favours no account.
- **The wedge.** sparse-churn did not wedge before the panic, which came about 15 s after VM_BEGIN. How far it ran is
  not measured, and whether R-57's face would have recurred later is not observed. The apertures after the panic are
  residue: they match pb10's post-panic ones, and are not an S-25 reading.

### Tree A step 5, wm0 (15:10-15:20): runs to the end, MISSED 23, 23 of 23 as predicted (on bf3db3175 it wedged in its first control)

- **The run:** kit bkit7-rev2sA, 26 of 26 steps, boot completed. `vp2-malloc (virtual-wmem): {'MISSED': 23}`, 23 of 23
  cells the same as the QEMU prediction. The bundle is under
  `wireshark/wmem-repros/results/2026-10-19-vp2c-treeA/vp2-malloc`.
- **The uaf-jumbo control,** where both bf3db3175 runs wedged, now ends in a reported `domain fault cause=25`.
- **A confound:** the module differs too. On bf3db3175, wm0 wedged on modules 3005ed52 and d71f207a, and module
  39d8492d never ran wm0 there. 39d8492d adds a resume sfence.vma and a satp refresh, which could plausibly bear on a
  first-touch store.
  - So wm0 is not a clean pair for R-57.
  - The clean pairs are step 4's sparse-churn (module 39d8492d on both bitstreams) and step 6's pym0 (3005ed52 on
    both).

### Tree A steps 5-6 (15:20-16:24): wm1 runs to the end, and pym0 on module 3005ed52 reads 20 of 20 where bf3db3175 read 0 of 20; TAL DONE

- **wm1** (kit bkit7-rev2sA): 26 of 26 steps, boot completed. `vp2-nested-pools (virtual-wmem-pools):
  {'NO-READING:control-failed': 23}`, 23 of 23 cells the same as its prediction (its QEMU twin's control-failed 23).
  On bf3db3175 it wedged in its first control, on both modules. wm0's caveat applies: module 39d8492d never ran wm1
  on bf3db3175, so this is not a clean R-57 pair.
- **pym0** (kit bkit7A: module 3005ed52, resume rd = x15; images as on bf3db3175, `t3-images-nb/pym0.images.SHA256SUMS`,
  21 of 21): 21 of 21 steps, `vp2-malloc (virtual-pymalloc): {'MISSED': 20}`, 20 of 20 cells the same as the QEMU
  prediction.
  - On bf3db3175 the same images on the same module read 0 of 20: every program faulted at 0x12, S-27's clobber
    (pym0, and pym0-fence with the fence on).
  - Only the bitstream differs. **S-27's RTL fix is two-sided on silicon: N=1 on tree A against N=2 on bf3db3175.**
  - Bundle: `cpython/pymalloc-repros/results/2026-10-19-vp2c-treeA/vp2-malloc`.
- `TAL DONE 16:24:27`. Every boot UNGATED (PREFLIGHT=0).

### After the tree A Linux steps (2026-10-19 16:25)

- **The S-29 bare batch** began on board 1 at 16:24:55. Board 2's cross-board canary was in its pass 2 at the same
  time, the first parallel bare runs on the two boards (board #2 phase 1, dev 685e2c3118f2).
- **The Tier 1 prep chain** (`board-tier12/s26/tier1-prep-chain.sh`), launched on TAL DONE, does these in order:
  1. stages S-26's pair image: rev7's stage with only `vm/vm-contract.sh` swapped for the pair's bounded,
     counter-reading one;
  2. bakes it under the buildroot bake lock;
  3. restores the stage to rev7's record;
  4. builds the pair's three firmwares on that Image (wrapper aab63d5 and monitor ba9a0de, differing only in
     defines);
  5. stages and builds rev8 (node_flat_records=4096).
  - It replaces rev8-prep.sh's waiter, which was stopped by process group before it did anything. The two would
    otherwise have staged the same Tier 1 stage at once.
  - Its stubbed dry run on a copy of the stage: the bake saw the pair's script (6be39dbbf3bd427a), the stage
    returned to rev7's record (67780eebca30b73c), each firmware got only its own defines, and rev8 followed.
  - **The Tier 1 bake now also bakes `contract-dense.dom`** when the stage holds it (rev8). Without it, rev8's cpio
    check would have stopped its build.
- **The order of the remaining boots on board 1:**
  1. the S-26 pair (three short boots: current cold, variant cold, mirror warm), pre-registered with its firmware
     hashes once the chain has built them;
  2. rev8;
  3. sql0-treeA;
  4. the two cpython allocator-boundary boots.
  - The runtime lane moved the S-26 pair ahead of rev8: two short boots that decide a root cause, against
    sql0-treeA's five hours.

### S-29 and S-23's fetch face on tree A, result (2026-10-19 16:24-16:31): 7 of 7 arms read their tree A vectors

- **Board 1,** resident caplifive_vp2c_8522d72fe.bit, one power cycle each, through the 25 arms' runner. That runner is
  in the wt-b0 tree, whose console predates the `[fpga-board]` line, so the runs carry none. The board is board 1 by
  the host lock and the runner's resident-bitstream assertion.
- **Compared by `compare-sv.py`**, count and values as 64-bit integers, against the RTL lane's simulated tree A vectors
  (dev 8e1df1e05a3c): `SVBATCH DONE 7 of 7 as pre-registered`.

  | Arm | Vector |
  |---|---|
  | sv-stale | 0xaaaa 0xaaaa 3 0xb 0xaaaa 0x55 |
  | sv-stale-f | 0xaaaa 0xaaaa 3 0xb 0xaaaa 0x55 |
  | sv-stale-b | 0xaaaa 0xbbbb 3 0xb 0xbbbb 0x55 |
  | sv-stale-bf | 0xaaaa 0xbbbb 3 0xb 0xaaaa 0x55 |
  | sv-stale-bn | 0xaaaa 0xcccc 3 0xb 0xaaaa 0x55 |
  | sv-fetch-c | 0xaaaa 0xc 0x80006000 0x80006000 0x55 |
  | sv-fetch | 0xaaaa 1 0x80006000 0x80016000 0x55 |

- **What it shows on silicon,** at N=1 per arm:
  - **S-29's stale translation:** sv-stale and sv-stale-f read the stale 0xaaaa.
  - **S-23's fetch face from a stale translation alone:** sv-fetch takes cause 1 with a PHYSICAL last value
    (0x80016000) different from the faulting virtual address, where its control sv-fetch-c takes cause 12 with the two
    equal.
  - **The cross-tree controls agree with the simulation,** sv-stale-bf (doubly fenced) and sv-stale-bn (no probe
    access). That makes the other arms readings rather than harness differences.
  - Tree B (the satp flush) is predicted to change sv-stale, sv-stale-f, sv-stale-b and sv-fetch. These arms say
    nothing about S-27, S-28, R-57 or Q-24.

### S-26's silicon pair on tree A, pre-registered (2026-10-19 16:36, before the boots): three cold boots, `board-tier12/s26/run-s26.sh`

**The candidate** (the compiler lane's read, audited by the runtime lane's claim-auditor: PLAUSIBLE BUT UNPROVEN).
- csupquantum (0x7C3) resets to 0. Its only writer is the monitor's `supervised_invoke` (`sbi_capstone.c:1002`), which on
  this firmware is reached by a process-ABI STEP (`context_step`), never by a classic DOM_CALL.
- So a virtual context entered before any process-ABI STEP loads a 0 countdown. It is preempted before its first
  instruction and escapes kind 1, again and again. b0-hello is a process-ABI launch, which is why it is S-26's
  workaround; b0-stats only reads 0xFC4.
- **The RTL basis** (the RTL lane, simulation on bf3db3175): a context entering with quantum 0 retires nothing and
  escapes kind 1, cause 0, bit-identically for budgets 40 through 1,500. At 2,000 the loop completes.
- **A model divergence:** QEMU arms a constant 5 ms of virtual time at every entry, so it can never show a quantum-0
  livelock.

**The arms.** Image 89baf46451448872 is rev7's stage with only `vm/vm-contract.sh` swapped for the pair's
(6be39dbbf3bd427a). That script runs `ok` in the background, reads resume_sfences and resume_satp_refreshes at +30, +90
and +150 s, and at the bound kills a still-running `ok` and reports VM_SPIN (retval 930).
- **The firmwares** are all built on that Image, from wrapper aab63d5 (caplifive-opensbi s26-boot-quantum) and monitor
  ba9a0de (capstone-sbi s26-mirror-quantum), and differ only in defines. Each prints QNT0, 0x7C3 as boot leaves it.
  Checked in the compiled ELFs:

  | Firmware | Defines | 0x7C3 writes and reads |
  |---|---|---|
  | cur fafbc0e38fb54c60 | QNT0 print only | 1 write (supervised_invoke), 1 read |
  | var 883f3af2b9964251 | + CAPSTONE_BOOT_QUANTUM=2000000 | 2 writes, 1 read |
  | mir 1a2244dfa19f9d40 | + CAPSTONE_MIRROR_QUANTUM=1000, a census-handler write, read back as QNT1 | 2 writes, 2 reads |

- **The order:** var, then cur, then mir; each a cold boot on board 1, tree A, module 39d8492d.
- **Predicted** (`s26-judge.py`):
  - **var** (b0-stats, vm-contract): QNT0 001E8480, and vm-contract returns 0 with rev7's identity branch.
  - **cur** (b0-stats, vm-contract): QNT0 00000000, then VM_SPIN (930) with resume_satp_refreshes strictly growing
    across t30 < t90 < t150, the escape/resume loop. Flat counters would be the other reading, escaped and never
    resumed (this README's boot-3 line). Neither is established today.
  - **mir** (b0-stats, b0-hello, b0-stats2, vm-contract): QNT0 00000000, QNT1 000003E8 at each b0-stats, b0-hello 0,
    then VM_SPIN with growing counters. **VOID if b0-hello does not return 0.**
  - A spin where a return is predicted, or the reverse, is a FAIL. The judge reads it so, tested on synthetic
    transcripts of every outcome.
- **What it decides:** var returning and cur spinning, both cold with b0-stats only, isolates the quantum at the first
  CSRUNV. mir spinning after a passing b0-hello separates "the quantum" from other state the first STEP sets up.
- A claim-auditor reads the result before S-26's registry entry changes.
- **Dry run:** a stub driver writing each arm's predicted transcript gives each arm exactly its firmware, rungs,
  oracles, board 1 and tree A, and the judge reads PASS for all three. The chain's own dry run is
  `board-tier12/s26/dryrun-tier1-prep-chain.txt`. UNGATED (PREFLIGHT=0).

**rev8's firmware, pinned before its boot:** 84d5f52ec24cea30, 19,565,064 B (Image 48bd5c6254863132). The stage is
rev8's: module 39d8492d, vm-contract.sh and vm-suite-inner.sh with node_flat_records=4096, and contract-dense.dom
b6a11ff3 baked. It boots after the S-26 pair; then sql0-treeA.

### S-26's silicon pair, results (2026-10-19 16:35-16:57, board 1, tree A, module 39d8492d): var and cur as predicted; mir's prediction FAILS, and a RETRACTION of four sentences of the first version (claim-auditor)

`board-tier12/s26/run-s26.sh`, image 89baf46451448872, the three firmwares as pre-registered (each driver loaded
`fw_<sha12>.bin` of its arm's hash; objcopy of each ELF reproduces it). Each arm was a cold boot (power off, toggle,
on, then reset halt), and each log carries `[fpga-board] 1`. N=1 per arm; UNGATED (PREFLIGHT=0).

| Arm | QNT0 | Rungs | vm-contract | Verdict |
|---|---|---|---|---|
| var (883f3af2b9964251) | 001E8480 | b0-stats 0 | 0, identity branch, steps=511 faults=426 | **as predicted** |
| cur (fafbc0e38fb54c60) | 00000000 | b0-stats 0 | 930, VM_SPIN at the 150 s bound | **as predicted** |
| mir (1a2244dfa19f9d40) | 00000000 | b0-stats 0, b0-hello 0, b0-stats2 0; QNT1 000003E8 at every census | 0, identity branch, steps=42151 faults=426, 62.5 s | **prediction FAILS** (930 predicted) |

**RETRACTION (2026-10-19, same evening; lane 10e4c990a9d2, not on dev).** The claim-auditor refuted or corrected four
sentences of the first version. Three went to the runtime and RTL lanes, and the fourth to the RTL lane only; all are
withdrawn there too.
- **"b0-hello's own STEP also ran partly at 1,000": REFUTED.** `supervised_invoke` writes CAPSTONE_SUPERVISOR_QUANTUM
  unconditionally before every arm (`sbi_capstone.c:1001-1002`; in the compiled ELF a 0x1e8480 load immediately
  precedes the csrw). So every STEP arm ran at 2,000,000. b0-hello's QNT1 lines are its census, before the STEP rounds.
- **"About 80 times as often": wrong arithmetic.** 42151/511 is the ratio of entries. Non-fault entries are 511 − 426
  = 85 against 42151 − 426 = 41,725, about 490 times. "1,000 cycles" is the value written to the register, not a
  measured escape interval.
- **"cur's raw console replays var's QNT0 before its own banner": false.** cur's board-raw.txt has one banner, one QNT0
  (00000000) and no 001E8480. The text before the banner is var's last rung. So the judge's anti-replay scoping was
  never exercised on real data.
- **"A 1,000-cycle quantum does not livelock on silicon": UNPROVEN as stated.** mir's context followed b0-hello's STEP,
  so the difference from the RTL simulation (on bf3db3175) is "the quantum and/or state the STEP leaves behind". The
  arm that would separate them (1,000 written at boot, cold, no b0-hello) has not run.

**What the arms show** (the auditor's wording):
- **cur.** With no process-ABI STEP, 0x7C3 reads 0 at boot (QNT0). The compiled firmware has no other 0x7C3 writer
  reachable without a managed context: cur's b0-stats prints only the census, with no domain created. A vm-contract
  context entered in that state was re-entered about 850 times a second (resume_satp_refreshes 22,621, 73,763 and
  124,857 at t30, t90 and t150; 125,905 after the kill), with no resolved fault (resume_sfences 0), no runtime grant
  past slot 4 and no fetch retry. It was killed at 150 s.
  - **The escape kind was not recorded**, so "kind 1" is inference; a cur repeat with diag_escapes=1 would read it.
  - The same shape (slots 0-4, no FRTY, the 150 s bound) was seen on 2026-10-18 on an earlier module. This README's
    boot-3 line ("escaped and never resumed") is wrong for this shape: the context is re-entered continuously.
- **var.** Writing 0x7C3 = 2,000,000 at boot, with no STEP, let the same vm-contract complete with rev7's identity
  branch (steps=511, faults=426). var and cur differ at the binary level only inside cap_env_init (a lui/addw/csrw
  0x7c3, plus register and stack shifts). .rodata, .data and the payload are byte-identical.
- **mir.** After b0-hello's STEP, a census-written 0x7C3 = 1,000 did not prevent completion: read back 0x3E8, and the
  only other writer needs a managed-context ecall that vm-contract does not make. That is steps=42,151 and faults=426,
  in 62.5 s. It does not separate the quantum from state the STEP leaves behind.
- **The judge's QNT0 source was corrected after the var arm,** with the prediction unchanged. QNT0 prints after
  OpenSBI's banner and before Linux, outside board.txt, so the judge reads it from board-raw.txt after this boot's last
  banner. The judge's synthetic tests had put QNT0 in board.txt, where the real driver never writes it: its positive
  control was not representative. var's on-disk judge.txt still reads the first judge's "NO DATA"; re-judged, it
  reads PASS. cur and mir were judged by the corrected version.

### rev8 on tree A, result (2026-10-19 16:56-17:03, board 1): JUDGE PASS on every pre-registered item; module 39d8492d's stopgap control and rev7's exact counter check both close

Firmware 84d5f52ec24cea30 (Image 48bd5c6254863132), rev8's stage, `run3rev8`. Each log carries `[fpga-board] 1`;
UNGATED (PREFLIGHT=0).
- **The rungs:** b0-stats 0, b0-hello 0 and vm-contract 0 (the identity branch) read their oracles, and vm-suite reads 1.
- `@@PARAM` diag_emerg=Y, node_flat_records=4096, supervisor_slots=1. One banner, 0 refusal lines.
- **rev7's exact check, now made:** across `ok`, resume_sfences rose 426, equal to ok's own STATS faults (426). resume_satp_refreshes
  rose 510, equal to ok's steps − 1 (511 − 1). resume_satp_changed and busy_frame_refusals read 0 at all five reads.
  So items 3 and 4 (the resume sfence.vma and the satp refresh) run on every resume, exactly.
- **The stopgap's positive control on silicon, closed:** dense-churn printed "collect skipped" once (retired 1,
  available 256, skips 1); progress reached derived=3000; then `domain fault cause=30` at pc 0x3fb8e14238
  (pc & 0xffff = 0x4238, the CDERIVE at image VA 0x24238); VM_EXIT 139. Every value is the QEMU twin's
  (a5dec6a32c49b3d3 at 4096).
- No CPMX refusal anywhere in the boot (CPMX-DIAG: no dump). FRTY retried twice, both at 011113AC (`retire+0x220`).

### sql0-treeA, result (2026-10-19 17:03-17:18, board 1, tree A): BLOCKED by S-23's data face in its first postgres session (pb10's exact signature); no SQL reading

- **The harness worked this time:** the measure-only bake read 25,856,520 B, as predicted; firmware 9efda912a753e75d;
  stage 28,810,594 B. The boot reached the batch, and postgres served its first session (`preflight-00`).
- **The panic:**
  - The console after this boot's own load_image runs `@@STEP-BEGIN preflight-00`, `@@HB 1`, `@@HB 2`, then
    `FRTY:01713D98` (the module's base differs this boot, so that is `vm_ioctl+0xab0`, the instruction after its
    CSRUNV), then the region and CPMP dump (slot 7 = region 21 [0xAC100000, 0xC0000000) RWX, CPHT 7), then
    `MTVL:BC2C0825 CPMX:00000002 INJS:00000002`.
  - Linux took `Oops - load access fault` at `mod_node_page_state+0x40` from `unpin_user_page+0x2a` (a3
    ffffffd83c2c0825, badaddr 0xbc2c0825, cause 5), "Fatal exception in interrupt".
  - The same physical address, kernel path and a3 as pb10's two boots on bf3db3175, here with a different Image and
    program.
- **Verdict:** VOID by the chain's own gate (no bundle). It is not a harness defect this time: the first SQL session
  hit S-23's data face. sql0 cannot be measured on silicon until S-23's data face is fixed or worked around.
  - Both tree A instances (step 4's sparse-churn and this) show a fetch retry at `vm_ioctl+0xab0`, right after a
    CSRUNV return, just before the data refusal. That ordering is an observation; its relation to the refusal is not
    established.
- The runner fix (the batch path, chmod 700) and the kit fix (staged execute bits) are confirmed on silicon by this
  boot: postgres ran.

### S-26's firming arms on tree A, pre-registered (2026-10-19 17:35, before the boots): two cold boots, `run-s26.sh cur-esc var1k`

The runtime lane's two arms, which firm up the pair after the auditor.
- **cur-esc** (9e4924294f94eafe, on image 3537c881f1020464):
  - The image is today's Tier 1 stage (rev8's) with only `vm/vm-contract.sh` swapped for the pair's script plus
    `diag_escapes=1` at insmod (1c151e73cffe3224). The stage was restored to its record afterwards.
  - The firmware is cur's: the QNT0 print, no boot write.
  - Rungs: b0-stats, vm-contract.
  - **Predicted** (the runtime and RTL lanes): QNT0 00000000; VM_SPIN at the 150 s bound with resume_satp_refreshes
    growing; and every printed escape (`capstone_vm: escape ... frame kind 0x1 cause 0x0`, budgeted) **kind 1, cause 0**,
    the quantum's masked cause.
- **var1k** (2406f5a523ca00e4, on the pair's image 89baf46451448872):
  - var's firmware with CAPSTONE_BOOT_QUANTUM=1000 in place of 2,000,000. The compiled ELF has `li t1,1000; csrw 0x7c3`
    in cap_env_init. So it differs from var by that one value.
  - Rungs: b0-stats, vm-contract. Cold, with no STEP and no b0-hello.
  - **Predicted** (the RTL lane, written before it runs): the context retires nothing and spins, the same signature as
    QNT0 = 0: QNT0 000003E8; VM_SPIN at the bound with resume_satp_refreshes growing. The RTL lane's tree-A bracket is
    identical to bf3db3175's: q0 and 1,000 retire nothing (28,987), 1,500 starts the context (28,989), 2,000 completes
    (29,030).
  - **A completion instead** means silicon differs from the simulation, the first suspect being the testbench's fixed
    12-cycle delay on each of the entry walk's ~70 register transactions. It would also settle mir's ambiguity: mir's
    return would then be the quantum, not STEP state.
- **The judge:** `s26-judge.py` with the two new arms. It is tested on EDITED COPIES OF THE REAL run directories
  (board.txt and board-raw.txt), not on written fixtures:
  - cur-esc with eight kind-1/cause-0 escape lines PASS; with one kind 0x2 FAIL; with none FAIL;
  - var1k on cur's spinning run with QNT0 000003E8 PASS; on var's returning run FAIL.
  - The three real arms re-judge unchanged: var PASS, cur PASS, mir FAIL.
- Board 1, cold boots, UNGATED (PREFLIGHT=0).

### sql0's harness ran clean, and S-23 blocked it: the cpython boots after the firming arms

The cpython allocator-boundary boots follow, pre-registered separately. A kernel panic there is S-23's NO-READING, not
a cpython verdict, and a fourth instance with its MTVL is evidence the RTL lane needs.

### S-26's firming arms, results (2026-10-19 17:29-17:45, board 1, tree A, module 39d8492d): cur-esc as predicted; var1k's prediction FAILS (1,000 at boot, cold, no STEP, completes)

| Arm | QNT0 | vm-contract | Escapes | Verdict |
|---|---|---|---|---|
| cur-esc (9e4924294f94eafe) | 00000000 | 930, VM_SPIN; resume_satp_refreshes 22,182 → 73,398 → 124,590 | 16 printed (the budget), every one `frame kind 0x1 cause 0x0` | **as predicted** |
| var1k (2406f5a523ca00e4) | 000003E8 | 0, identity branch; `steps=41805 faults=426` | — | **prediction FAILS** (930 predicted, from the RTL lane's tree-A simulation) |

- **cur-esc settles the auditor's open point.** The escape kind is now observed, not inferred: the spin at quantum 0 is a
  loop of kind-1 escapes with cause 0, re-entered continuously. Its counters reproduce cur's (22,621 → 73,763 →
  124,857) to within 2%.
- **var1k answers the two questions the mirror left open:**
  - **The threshold:** with a 1,000-cycle quantum written at boot, cold, and no process-ABI STEP, the context completes
    on silicon, where the RTL simulation (tree A and bf3db3175 alike) has 1,000 retire nothing.
  - **mir's ambiguity:** var1k's 41,805 steps match mir's 42,151 (after b0-hello's STEP), so the quantum decides,
    and STEP state does not.
  - On silicon the threshold for no progress lies below 1,000. The RTL lane's first suspect for the difference is the
    testbench's fixed 12-cycle delay on each of the entry walk's ~70 register transactions.
- **S-26 at this point** (N=1 per arm, UNGATED):
  - a quantum of 0 at the first CSRUNV gives a loop of kind-1, cause-0 escapes without progress;
  - writing 2,000,000 at boot, or 1,000, removes it with no STEP;
  - the threshold is between 0 and 1,000 cycles on silicon.
- A silicon bracket (100 / 300 / 600 in one boot) is now worth a boot, by the RTL lane's own condition.

### The S-23a/S-26 monitor revision, stage 1, and S-26's silicon bracket, pre-registered (2026-10-19 18:15, before any boot)

Board 1, tree A resident (caplifive_vp2c_8522d72fe). Every boot is cold, PREFLIGHT=0, UNGATED. One script runs it all:
`chain-fix-stage1.sh` 7c324c86d7630d93.

**The monitor revision** (the runtime and RTL lanes' scope): capstone-sbi 5b5e86b on 007d81d, branch
vp2/s23-strobe-guard, pushed; wrapper aab63d5.
- **S-23a:** after a fetch retry (FRTY), one `csrsi 0x7c4, 4` (csupctl's namespace-flush strobe), issued only when a
  guard vouches for every CPMP slot. A loaded slot must name a live region that names it back. An empty slot must hold
  no capability (LCC's total type query reads 7).
  - At every FRTY the monitor prints, for each loaded slot, FRGS = (slot << 24) | (live << 20) | (maps back << 16) |
    region.
  - It names each slot it does not vouch for (FRGU).
  - FRGD = (unvouched << 16) | vouched. Its high half is the verdict: 0 strobes, anything else skips.
- **S-26:** CAPSTONE_BOOT_QUANTUM=2000000 writes 0x7C3 at boot, and CAPSTONE_QUANTUM_PRINT reports it (QNT0).
- **Not in stage 1:**
  - S-23b (the data retry, DRTY): HELD until its guard exists and the RTL lane's probes read.
  - S-30 (Linux's region below the tag store): its own firmware later, as a matched pair, gated offline first by the
    r55-regions replay. Design: carve [0xBC2D2000, 0xC0000000) linear and hold it privately. Moving the boundary is not
    the fix: swap_cpmp loads any live, unloaded slot covering an S fault, with no owner or permission filter, so an
    appended remainder would stay reachable.
  - Emptying the CPMP slot in make_hole and pop_region: a source trace of their callers is running first.
- **Kits:** `make-fix-kits.sh` ed4c2769d7acef6d. Each kit is a recorded kit with board.env lines changed and SHA256SUMS
  regenerated:
  - bkit7-rev2xF 1a12f2cbc528d416 and bkit7-rev2xmF d331cdd245a63d52: sql0-treeA's kits on the fix monitor;
  - bkit7-pb10A f3968b063170d4bb: pb10's panic kit bkit7-fix6t (module d71f207a, BOARD_STEP_S=300) with tree A's
    bitstream, old monitor 007d81d;
  - bkit7-pb10F ecd84d4842b1a941: bkit7-pb10A on the fix monitor.
- **The dry run:** it reached every last step and took every gate branch (`dryrun-fix-stage1.txt`):
  - the chain on both canary verdicts;
  - sql0-fix's VOID, HOLD and busy exits, with the real F kits passing its checks;
  - the pb10 pair's two arms, HOLD and busy, with the real kits passing;
  - run-s26's fix1 and q600 on the real firmwares through a stub driver;
  - the judge's new arms on copies of real run dirs. A var run read as fix1 FAILS (no b0-hello), and a 150 s run read
    as a bracket arm FAILS.

**1. fix1, the canary** (`run-s26.sh fix1`, firmware f40b38ae6b0f74f7 on the pair's image 89baf46451448872; rungs
b0-stats, vm-contract, b0-hello, so vm-contract runs with NO process-ABI STEP before it):
- QNT0 001E8480;
- b0-stats 0;
- vm-contract 0 on the identity branch, as var's 511 steps;
- b0-hello 0.

This shows S-26's fix on the monitor that ships. Any FRTY is followed by FRGS lines and an FRGD; predicted high half 0.
The judge passing writes CHAIN_FIX1_PASS. Anything else stops sql0 and pb10, and the bracket still runs.

**2. sql0-fix** (`tier3-sql-arm-rev2e.sh` ecfa593bc44571a3): sql0-treeA (rev2d) with ONE change, the monitor.
b0-hello's STEP precedes vbatch, so the boot quantum changes nothing there. Same runner 1f099612d70a9576, image
64010720f1a6b3bc, fixture 956173e054762e91, a measure-only bake first (cap 26,000,000 B). The runtime lane's
pre-registration is decisive either way:
- **slot 7 vouched at the FRTY before the refusal** (FRGS live = 1, maps back = 1; FRGD high half 0): the deny was false,
  a live capability cleared by a colliding broadcast. The strobe repairs it, and sql0 READS: any refusal is now followed
  by FRTY, with no Oops. Its cells are scored against t3-sql-trim-vm/bundle as rev2d's would have been.
- **not vouched** (an FRGU naming the slot, or a capability in a slot the monitor holds empty): the guard has found a
  retired capability left loaded. There is no strobe and the refusal stays, which is a containment finding of its own.

**3. pb10's matched pair on tree A** (`tier3-pb10-pair.sh` 94f2b6824c9eeb79): ffmpeg/pool, Sublet. Two cold boots that
differ only in the monitor, the fix arm first:
- **pb10-fix** (bkit7-pb10F): predicted to READ 4 of 4 as the QEMU prediction bundle t3-pb10/bundle (CAUGHT 4), under
  the same vouched / not-vouched reading as sql0.
- **pb10-treeA** (bkit7-pb10A): predicted to panic in case 01 as on bf3db3175 twice: MTVL BC2C0825, mod_node_page_state
  from unpin_user_page. Tree A does not fix the data face; sql0-treeA panicked on it.
- **If pb10-treeA does NOT panic:** the pair cannot attribute a pb10-fix READ to the strobe, and it is reported as
  one-sided.
- Results are filed as virtual-nested-pools-fix and virtual-nested-pools-treeA under results/2026-10-19-vp2c-treeA/.

**4. S-26's silicon bracket** (`run-s26.sh q600 q300 q100`): three cold boots, DESCENDING, each one variable against
var1k.
- **The firmwares:** var1k's recipe (capstone-sbi ba9a0de, wrapper aab63d5, no mirror) with CAPSTONE_BOOT_QUANTUM 600 /
  300 / 100: q600 85df5930f42dfec1, q300 95f97ad9426a7232, q100 104d3f8f629ccdee.
- **The image:** cd3eb03c37325a62 (`bracket-prep.sh` bfb58a5d3fb84c1b). It is the pair's stage with only vm-contract.sh
  swapped for `vm-contract-long.sh` 03ddf824ea953f30: bound 600 s instead of 150 s, counters read at 30 s and then
  every 60 s. The stage was restored to its record afterwards.
- **Why three boots and not one ladder:** the ladder would need a new monitor counter and a per-rung-block judge.
- **Why descending:** a killed spin followed by another rung has never been observed.
- **Why the longer bound:** var1k took 61.5 s at 1,000, so an arm can still be progressing at 150 s. The counters cannot
  separate progress from a spin: cur's spin and var1k's progress both grew at ~700 steps/s.

The model, calibrated on var (511 steps at 2,000,000) and var1k (41,805 at 1,000):

    steps(q) - 511 = 41,294 * (1000 - E) / (q - E)

E is the per-entry cost in quantum cycles.
- A completed arm reads E.
- An arm cut at the bound reads only a lower bound on E, from its last count. A cut is an INEQUALITY, never "spun".
- Every completed arm must give the same E, and a spin must have q <= E. If they disagree, the model is wrong.

Steps predicted for each E (the RTL lane's table, checked: at q = 1,000 the formula returns var1k's 41,805):

| E | q = 600 | q = 300 | q = 100 |
|---|---|---|---|
| 100 | 74,840 | 186,334 | spin |
| 200 | 83,099 | 330,863 | spin |
| 300 | 96,864 | spin | spin |
| 400 | 124,393 | spin | spin |
| 500 | 206,981 | spin | spin |
| 600 or more | spin | spin | spin |

- **The RTL lane's prediction:** E_sil lies between ~660 and 1,000. Its basis is the simulated 1,500-2,000 less the
  entry walk's ~840 cycles of fixed testbench memory delay, intersected with var1k's completion. So 600 SPINS, and 300
  and 100 with it.
- **This lane makes no numeric prediction.** The step count is the primary reading.

### S-30's matched pair, pre-registered (2026-10-19 18:27, before its boots): after stage 1 and the bracket, board 1

**The fix:** caplifive-opensbi 9bee361 (branch s30-hold-tagstore, pushed, on aab63d5). `cap_env_init` carves
[0xBC2D2000, 0xC0000000) LINEAR out of Linux's region after R-55's exact cut. The range is the tag store and the
revocation-node pool, and 0xBC2D2000 is the DTB's RAM end. The piece is held in a monitor global, never in the region
table, on every FPGA bitstream. Its decoded bounds are reported as S30H base, then end.
- **The offline gate passed before any boot:** `r55-regions/replay.py --hold 0xBC2D2000:0xC0000000` (567f7f7f0a35a760;
  the three README controls are unchanged).
  - On a2-0568f93a9, a2-respin-afcd19bea and a2-sublet-d93a8d66b: OVERLAP 0, no inexact piece, slots 112 / 82 / 116,
    all under 127.
  - Linux's top slot is nominal and decoded [0xbc200000, 0xbc2d2000). The held piece is decoded [0xbc2d0000,
    0xc0000000).
  - The check's own control fires: the range protected without the carve gives OVERLAP 1.

**The image:** 3d3e400d7b0646ff (`s30-prep.sh` 7ec87535bfc76a8a). It is the pair's stage (rev7's record, with the pair's
bounded vm-contract.sh, exactly as image-s26) plus ONE vmrun.sh case, s30-read (`vmrun-s30.case` c47423fa27ec0924). The
stage was restored to its record, and the bake checked by hash that the initramfs carries vmrun.sh.

**s30-read**, the registry's demonstration, READ-ONLY. Every read is user-mode busybox `devmem` through /dev/mem, so a
refusal reaches devmem as a signal, not a kernel fault.
- It reads the node pool's first record (0xBFF00000, two words).
- It reads the tag bytes of the monitor's dom_stack, [0x80080380, 0x80088380), whose globals include the region table:
  257 aligned 64-bit words from 0xBC2DAD60, which is 0xBC2D2D2D + 0x8038.
- Each group stops at its first refused read. retval = node refused (1) | tag refused (2).
- No write is made. A write test needs the lead's word.

**The arms** (`run-s30.sh` c0f9568c0410b84c, judged by `s30-judge.py` 5d6e125bbc535175). Two cold boots that differ
only in the wrapper. Monitor 5b5e86b, defines as fix1's, rungs b0-stats, vm-contract, b0-hello, s30-read (the read
last).
- **s30-old** (wrapper aab63d5, firmware 4104ef2029145164) predicts:
  - the canary as fix1's: QNT0 001E8480, every rung 0, no Oops;
  - no S30H line;
  - s30-read 0, with at least one nonzero tag word. That is the positive control that the reads reach tag memory, and
    it makes S-30 DEMONSTRATED for reads.
- **s30-new** (wrapper 9bee361, firmware f187c4d4f3ade200) predicts:
  - the same canary, so Linux's RAM below 0xBC2D2000 is still reached;
  - exactly two S30H lines: base <= 0xBC2D2000, end 0xC0000000;
  - s30-read 3: both groups refused at their first read, and the boot continues.
- **Reading the outcomes:**
  - s30-old reading 0 with no nonzero tag word: an instrument failure (the reads did not reach tag memory), not
    evidence about S-30.
  - s30-new failing its canary: S-30's carve broke Linux. That is a VOID about the read and a finding about the carve.
- `chain-s30.sh` b30301b8f9863d2c starts it after stage 1's CHAIN_FIX DONE. If stage 1's process is gone without DONE,
  it HOLDs.
- The dry run is in `dryrun-s30.txt`:
  - the prep and the read rung against a fake devmem (0 when reads pass, 3 when they fault);
  - both arms through a stub driver;
  - the judge's fixtures built from fix1's real run dir, with an s30-new fixture judged as s30-old FAILING;
  - the chain's HOLD branch.

### Amendment to stage 1's reading (2026-10-19 18:27, while sql0-fix runs, before any of its result was read)

The runtime lane withdrew part of the reading (dev b054c5bc926e).
- **What CPID is:** in the CPMX dumps, CPID is `i | (cpmp_region[i] << 8)` (5b5e86b :3645). "00001507" means slot 7
  holding region 21, not a revocation-node id.
- **What CPHT is:** this loop's slot index, not a tag. No tag was ever printed.
- **What FRGS reads:** the monitor's bookkeeping. It cannot say whether a deny was true or false.

The pre-registered "slot 7 vouched → a false deny, repaired by the strobe" is therefore WITHDRAWN as a grading rule.
sql0-fix and pb10-fix are graded on the narrower reading:
- **They READ:** a deny present at the FRTY was removed by the strobe or by its pipeline flush. Which of the two is not
  separable here.
- **They panic as before:** both are ruled out as the repair.

FRGS, FRGU and FRGD are reported as bookkeeping. An FRGU still names a slot the monitor cannot account for, and is
reported as such. The tag and validity of the refused slot itself need the next monitor's reads at the refusal:
`__capfield(tmp, 1)` and `__capfield(tmp, 0)` in swap_cpmp's no-region path. That build is held until the RTL lane says
whether a CCSRRW read of a CPMP CSR returns the entry's hardware tag.

### Amendment to S-30's pair (2026-10-19 18:42, before its boots): s30-new's refusal is graded by its console lines

The runtime lane asked for this. busybox 1.36.1's devmem opens /dev/mem O_RDONLY|O_SYNC and loads through mmap
(miscutils/devmem.c:73-88), with no read() path. A refused read is therefore a U-mode load access fault, and it kills
devmem, not the kernel. On the console it shows as:
- the monitor's no-region path: the CPMX dump, MTVL:<address> and CPMX:00000002;
- then INJS:00000002, which is inject_into_s(CAPSTONE_NO_CPMP_REGION) at 5b5e86b :3815, the same pair pb10's panics
  printed;
- then the shell's rc 135 (SIGBUS) or 139 (SIGSEGV) for devmem.

s30-new now also requires, in this boot's console:
- MTVL:BFF00000 and MTVL:BC2DAD60;
- CPMX:00000002 and INJS:00000002;
- both S30R refusals at rc 135 or 139.

`s30-judge.py` is now 5b8cd550ac112e51. A fixture without those lines FAILS, and one with INJS:00000005 in place of 2 FAILS.

### After sql0-fix: the pb10 pair withdrawn, the refusal read pre-registered (2026-10-19 18:35, before its boots)

**sql0-fix, first reading** (its scoring is still running; the full record follows when it ends). In vbatch's first
postgres session:
- FRTY:01713D98, with FRGS for 11 loaded slots, slot 7 among them: FRGS:07110015, region 0x15 = 21, live, mapping back;
- FRGD:0000000B, so the strobe was issued;
- then MTVL:BC2C0825, INJS:00000002, and an "Oops - load access fault".

That is the same signature as sql0-treeA. On the amended reading, neither the strobe nor its pipeline flush removed the
deny. The runtime lane's prediction (READ) is refuted.

**The pb10 pair is withdrawn before it ran.** Given sql0-fix, both arms are predicted to panic, and that reading would
not be informative. The stage-1 chain was stopped before CHAIN_PB10_GO, and sql0-fix's runner finishes on its own. In
its place, `chain-rf.sh` dc75a0e1b8a319f1 runs on board 1, each step a cold boot, after sql0-fix's runner exits:
1. **sql0-rf** (`tier3-sql-arm-rev2f.sh` 378c41c038f9d059): sql0-fix with ONE change, the monitor,
   capstone-sbi 03a1ffe (vp2/s23-refusal-read, pushed: 5b5e86b plus CAPSTONE_S23_REFUSAL_READ). Kits bkit7-rev2xR
   840c9dce535734fb and bkit7-rev2xmR 27b95ca16a39560a (`make-rf-kits.sh` e9313bc9d65804c3).
2. S-30's pair, as pre-registered.
3. S-26's bracket, as pre-registered. Its order relative to S-30 is the only change.
4. **pb10-rf** (`tier3-pb10-rf.sh` a8ca6fb27afd5025, kit bkit7-pb10R 6d4d6987a7683c51): a second instance of
   the same reading.

**The refusal read.** At a refusal no region covers, before report_cpmx_diag, each loaded slot is read once:
- the LCC type (zimm 1) first;
- only when the type is not 7, the bounds and, if they cover mtval, the validity (zimm 0);
- then a write-back of exactly that value.

It prints RFSL (slot << 16 | region), RFTY and RFVD (0xffffffff when the validity was not asked). The compiled order
was checked: read, lcc 1, lcc 3, lcc 4, lcc 0, write-back.

**Predictions:**
- sql0-rf and pb10-rf panic as before; the read changes no behaviour.
- At the refusal, slot 7 (region 21) prints **RFTY not 7**, a capability type. This is already implied: every earlier
  dump ran LCC 3, 4 and 5 on slot 7 at M after the refusal and printed its bounds rather than trapping, and every LCC
  but zimm 1 raises on NOT_CAP.
- **RFVD is the reading:**
  - 0: the physical node is dead, a true deny the monitor's bookkeeping cannot see;
  - 1: no allow term we can read explains the deny (transient, or the validity bit itself).
- **If RFTY ever reads 7** (a clear tag), the dump that follows traps at M on its cap_base. That is a predicted
  consequence, not a new failure. On the console: RFSL, RFTY:00000007 and RFVD:FFFFFFFF, then no RDID, MTVL, CPMX or
  INJS line, and the boot ends there.

The dry run, with the real kits passing every check, is in `dryrun-rf.txt`.

### Amendment to sql0-rf / pb10-rf (2026-10-19 18:37, sql0-rf booting, before any of its result was read)

**sql0-fix's record.** The run ended 18:35, "virtual boot: DID NOT COMPLETE", and its scoring is VOID: no inputs.json,
since the first session panicked. This is sql0-treeA's outcome.

**The tag question is closed from source** (dev 21d2433b165c; this replaces the three-writer census sent earlier).
- On tree A (8522d72fe), cpmp_tag_d has FOUR writers:
  - the hold (csr_regfile.sv:1189);
  - the context-entry INIT that zeroes every entry and tag (:2077);
  - the restore walk (:2091);
  - the sixteen CCSR sites.
- The RTL lane answered NO to whether a kernel load can retire during a walk. `flush_commit_o = flush_commit |
  dom_switch_busy_i` (commit_stage.sv:876) flushes everything in flight for the whole switch, so a refusal sampled on a
  cleared tag is never taken.
- Together with the dump inference, RFTY on slot 7 is predicted NOT 7, and RFVD remains the reading.

**A named branch:** the runtime lane's candidate, from the module source (lane module-resume-rd-x0 8807df747411), with
the RTL lane asked to confirm or kill it.
- **What the module source shows:**
  - Both kernel callers in the panics sit directly after a module CSRETIRE window.
  - retire() (:714-735) runs select_root, CSRETIRE, restore_root, then unpin_user_page on each page. The panics' PC is
    mod_node_page_state from unpin_user_page.
  - The remap path (:946-960) runs CSRETIRE + CSMINT, the unpin loop, then wake_up_all. sparse-churn's PC was
    __wake_up_common_lock.
  - The CPMP invalidation compare is index-only.
- **The branch:** the retire walk is still broadcasting after the instruction retires, and a context index equal to slot
  7's physical index clears slot 7 until the drain flush re-arms it. That is transient, and it fits die()'s clean
  re-read.
- **Its prediction:** RFVD reads 1, and the refusal falls inside a vm ioctl that has just retired or remapped an arena:
  the Oops's call trace names unpin_user_page or wake_up, under capstone_vm's ioctl.
- RFVD 0 contradicts the branch.

### Correction to the 18:37 amendment (2026-10-19, sql0-rf still running, none of its result read)

The runtime lane withdrew the module-source premise of the named branch as stated.
- **What the traces show:** the Oops traces print only epc, ra and one frame. That holds for sql0-treeA's, pb10's and
  tree A step 4's raw driver.log.
- **Why the premise falls:** the module has nine unpin_user_page sites, and only three follow a lifetime window: :732
  retire, :870 add, :963 remap. And __wake_up_common_lock is any wait queue. "Both kernel callers sit directly after a
  CSRETIRE window" is therefore consistent with the evidence, not shown by it.

**The branch now reads only as follows.** Under the late-broadcast candidate (already on the S-23 folder's open list),
RFVD = 1.
- The call-trace prediction ("under capstone_vm's ioctl through unpin_user_page or wake_up") is WITHDRAWN as a
  discriminator. No retire before the refusal will be recorded as observed.
- sql0-rf is unchanged and stays one variable, the monitor.
- Tying a refusal to a retire needs a module marker on the retire and remap paths. That is a new module revision, with
  its own QEMU gate and Tier 1 boot, for a later load, and only if the RTL lane's simulation says the walk outlives the
  instruction.

### sql0-rf, result (2026-10-19 18:36-18:50, board 1, tree A): the refused slot holds a tagged LINEAR capability whose node is VALID; the deny is not explained by any allow term the monitor can read

Firmware aa718bb72a85a4e9 (capstone-sbi 03a1ffe + wrapper aab63d5), kit bkit7-rev2xR, UNGATED (PREFLIGHT=0). The
result lines from this boot's console only, after its OpenSBI banner, are in `records/sql0-rf.result-lines.txt`.
- **The outcome is as predicted, the same panic:** "virtual boot: DID NOT COMPLETE", scoring VOID with no inputs.json.
  The session sequence was:
  1. FRTY:01713D98, the same pc as sql0-fix;
  2. FRGS for 11 loaded slots, slot 7 FRGS:07110015, then FRGD:0000000B (all vouched, strobe issued);
  3. the refusal read: RFSL:00070015 (slot 7, region 0x15 = 21), RFTY:00000000, RFVD:00000001;
  4. MTVL:BC2C0825, CPMX:00000002, INJS:00000002;
  5. Oops - load access fault at mod_node_page_state+0x40, ra unpin_user_page+0x2a.
- **RFTY 0 is CAP_TYPE_LINEAR.** LCC's type answer is cap_type - 1, and ariane_pkg.sv:662-671 at 8522d72fe has
  NOT_CAP 0 and LINEAR 1. The slot's hardware tag is SET and it holds a linear capability. This is as predicted (not
  7), and it is the first direct reading of the tag at a refusal.
- **RFVD 1:** the physical node answers VALID. The deny is not a dead node.
- **The CPMX dump agrees:** region 21 is in slot 7, RDEN C0000000, RDPM 7 (RWX). MTVL BC2C0825 lies inside
  [0xAC100000, 0xC0000000), and the access is a load, which perm 7 grants.
- **So, of the CPMP allow terms we can read at M, none explains the deny:** tag set, type a capability, bounds covering,
  permission granting, node valid. What is left is a term M cannot read: the slot's own validity tracker flop
  (cpmp_revnode_valid_q), or a transient that has passed by the time M looks.
  - The guarded strobe at the FRTY, which re-arms every tracker, did not prevent the refusal (sql0-fix and sql0-rf
    alike). So a cleared tracker, if that is the term, was cleared AFTER the FRTY or re-cleared after it.
- **The late-broadcast candidate** (the 18:37 amendment, as corrected at 69982fff8e4a) predicted RFVD 1. The reading is
  CONSISTENT with it and does not establish it.
  - No retire before the refusal is observed; the console prints one frame.
  - Separating it needs the RTL lane's simulation of the walk outliving the instruction, then a module marker on the
    retire and remap paths (a new module revision).
- **N=1.** pb10-rf, last in this chain, is the second instance.

### S-23b step 1 on sql0 and pb10, pre-registered (2026-10-19 18:57, before their boots); chain-rf stopped, pb10-rf withdrawn

**The monitor:** capstone-sbi f13dbc5 (vp2/s23-data-retry, pushed), which is 03a1ffe plus CAPSTONE_S23_DATA_RETRY, off by
default. It applies to an S/U load or store access fault, after the refusal read. If some loaded slot passes every
readable allow term for that access, the access is retried with NOTHING re-armed, and DRT1 is printed with mepc. The
terms are: tagged, a capability type, bounds covering mtval, perm[2] for a load or perm[1] for a store
(pmp_data_if.sv cpmp_check at 8522d72fe), and validity 1.
- Every slot is read and written back exactly, with the same id, so nothing is re-adopted. A true deny recurs, so the
  retry grants nothing the hardware does not.
- A second refusal at the same mepc within 4096 instructions prints DRTX and is injected as before.
- The per-slot re-adopt (step 2) is NOT included. It waits for the RTL lane to confirm the CCSRRW pair.
- The compiled order was checked: mstatus, mcause, then per slot read, lcc 1, lcc 3/4/5, lcc 0, write-back; then mepc,
  minstret.

**Kits** (`make-drt-kits.sh` cc086f98e4b98a03): bkit7-rev2xD 2db6294a0ad923cb, bkit7-rev2xmD 360ca0a6dffc1118
and bkit7-pb10D 67ca0fbb35a614e9. Each is the matching refusal-read kit with only the monitor changed.

**The chain:** chain-rf was stopped at 18:54 while the S-30 pair ran, and run-s30.sh finishes the pair on its own.
`chain-drt.sh` 46a9f78b1b1a2813 waits for run-s30.sh to exit, then runs, each a cold boot on board 1:
1. sql0-drt (`tier3-sql-arm-rev2g.sh` 6dcee9f5219ffff7);
2. pb10-drt (`tier3-pb10-drt.sh` 2abb23c4bbe68ac6);
3. S-26's bracket, as pre-registered.

**pb10-rf is WITHDRAWN before it ran.** pb10-drt prints the same refusal-read lines, so it gives that reading AND the
retry.

**Predictions, both arms.** At the first refusal, the refusal read is as sql0-rf's: slot 7, RFTY 0, RFVD 1. Then:
- **A transient deny** (consistent with the late-broadcast candidate): DRT1 with mepc, no DRTX, no MTVL/CPMX/INJS for
  that refusal, and no Oops. The session proceeds.
  - sql0-drt's cells are then scored against t3-sql-trim-vm/bundle, the QEMU record: 9 cells, as rev2d's.
  - pb10-drt is scored against t3-pb10/bundle (CAUGHT 4).
  - The number of DRT1 lines in the boot is itself a reading: how often the deny occurs.
- **A persistent deny** (the slot's tracker cleared and staying cleared): DRT1, then DRTX at the same mepc, then MTVL,
  CPMX:00000002, INJS:00000002 and the same Oops as sql0-rf. That points at step 2 (the per-slot re-adopt).
- **Neither**, for example a refusal at a slot that fails a readable term: no DRT1 for it, and the refusal is injected
  as before, with its RF lines saying which term failed.

The dry run, with the real kits passing every check, is in `dryrun-drt.txt`.

### Grading details for sql0-drt / pb10-drt (2026-10-19 19:00, before either boot)

- **Under "transient":** report every DRT1 in the boot with its mepc, decoded to function+offset against the kernel's
  System.map, as the S-23 folder does, and the count per boot. A single retry at mod_node_page_state is a different
  reading from dozens across many kernel paths. The count also says how often the deny occurs per session.
- **DRTX:** it is the persistent branch only at the SAME mepc as a preceding DRT1. The window is keyed on mepc, so no
  other DRTX can occur.
- **A refusal that fails the gate** prints no DRT1. It belongs to "neither", and its RF lines name the failing term.

### Tree B on BOARD 2, AUTHORISED and pre-registered (2026-10-19 19:30, before the flash)

**The lead's own word in this lane's session (19:2x):** use both boards, flashed with the latest bitstream. Asked
"which do you authorize" for board 2, the answer was "конечно прошивай самым последним бистримом" ("of course, flash it
with the latest bitstream"). The latest sealed bitstream is tree B: caplifive_vp2c_8de3c48cf, sha256 0098aa5a99c3bae2...,
WNS -16.449, sealed 17:45, the newest in the archive. So:
- Board 2 is NV-flashed from caplifive_sublet_d93a8d66b to tree B. Volatile flash does not bring up board 2's debug
  TAP, so it is NV, then a power cycle and 35 s.
- Board 1 keeps tree A for the S-23/S-26 chain, which continues there.

**Scripts** (private, /tmp/capstone/treeB2-session). They are the tree B session's (the plan, dev ddea90b284fc) with
every board-1 assumption replaced: FPGA_BOARD=2, board 2's URL file, dev's driver with the lease client
(/tmp/capstone/wt-b2, dev 10e0d5121da8), and outputs under treeB2-session.
- `chain-treeB.sh` af7626b66ea49e47, `run-treeB.sh` a8598e714df5bd70,
  `run-treeB-c7.sh` f63a136b95fc7b34, `flash-bit.py` 4a98cf5a7a081bc2,
  `preflight-treeB.sh` 5752a96ca5b9fedd.
- flash-bit.py flashes only tree B. It requires board 2's previous resident caplifive_sublet_d93a8d66b in the store as
  the flash-back path, and reads the resident back after the power cycle.
- The preflight was run for real (local checks, no board): GO, the DTS paired with set 7's firmware. A stubbed dry run
  of the chain reached CHAIN_TREEB_STEPS123_DONE.

**Steps, each gating the next:**
1. preflight;
2. flash, with its readback naming tree B;
3. the canary twice: the S-16 set, the Sublet floor and s1gap, against d93a8d66b's records, then pass 1 against pass 2.
   Predicted 9 of 9.
4. the 25 vp2 arms, each against its vector. Predicted 25 of 25: tree B's core/ is byte-identical to d427e7391's, and
   the 95-test sweep and the 152-test supervised list are identical to tree A.
5. corpus set 7 against d93a8d66b's record, as on tree A;
6. then S-29's seven bare arms (`tier3-sv-batch-treeB2.sh` b58de1b7de7ee627, board 2's copy of
   `tier3-sv-batch-treeB.sh` 5153f65bf6239fcb) against the RTL lane's TREE B column of compare-sv.py.
   Predicted: sv-stale, sv-stale-f, sv-stale-b and sv-fetch change from tree A's readings to the column, and
   sv-stale-bf and sv-stale-bn read as on tree A (ISSUES S-29).

**Stops:**
- a canary failure STOPS with tree B resident on board 2; the flash back to d93a8d66b needs the lead's word.
- not 25 of 25 STOPS before set 7.

Every result records board 2. Before any tree-B-versus-tree-A reading is compared across the two boards, the canary is
the same image set on both bitstreams, judged against the same d93a8d66b records. UNGATED (PREFLIGHT=0).

### S-30's matched pair, result (2026-10-19 18:50-19:05, board 1, tree A): DEMONSTRATED for user-mode reads, and refused by the held carve

Both arms read as pre-registered (JUDGE PASS each, N=1 per arm, UNGATED). The result lines from each boot only are in
`records/s30-{old,new}.result-lines.txt`. A claim-auditor checked the readings: each log has one OpenSBI banner, the
firmware and the judge hashes match, and the image is identical. Its four wording fixes are applied here.
- **s30-old** (wrapper aab63d5, firmware 4104ef2029145164). From user mode, busybox devmem through /dev/mem read:
  - both words of the revocation-node pool's first record (0xBFF00000: 0x0, 0x40000000);
  - the 257 words of the shadow-tag array at dom_stack's tag addresses from 0xBC2DAD60. They returned a sparse 00/01
    pattern: 23 words nonzero, the one recorded value 0x0000010001000100.
  - The bytes are not yet matched to the firmware's capability layout. The cheapest settling check is to print all
    257 words and match them against dom_stack's gp-relative capability offsets.
  - The canary rungs before the read returned 0.
- **s30-new** (wrapper 9bee361, firmware f187c4d4f3ade200). The held carve is nominally [0xBC2D2000, 0xC0000000),
  decoded [0xBC2D0000, 0xC0000000) per S30H.
  - Both reads were refused at the first address of each group, through the monitor's no-region path (MTVL:BFF00000
    and MTVL:BC2DAD60, each with CPMX:00000002). No live region covered either address.
  - Linux's top slot was exact at [0xBC200000, 0xBC2D2000) (this boot's dump: RDID 15, RDBA BC200000, RDEN BC2D2000).
  - INJS:00000002 is the constant injection code, not mcause; mcause was not printed. devmem died of SIGSEGV
    (rc 139), and the boot continued.
  - The canary rungs returned 0, including vm-contract at 511 steps.
- **The two firmwares differ only in the wrapper source.** Its side effects are the region table and +32 B of monitor
  globals. Image, monitor (5b5e86b), defines and bitstream are identical.
- **Residual, kept with the fix.** After the carve, Linux's top region and the unloaded held piece, both LINEAR,
  overlap in decoded bounds by two pages, [0xBC2D0000, 0xBC2D2000). Linearity's exclusivity there holds by the
  monitor's discipline: the held piece is never loaded or handed out. Any future use of it (a write path, the pool
  handed to a domain) must first re-carve it to an exactly representable base at or above 0xBC2D2000.
- **Not tested:** writes.

### Tree B's discriminator on board 2, pre-registered (2026-10-19 19:44, before its boots): sql0 and pb10 with the data-retry monitor

The runtime lane asked for this. Tree B's S-29 flush removes the stale I-TLB that triggers FRTY. So on tree B, whether
S-23's data face still appears says whether it sits downstream of the fetch-retry path or independent of it.

**The arms.** `chain-treeB2-linux.sh` 3dd33c6d7ce43461 runs on board 2 after S-29's seven bare arms
end DONE (if they end STOP or HOLD, it HOLDs). Each arm is a cold boot:
1. sql0-treeB (`tier3-sql-arm-rev2h.sh` 7f9a568e54177fb2);
2. pb10-treeB (`tier3-pb10-treeB2.sh` 2db50c6830ed9024).

**The kits** (`make-treeB2-kits.sh` 5ae8d3f4982efdd5) are the data-retry kits (monitor f13dbc5) with only
four things changed: tree B's bitstream, FPGA_BOARD=2, board 2's URL file, and dev's driver directory with the lease
client (the same run_baked_rungs_fpga.py, fd38816a90c64c22). bkit7-rev2xBD 1a3056a854f4fe0b, bkit7-rev2xmBD
de0d947a4112c93d, bkit7-pb10BD 9248619d2971f0c8. Results go under results/2026-10-19-vp2c-treeB/.

**Each arm reads one of three outcomes:**
1. **No FRTY and no refusal:** the data face is gone with the fetch face, i.e. it sits downstream of the fetch-retry
   path. sql0's cells are scored against t3-sql-trim-vm/bundle, pb10's against t3-pb10/bundle (CAUGHT 4).
2. **No FRTY, but a refusal** (RFSL, RFTY, RFVD, then DRT1, then DRTX or progress): the data face is INDEPENDENT of
   FRTY. Its refusal-read lines and DRT outcome are graded exactly as sql0-drt's.
3. **An FRTY:** the fetch face survives tree B's flush. That is a finding about the S-29 fix, reported first.

No prediction among the three is made; the arms discriminate.

**Amendment to tree B's discriminator (19:50, before either arm started).** Outcome 1 is reworded. Tree B differs
from tree A in its flush-all on every switcher satp write, not only in FRTY's absence, and that changes switch timing
and the caller's first translations. Outcome 1 now reads: "no FRTY and no refusal is consistent with the data face
depending on the FRTY path OR on what tree B's flush changes; N=1; cells scored against QEMU". Outcomes 2 and 3 stand.

### sql0-drt, result (2026-10-19 19:06-19:48, board 1, tree A): the data retry carries all 21 postgres sessions through S-23's deny (17 DRT1, 0 DRTX, no Oops); the SQL cells are VOID for a staging defect it exposed

Kit bkit7-rev2xD, monitor f13dbc5, UNGATED. The result lines from this boot only are in
`records/sql0-drt.result-lines.txt`, read from the reassembled UART stream after the boot's banner.
- **S-23: the transient branch, as pre-registered.** The data face was met 17 times, every time retried with nothing
  re-armed, never refused twice at the same mepc, and no Oops.
  - **The retries:** 17 DRT1, all with mepc 0x803833B6. Against the kernel's System.map that is
    `_raw_spin_lock_irqsave+0x56`. The map is controlled by sql0-rf's own Oops lines, ffffffff800a22c8 =
    mod_node_page_state+0x40 and ffffffff800ab24a = unpin_user_page+0x2a, both reproduced.
  - **The refusal read before each:** RFSL 00070015 (slot 7, region 21), RFTY 0 (LINEAR), RFVD 1. All 17 are read
    intact, so no DRT1 grades as "RFVD unread".
  - The interim message's "two garbled RFVDs" was my grep reading UART chunks split mid-line, not a console loss.
    It is WITHDRAWN.
  - **The deny's field rate:** 17 in one boot of 21 sessions.
- **Each DRT1 is followed on the NEXT console line by FRTY:017113AC,** with FRGD 0000000B, so the strobe was issued.
  FRTY prints mepc only. No FRTY in this or any earlier boot carries a physical mtval, the slot that covered it, or an
  instruction count. So the distance is one console line, and nothing more can be read.
  - 0x017113AC is not kernel text, and is left undecoded.
  - The runtime lane's hypothesis is recorded as a HYPOTHESIS, not a reading: one clear event hits the data slot and
    the slot covering the fetch, since the fetch and data checks read the same per-slot cpmp_revnode_valid_q
    (pmp_data_if.sv :149/:170/:185).
- **The SQL cells are VOID.** "virtual boot: completed; 21/21 sessions ran", then the runner's "POSITIVE CONTROL
  FAILED: the image did not answer a trivial SELECT". Every session stopped at postgres startup with `FATAL: could not
  open directory "pg_notify"`.
  - The board bake's manifest stages FILES only, so the trimmed cluster's 13 empty directories never reach the board.
    They are pg_commit_ts, pg_dynshmem, pg_logical/{mappings,snapshots}, pg_notify, pg_replslot, pg_serial,
    pg_snapshots, pg_stat_tmp, pg_tblspc, pg_twophase and pg_wal/{archive_status,summaries}.
  - Every earlier sql0 boot panicked in session 1, so the defect was never reached.
  - **The fix:** run-arm.py 8826a2ca21a33acd (lane sql-empty-dirs) records the fixture's empty directories at staging,
    and each session's step mkdir -p's them after its copy; on QEMU that is a no-op. Its QEMU check is running.

### pb10-drt, result (2026-10-19 19:48-19:57, board 1, tree A): 4 of 4 cells as the QEMU prediction (CAUGHT 4) through S-23's data face, retried once

Kit bkit7-pb10D (module d71f207a, monitor f13dbc5), UNGATED. "virtual boot: completed", vp2-nested-pools CAUGHT 4, and
compare-pilot 4 of 4 cells the same as t3-pb10/bundle (rc 0). Bundle:
`ffmpeg/pool-repros/results/2026-10-19-vp2c-treeA/vp2-nested-pools-drt`.
- **S-23 was met ONCE.** The sequence: RFSL 00070015 (slot 7, region 21), RFVD 00000001, then DRT1 at mepc 0x800A22C8.
  That is `mod_node_page_state+0x40` (the map is controlled as in sql0-drt), the exact epc of pb10's two panics on
  bf3db3175.
- The retry succeeded: 0 DRTX, 0 MTVL/INJS, no Oops. There were 2 FRTY, each with FRGD (counts from this boot's
  board-raw.txt after its banner).
- **pb10-rf was withdrawn for this arm,** and this arm gives both of its readings: the refusal read (slot 7, LINEAR
  capability per sql0-rf, node valid) and the transient branch.
- **With sql0-drt, that is two programs and two kernel sites,** `_raw_spin_lock_irqsave+0x56` and
  `mod_node_page_state+0x40`, both refusals under region 21 in slot 7. Both are carried through by a plain retry with
  nothing re-armed. S-23b step 1 is the workaround on this evidence, and step 2 stays unbuilt.
