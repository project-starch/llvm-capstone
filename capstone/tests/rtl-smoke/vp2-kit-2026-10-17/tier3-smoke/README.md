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
- Over the six subsets, sq0 (sqlite/engine-repros vp2-malloc) reads 18 of 33 cells equal to its prediction. The other
  15 carry S-27's register-2 fault (tval 2, or a5 among the faulting instruction's operands).
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
