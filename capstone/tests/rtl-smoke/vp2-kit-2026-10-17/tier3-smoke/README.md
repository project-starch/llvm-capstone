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
