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
