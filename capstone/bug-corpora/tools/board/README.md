# The vp2 board kit for Tier 3 (`tools/boardvm.py`)

`run-virtual-cases.py --board-kit <kit>` runs a corpus arm's steps in one vp2 board boot. boardvm.BoardBatch stages
the images, writes the guest script for `/test-domains/vbatch/`, and calls the kit's recipe.

The recipe, `board-run.sh` (copy it into the kit beside `bake-manifest.sh`):
1. bakes a private board image from a manifest of the board controls' base files plus the stage;
2. builds the firmware;
3. boots it through the baked-rung driver, with b0-stats as the board control first;
4. leaves the driver's per-rung transcript in `<work>/board.txt`.

boardvm then reads only the batch rung's segment from that transcript, never the raw capture.

A kit directory:

| file | what |
|---|---|
| `board.env` | the variables below, `KEY=value` (sourced by the recipe; keep values literal) |
| `board-run.sh`, `bake-manifest.sh` | copies of the two scripts here |
| `adapter/capstone-vexec`, `adapter/module/capstone_vm.ko` | the board launcher and module |

`board.env`:

| variable | meaning |
|---|---|
| `BOARD_BITSTREAM` | the resident bitstream's name (preflight and the bundle's identity) |
| `BOARD_BASE_MANIFEST` | `src\|dst\|mode` lines for the board control: b0-hello.dom, b0run.sh, capstone-exec, capstone-job (the vp2 Tier 1 bake's base set) |
| `BAKE_REPLACE_KO` | (optional) the process-ABI capstone.ko the board control loads as /capstone.ko |
| `BOARD_FW_RECIPE` | the private-firmware recipe (`tests/rtl-smoke/supmon-2026-10-03/build-fpga-fw.sh` with FW_WRAP/FW_MON/FW_IMAGE_DIR) |
| `BOARD_FW_WRAP`, `BOARD_FW_MON`, `BOARD_MONITOR_COMMIT` | the wrapper and monitor worktrees, and the commit the monitor must sit at, clean |
| `BOARD_FW_DEFINES` | the firmware's defines; on vp2 they include `CAPSTONE_VP2_FETCH_RETRY` (S-23) |
| `BOARD_FW_CAP` | the firmware size cap in bytes (the DECLARED-bake cap: 26000000) |
| `BOARD_DRIVER_DIR`, `BOARD_DRIVER_SHA` | the rtl-smoke directory holding `fpga_driver/`, and the pinned sha256/16 of `run_baked_rungs_fpga.py` |

The checks:
- `python3 tools/test_boardvm.py` checks boardvm offline. It parses the guest script and the controller, refuses a
  transcript carrying a replayed batch marker, reads a rung without END as incomplete, and records the identity keys.
- `BOARD_STUB=1 bash board-run.sh <stage> <work> <timeout>` prints the bake, firmware and driver commands and runs
  nothing.
- `BOARDVM_STUB_TRANSCRIPT=<file>` makes BoardBatch.execute take a canned transcript instead of the board.
- The first real use is a pre-registered smoke boot: the virtual positive control alone. It goes through the board-run
  skill like any other boot.
