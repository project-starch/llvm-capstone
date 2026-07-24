#!/usr/bin/env bash
# Stage-0 QEMU proof for ladder rung 7 (CoreMark 1.01 matrix): build in the
# silicon config and run in a pure-cap domain on QEMU with gp fabrication OFF.
# The domain must return the same crc16 the native oracle computes.
#
# WHY a dedicated wrapper (vs. `run-ladder-qemu.sh coremark_matrix`): the silicon
# image pins all globals at base+0x1000, so domain .text must fit the 4 KiB PCC
# window. CoreMark's matrix kernel is ~4.7 KiB of .text at -O0 (the ladder
# default) and overflows the window. It is ~1.5 KiB at -Os and fits with room to
# spare. -Os is safe here because the kernel is almost entirely integer math (one
# gp-delivered global, delinearised once); the oracle-match assertion below would
# catch any optimisation-induced miscompile loudly. A benchmark run at -O0 is not
# a meaningful CoreMark anyway. Everything else is the standard silicon config.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DOMAIN_OPT_LEVEL=-Os
exec bash "$SCRIPT_DIR/run-ladder-qemu.sh" coremark_matrix
