# The monitor's CSCHECKR/W and CSKINDS emulation (native route on C.5a/E5), as an RTL simulation arm

MEASUREMENT ONLY. The emulation, `capstone_cscheck.S` on capstone-sbi `native/cscheck-emu` (e04ecb826b5c), needs
CAPCREATE, and QEMU does not decode it. So the emulation has not run anywhere yet. This arm runs it the way
`../vp2-kit-2026-10-17/csmint-emu/va-csmint-mon.S` ran the CSMINT emulation.

The arm takes the real path. S issues the instruction and the M trap entry builds the monitor's frame. It
calls `cscheck_dispatch`, then `csmint_dispatch`. The emulation writes rd into the frame. It uses the same
kit headers and `link.ld`. It `#include`s `capstone_cscheck.S` and `capstone_csmint.S` from that capstone-sbi
commit.

The readings are pre-registered in the header of `va-cscheck-mon.S`, before any run: 13 rows plus the final
read-back. Recompute symbol values with `llvm-nm` on the built ELF. Defaults: `CSC_STORED_NONLIN` 2 (the
RTL's stored enum, Q-35) and `CSC_INDEX_BITS` 20 (E5).

The arm is for the RTL lane to run on the E5 tree; the board lane does not run anything on the RTL lane's
worktrees.
