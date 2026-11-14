# Mapping block state test

Run `bash capstone/kernel/tests/sublet-blocks/run.sh` from the repository root.
The script compiles the state header extracted from patch 0040 with the host C
compiler. It checks partial removal, last-page revocation, a narrowed renewed
reference, tagged and live operands, depth 1, bounds, ownership, refill and a
refused revocation. The revocation callback is simulated. This is a state test,
not execution of the RISC-V CREVOKE instruction.

The negative control removes the depth-1 condition from that same header. The
whole-block munmap through a depth-2 reference then succeeds. The unchanged
oracle must fail, and the runner rejects a control that passes.

The Linux build check applies `series` followed by `series.s6-mappings` to Linux
6.1. Build with `CONFIG_RISCV_CAPSTONE_TASK=y` and the usual single-hart
configuration. The handle and return interfaces are described in
[the mapping design](../../../docs/design/sublet-mappings.md).
