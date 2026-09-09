# What does and does not serialise against a CSR write (CVA6 + Capstone, read 2026-09-09)

Facts read from the RTL while fixing R-26, kept here because the next hazard of this family will be argued from them.

- **A CSR write takes effect at commit.** The CSR unit is a one-entry buffer in issue (`issue_read_operands.sv`, the
  `fus_busy.csr` stall): a second CSR op waits, every other op does not. Younger loads/stores/DYN ops issue and execute
  before the CSR write commits.
- **Side-effecting writes flush.** `csr_regfile.sv` raises `flush_o` for `mstatus`, `satp` and the other listed CSRs;
  the controller then flushes IF/ID/unissued/EX and refetches at `pc_commit + 4`, so everything younger re-executes
  against the new value. **Capability CSR writes (`CCSRRW`) were not in that set until `9d8797560`** — the CCSRRW path
  clears `csr_we` before the flush block. Now every committed CCSRRW to CTVEC/CEPC/CSCRATCH/CPMPn flushes.
- **What opens the window.** Anything that delays the CSR op's commit while younger ops proceed: an older cache-missing
  load is the ordinary case (R-26's `ldmiss` arm: check five cycles before the write). A divider does **not**: it holds
  the shared fixed-latency unit, so the CSR op cannot issue until the divide is done, and the window is zero cycles wide
  by timing, not by design.
- **`fence`, `fence.i`, `sfence.vma` commit only when the store buffer is empty** (`commit_stage.sv`); a draining store
  moves their flush by the memory latency. Their flush reaches EX and resets the DYN unit — which is how R-27 was found:
  a flush inside a revocation-node query window orphans the node's response (`ex_stage.sv` drain, `66c4e7517`).
- **Which flushes reach EX:** `flush_csr`, the fences, exceptions and interrupts, `eret`, AMO/switch `flush_commit`.
  A branch mispredict flushes IF and unissued only (`controller.sv`).
- **The domain switcher is not exposed to a pending CCSRRW:** it reads `cscratch`/`cepc` only after it accepts the
  CALL/RETURN commit handshake, and commit is in order, so an older CCSRRW has always committed (the `r26-v2-cscratch`
  arms).
- **How to build a test that opens the window:** an older load from an untouched line, the CSR write, the younger
  instruction under test; a positive control that must trap (a load outside every entry) and a post-`fence.i` arm that
  must trap (proves the write landed). Read the verdict from exit codes, not from the harness's "SUCCESS" line, which it
  also prints at a timeout. Pass the memory-delay define on the run that builds the model and read it back from
  `work-ver/Variane_testharness__verFiles.dat` before labelling a result with a latency.

Records: `docs/history/08-09-2026_22-00-00_r26-ccsrrw-stale-read-demonstrated.md`,
`docs/history/09-09-2026_16-00-00_r25-r26-fix-cycle.md`; packages `tests/fpga-repros/R26-…/`, `R27-…/`.
