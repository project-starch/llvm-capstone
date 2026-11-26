# R-81, R-82 and R-83 on Line B: must-fire monitors over the 63 Line B tests (pre-registered 2026-11-26, before the run)

**What is open.** ISSUES R-81, R-82 and R-83 are three ways b42c8c644's lifetime unit could stay busy forever. Each was
HYPOTHESIZED by the claim-auditor reading R-77's PRE fix, and none has been constructed:
- R-81: a read whose response never arrives;
- R-82: a record write never granted behind the store port;
- R-83: a flush while CSMINT's create channel waits.
With the PRE fix, any of them is a core hang rather than a false 25.

**The instrument** is capstone-ariane b5b2-hangmon-c 808ddfabd, which is 4bbdbc6ed plus monitors under `ifdef
SUP_TRACE` only. 4bbdbc6ed's core is b42c8c644's plus one `$display`. The monitors:
- **R81-RDMAX / R82-WRMAX:** every new maximum of the unit's read wait (rd_wait_q without a response) and of its held
  write (wr_v_q without an ack). R81-LONG / R82-LONG fire at 100,000 cycles.
- **R83-FLUSH:** flush_i while cm_q is CM_CRQ or CM_CRS. Its controls are R83-CRQ (the create request state entered)
  and R83-FLUSHSEEN (a flush while the CSMINT sequencer is not idle).

**The run:**
- All 63 tests of verif/tests/testlist_b5b2.yaml, through .sup/run-arm-x.sh.
- The Anvil prologue and PROVENANCE; S12_MEM_DELAY 12; SUP_TRACE; CAPSTONE_SIM_DEBUG_MINT, which the legacy-minting
  tests need; time_out 3,000,000.
- Counts come from each test's simulation log. A test with no log is DID NOT RUN, never a zero.

**Predictions:**
- **R83-FLUSH:** 0 in every test. This is the auditor's argument: every flush source is commit-driven, and an issued
  CSMINT is the oldest instruction.
- **R83-CRQ:** at least 1 in b5b2-sbdrain-mint, the test that executes CSMINT (-DSB_MINT). If it reads 0 there, the
  R-83 monitor is VOID.
- **R83-FLUSHSEEN:** recorded, not predicted.
- **R81-LONG and R82-LONG:** 0 in every test.
- **R81-RDMAX and R82-WRMAX:** each at least 1 in every test whose unit reads or writes a record. The pool's
  initialisation writes at reset, so every test should show R82-WRMAX. The maxima are recorded, with a band of under
  10,000 cycles.

**What this run cannot show:**
- A zero R83-FLUSH says the 63 tests never make the condition. It does not say the condition is unreachable. The
  entry's own untested cases remain: an interrupt landing at the CSMINT, debug-mode entry, and a namespace flush.
- A directed arm, CSMINT under a dense timer-interrupt storm, comes next, whatever this run reads.
- Bounded maxima say no test starved the unit. They do not show that a lost D$ response (R-81) or a never-granted
  store port (R-82) cannot happen.
