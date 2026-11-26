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

## The 63 tests, read (2026-11-26)

The run: s2pre-wt `.sup/hm`, rtl 808ddfabd, dirty=no. verFiles defines: `+S12_MEM_DELAY=12+SUP_TRACE+CAPSTONE_SIM_DEBUG_MINT`.
All 63 tests produced a log and every one ended SUCCESS. Counts are from each test's `.iss`, read by an exact-token
Python count. A directory with no `.iss` would exit 2.

| monitor | result |
|---|---|
| R83-FLUSH | **0 in all 63** (as predicted) |
| R83-CRQ | 37 in all, 3 in b5b2-sbdrain-mint (the positive control: the create channel is entered), 1 in each of 34 other tests whose setup mints the kit root by CSMINT |
| R83-FLUSHSEEN | 0 in all 63 (recorded): no flush met the CSMINT sequencer anywhere but idle |
| R81-LONG / R82-LONG | **0 in all 63** (as predicted) |
| R81-RDMAX | at most 1,221 cycles (b5b2-srvcost and -cold, the probe reads of a cold record), 246 in the crevint and crevsv tests, 0 in 7 tests that make no lifetime read |
| R82-WRMAX | at most 60 cycles (the crevint and crevsv tests), 0 in 27 tests |

- **One prediction missed:** "R82-WRMAX >= 1 in every test" read 0 in 27 of the 63. The counter counts the cycles a
  write waits without its ack, so a write granted in its first cycle reads 0. The prediction misread the counter, not
  the unit. The counter does fire (36 tests, up to 60 cycles), so it says nothing for or against R-82.
- **What it shows:** in none of the 63 Line B tests does the unit wait long on a read or a write, and no flush ever
  meets CSMINT's create channel.
- **What it does not show:** that R-81, R-82 or R-83 is unreachable (the limits above). R-83's interrupt case runs
  next: capstone-ariane 066c97947, b5b2-csmint-irq at TICK 160, 80, 40 and 20, pre-registered in its header before
  any run.

## R-83's interrupt case, read (2026-11-26)

The run: s2pre-wt `.sup/hmirq`, rtl 066c97947 (808ddfabd + the test), dirty=no, the same defines. Four arms, safest
first, each pre-registered in the test's header before any run:

| TICK (mtime ticks) | completed mints | last rd | exceptions | timer interrupts | R83-CRQ | R83-FLUSHSEEN | R83-FLUSH | result |
|---|---|---|---|---|---|---|---|---|
| 160 | 0x40 | 0x13f | 0 | 386 | 65 | 0 | 0 | SUCCESS, 95,590 cycles |
| 80 | 0x40 | 0x13f | 0 | 507 | 65 | 0 | 0 | SUCCESS, 96,962 cycles |
| 40 | 0x40 | 0x13f | 0 | 692 | 65 | 0 | 0 | SUCCESS, 99,499 cycles |
| 20 | 0x40 | 0x13f | 0 | 749 | 65 | 0 | 0 | SUCCESS, 99,979 cycles |

- **All four arms read as predicted.** 64 mints complete with no exception under 386 to 749 interrupts, and R83-FLUSH
  is 0.
- **No flush ever met the CSMINT sequencer outside idle**, over 2,334 interrupts and 260 entries of the create channel
  (R83-FLUSHSEEN 0). That fits the auditor's argument that an interrupt is taken on a younger instruction, after the
  issued CSMINT has left the sequencer.
- **The monitor's parts are each live:** flush_i is the signal on which the LSU's EDGE-FLUSHPEND monitor fires, and
  the sequencer leaves idle 65 times per arm. Only their conjunction is absent, and the conjunction is the thing being
  tested.
- R81-RDMAX at most 186 cycles and R82-WRMAX at most 35; both LONG monitors are 0.
- **Not tested:** debug-mode entry at a CSMINT, and a namespace flush (an srevroot write) racing one. R-83 stays
  HYPOTHESIZED; this narrows it, it does not close it.
