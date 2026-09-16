# D3 — the monitor's one plain access through an integer base, fixed and validated in simulation

2026-09-16. Closes the desk item that the consolidated board queue records as a **prerequisite** for
the R-34 bitstream rather than a follow-up to it.

## What was wrong

`capstone-sbi` `sbi_capstone.S`, in `_handle_non_ecall`'s emulated-CSR writeback — the path taken at
every `rdtime`:

    slli t5, t4, 3
    add  t5, sp, t5          <- integer add: sp is a capability, the result is not one
    sd   a0, 16(t5)          <- plain store through an untagged base

In machine mode with capmode set, the load/store unit's first clause reads that base as `NOT_CAP` and
refuses the store with mcause 24. It is invisible today because the exception is raised and dropped
(R-34), and it becomes the monitor faulting inside its own trap handler the moment delivery is fixed.

**One site, confirmed independently of the RTL lane's sweep.** A scan of all 205 lines for memory
accesses whose base register was last written by integer arithmetic returns exactly this one. The
scan's positive control is the site itself: it is a known-true case, and a scan that did not report it
would be measuring nothing. That matters here because the first version of D3's other instrument
(`tests/monitor/scan-integer-bases.py`) restricted itself to functions that adjust their frame with a
capability instruction, silently excluded this very fragment, and returned a confident zero.

## The change

On branch `d3-monitor-capability-writeback` at `2dcd3a5`, held **off** `capstone-bootstrap` so the
board drivers keep baking the monitor they pin by hash (`4274268`). Move the capability's cursor in
place instead of computing an integer address from it:

    slli t5, t4, 3
    CINCOFFSET  rd = sp, rs1 = sp, rs2 = t5
    sd   a0, 16(sp)
    sub  t5, x0, t5
    CINCOFFSET  rd = sp, rs1 = sp, rs2 = t5

`rd` equal to `rs1` is not cosmetic: in place, the source-consumption question does not arise at all.
Both encode to `19e1115b`, read back from a disassembly of the assembled object rather than assumed,
and no surrounding instruction changes. `t5` leaves the block holding the negated offset instead of
the old integer address; nothing downstream reads it.

## Where it was validated, and why nowhere else would do

**On the deployed bitstream the old form and the new one both run clean**, because the exception is
never delivered — so a pass there is not evidence. Only the R-34/R-24 delivery-fix branch can separate
them. Directed matched pair `d3-monitor-writeback.S` (frozen at `tests/monitor/`, with its list and
its readings), run against `c77c65324`, terminating in 601 cycles against a 2,000,000 timeout:

| arm | reading |
|---|---|
| capmode OFF, the old shape | stores, reads back `D3D3`, cause 0 |
| the replacement, `sp` holding a NONLIN RW capability | cause **0**, its store reads back **`B0B0`** |
| the capability before and after the in-place move | **identical** cursor, bounds, type, perm, revnode |
| the old shape with capmode SET, run last | cause **24**, store refused, `B0B0` still intact |
| traps in the whole run | exactly **1**, and it was that one |

Witnesses printed in the same run rather than assumed: a capability survives a `CSCRATCH` round trip,
so capmode is set; `mstatus` reads `0xa00000000`, so MPRV is clear and MPP is zero and the load/store
privilege is the current one, M. Both halves of the gate are satisfied.

**The old shape is the positive control and the run carries no verdict without it.** Had it not
trapped, this build would not be delivering the exception at all and the replacement's clean return
would have said nothing. Ordering follows from the same worry: the replacement runs first because it
is expected to return, and the old shape runs last because on a build without R-24's renumber a live
`NOT_CAP` clause enters debug mode instead of trapping, which is a hang that would cost every reading
not yet printed.

## Two things found on the way

**The harness's own pass sequence is an instance of the same defect.** `RVTEST_PASS` is
`sw TESTNUM, tohost, t5`, which the assembler expands to `auipc` then a store through that integer
base — so after CAPENTER it traps, the handler steps over it, `tohost` is never written and the test
times out. That is ten of the twelve tests the delivery branch's sweep leaves failing. This file exits
by minting a capability over `tohost` and storing through it, which works: the run reports SUCCESS.
The same three lines would convert those ten timeouts into real readings.

**A stale waveform was renamed into this run's output directory.** The harness tail does
`[ ! -f verilator.vcd ] || mv verilator.vcd <dir>/<testname>.vcd`, so a `verilator.vcd` left in
`verif/sim/` by an earlier traced run is moved under **today's** test name. A 9.3 GB file dated
2026-09-15 17:17 appeared as `d3-monitor-writeback...vcd` in a directory created at 14:51 today, on a
run that requested no tracing and therefore could not have produced one. The `rtl-sim` skill's
"delete the artifacts before every run" is scoped to the test name and cannot catch this: until the
rename, the file carries a different name entirely. Deleted; its own results were already distilled
into the three committed extracts in the R-34 folder.

## What is still open

The change is **not** in any bitstream and not on `capstone-bootstrap`. It lands with the R-34/R-24
pair, whose synthesis and flash are the lead's call, and the branch is what the R-34 folder's "what
the fix exposes" section should be read against.
