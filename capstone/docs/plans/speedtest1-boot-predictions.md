# speedtest1 boot — predictions written BEFORE the run

**2026-09-10, board lane.** Every figure here is committed before the boot. A reading that was not
predicted is a finding; a prediction adjusted afterwards is not evidence.

## What the boot runs, and why it does NOT wait on the flash

Nothing in this workload reaches R-30 or R-31, so it runs on the **resident**
`caplifive_r25r26r27_66c4e7517.bit`. The bench lane recommended decoupling it from the flash and the
lead agreed. If a flash happens for its own reasons the boot can ride along; it is not a reason for
one.

**Set `FPGA_BITSTREAM` explicitly.** Several drivers still carry stale defaults, and a wrong one makes
the resident-silicon guard hard-stop rather than warn.

## Predictions

Instruction counts are measured under emulation with instruction counting on, and are **exact**.
Cycle ranges use the corrected on-silicon spread of **1.13–6.44 cycles per instruction** at 25 MHz —
corrected today from a glossary line that said 2.0–3.2 and was contradicted by its own table.

| arm | instructions | predicted seconds | verdict |
|---|---:|---|---|
| `k800` control | — | — | must read **4**, or the boot is VOID |
| `parsenumber --size 1` | 45,972,687 | 2 – 12 | **instruction count + the four phase lines** |
| `orm --size 1` | 232,147,258 | 11 – 60 | hash `465769 f3699caa…` |
| `main --size 1` | 691,582,718 | 31 – 178 | hash `112006 38bb59fd…` |
| baseline `parsenumber` | 29,735,656 | — | ratio 1.546 |
| baseline `orm` | 174,691,784 | — | ratio 1.329 |
| baseline `main` | 544,373,079 | — | ratio 1.270 |
| `fillcost` rung | ~1,024 | — | **the measurement**; a reading near zero means the loop was optimised away |

**⚠ `parsenumber`'s hash CANNOT FIRE.** It is `0 0e12171d…` at sizes 1, 5 and 20 identically, because
the leading field counts result rows and that testset returns none. A broken run produces the same
value. **Judge that arm on its instruction count and its phase lines, never on the hash.**

**The falsifier, in the right direction.** Board cycles must be **HIGHER** than the instruction count,
by the cycles-per-instruction factor. **A board cycle count at or below the instruction count means
the arm did not do the work.** (An earlier version of this falsifier was backwards, because the
figures it rested on were host timestamp counts rather than guest cycles.)

**A reading above 6.44 cycles per instruction is a finding about the memory system, not a failed arm.**

## Ordering — cheapest first, wedge-capable last

`k800`, then `parsenumber`, `orm`, `main`, then the three baseline arms, then the instret image, then
the two counter probes.

* **The instret image goes after everything of value.** Adding a `minstret` bracket to a rung flipped
  it to a deterministic miscompute on this silicon in July, and the surviving suspect was the store
  itself. Its hash is the mitigation: match and we gain the instruction split; mismatch and we lose
  one arm and gain a second data point on an open defect.
* **The two counter probes are last** because they are the only invocations that can wedge the
  machine, each in its own invocation so a gated CSR kills only itself.

## Settings that must not be left at default

`SQLITE_STAGE_TIMEOUT` defaults to **90 s** and `SQLITE_IDLE_S` to **30 s**, and the domain emits
nothing between entry and return because the report lands only when the host prints it. Both are set
from the predictions above, not left at default. The driver's abort-marker path also raises a
"was not staged" hard stop on a domain fault return, which is false and discards every arm behind it.
