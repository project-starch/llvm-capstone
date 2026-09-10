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

## Amendment, same day, still before the boot: the fill-cost pair

Written after rebuilding the `fillcost` rung and before any board arm ran, so it is still a
pre-registration and is kept separate from the table above rather than edited into it.

**Why the rung was rebuilt.** The first version was a C `for` loop over a capability-typed double
pointer. It emitted `stc`, and it measured the wrong thing: at the domain build's `-O0` everything
spills, so the body was **nineteen instructions per store**, one of which was the store. A total
dominated 18:1 by loop overhead cannot answer "what does the store cost" — backing the store out of
it would have meant stacking an assumed CPI for the overhead on top of a measured total. The monitor's
own fill is a four-instruction asm loop, so the rung is now a four-instruction asm loop.

**The control is the measurement.** `fillnop` is the same loop with the capability store replaced by
a `nop`: same iteration count, same pointer walk, same branch, **same total instruction count (242 in
the image, verified by disassembly)** — one instruction different. So

    (fillcost cycles − fillnop cycles) / 256

is the marginal cost of one 16-byte capability store, with the loop overhead cancelled rather than
assumed. Neither arm alone gives that number, which is why both are in this boot.

| arm | retval must be | instructions in the bracket | predicted |
|---|---:|---:|---|
| `fillcost` | **768** | ~1,030 | cycles above `fillnop` by 256 × the store cost |
| `fillnop` | **256** | ~1,030 | 1.1 – 1.5 cycles per instruction |

**The retval encodes two separate facts**, so a clean-looking zero cannot hide either: `+256` is the
asm loop's own counter, returned rather than assumed, and `+512` says slot 0 — seeded **non-zero**
before the loop — read back as zero, i.e. the store landed. A deleted loop reads 0. This matters
because the buffer is BSS: a rung that only checked "does it read back zero" would have passed with
the loop removed entirely.

**Predicted delta.** A per-store delta near **3 cycles** makes the reclaim a few percent of the
boundary path; near **50** makes it a quarter of it. The measured 1024-byte copy rate of 3.52
cycles/byte extrapolates to a 14,400-cycle ceiling for the whole fill, and that **overstates** it,
because a copy loads and stores where a fill only stores.

**One thing this pair cannot separate.** It measures the fill's *physical work* on the resident
bitstream, where the capability is LINEAR and the pointer is walked by an explicit `cincoffsetimm`.
The monitor's real fill walks by the **UNINIT cursor advance** instead. It does not give the reclaim
path end to end, and only the flashed bitstream can.

> **CORRECTED after the run.** This paragraph originally went on to say the monitor's loop is "one
> instruction shorter per iteration than the rung's". It is not: `C_RECLAIM` is
> `beq / stc / addi / j` and the rung is `stc / cincoffsetimm / addi / bltu` — **both four**. The
> monitor spends its fourth instruction on a second branch where the rung spends it on the pointer
> walk. Nothing downstream depended on the difference, but the sentence was wrong and is withdrawn.

**A miss that is worth recording because nothing else caught it.** The rewrite first guarded its asm
on `__riscv`. The domain target is `capstone64-unknown-elf` and does **not** predefine it, so the
build fell through to the portable C loop and produced exactly the shape the rewrite existed to
remove. The oracle could not see it — the C loop stores too, so QEMU returned the same 768 — and the
staging gate passed. Only the disassembly showed it. The guard is now `__CAPSTONE__`, and a target
that is neither the domain nor the declared native oracle is a **compile error** rather than a
silent fallback.
