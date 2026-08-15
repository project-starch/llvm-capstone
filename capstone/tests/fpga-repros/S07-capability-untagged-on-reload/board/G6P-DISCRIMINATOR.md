> **STATUS CHANGED, 2026-08-15.** This document says below that the patch "has never produced a
> verdict, because the defect stopped reproducing". **That blocker is gone** — the defect
> reproduces on `caplifive_s06fixs08fix.bit`. Two caveats before running it: the recipe below
> uses `L2.dom` as the leading control, and `L2` is now itself a wedging arm, so a different
> control is needed; and the rs1-vs-rs2 question this patch was built to settle is **largely
> answered already** by the `ldc`-site instance (rs1-only guard), so its remaining value is
> narrower — confirming whether the `cincoffset` sites share that cause.

# The rs1-vs-rs2 discriminator: a 4-byte patch, built and waiting

This separates the two things mcause 25 can mean at our faulting instruction. It is **built and
verified but has never produced a verdict**, because the defect stopped reproducing before it could
be run against a control that wedges. It is written down so it can be run the moment reproduction
returns — by us or by you.

## Why a binary patch and not a recompiled probe

We first tried an in-place software probe that queried both operand types per iteration. It works —
its control returned `0x5B400000` on two boots, proving the LCC type query reports NOT_CAP on this
silicon — but the instrumented build **never reaches the loop it instruments**: it fails at
`CREATE` with `rc=11`. A matched pair in one boot settled why: the uninstrumented binary printed
its three rows twice in the same boot where the instrumented one failed twice. The ~85 added
instructions were the entire difference.

That is worth knowing independently of S-07: **this workload's behaviour is sensitive to its own code
layout**, so any in-frame instrument must be run against its uninstrumented twin in the same boot.

The patch below avoids the problem completely: **4 bytes, file size identical, every other byte of
the domain untouched.**

## The patch

At domain VA `0x1516a8` (`output_text+0xdc`), file offset `0x1426a8` = `VA - 0x10000 + 0x1000`:

```
before:  db 95 c5 18     cincoffset     a1, a1, a2
after:   db a5 05 00     cincoffsetimm  a1, a1, 0x0
```

The replacement encoding was derived from two real instructions in the same binary rather than
hand-assembled, then disassembled to confirm.

| | sha256 (first 16) |
|---|---|
| `G6.dom` (unmodified) | `f93a9188a9a4433c` |
| `G6P.dom` (patched) | `8f77d68dbb780dfb` |

## Why it discriminates

`CINCOFFSET` raises `UNEXPECTED_OPERAND` on **rs1 NOT_CAP or rs2 not-NOT_CAP**
(`capstone_flu_unit.anvil:29-31`). `CINCOFFSETIMM` checks **only rs1** and has no rs2 arm
(`capstone_flu_unit.anvil:57-61`) — verified before spending a boot, precisely so the patched arm is
not a check that cannot fire.

So on the patched build:

* **it still wedges at `0x1516a8`** → the reloaded capability is genuinely NOT_CAP. **A-family.**
  One observation is enough; no statistics needed.
* **it stops wedging while the unmodified control still wedges** → the offset operand was gaining a
  tag. **B-family**, and S-07 is then two issues rather than one.

Semantics are broken by design: with the offset gone every byte writes to `payload[0]`, so the
output is garbage. The domain still runs and still returns its stage marker, which is all the arm
needs.

## How to run it

Both domains must be in the same boot, alternating, with a control first:

```
SQLITE_STAGE_DOMS="/test-domains/L2.dom,<G6P>,<G6>,<G6P>,<G6>,<G6P>,<G6>"
```

The `G6` arms are not decoration — they are the same-boot evidence that the workload can still wedge
under the conditions being measured. Without them, a run of clean patched arms is indistinguishable
from a quiet board, which is exactly the trap that voided our first attempt at this.

**Stopping rule, fixed in advance:**

* any wedge on the patched arm → A-family, stop;
* patched arm reaches ~12 genuine executions with zero wedges **while the control has wedged at
  least once in the same series** → B-family;
* control never wedges → **the series is void**, regardless of what the patched arm did. This is not
  a technicality: it is what happened to us, and reading the patched arm alone would have produced a
  confident wrong answer.

Note each boot yields only about four genuine executions before the monitor's region pool exhausts
at `SQ: id=5` (`RGNN:00000020`), so a full series is several boots.
