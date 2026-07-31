# Question: should `delin` on an already-NONLIN capability trap? (we think this one is ours)

Short version: we were emitting a **redundant `delin`** in domain code, it wedges the board, and
we have removed it. Everything works without it. We are not asking you to change anything — we'd
just like to know which of two readings is right, and to flag one thing that might be worth a
look on your side.

The cap-table `gp` model you pointed us at is working well, by the way: compiler-built domains
create, enter, reach globals via `ldc rd, i*16(gp)`, and return cleanly on the captype-fixed
CVA6, and we now have real cycle numbers off three benchmarks.

## What we found

A domain executed `delin` on a capability it had loaded from the gp cap-table. The `cscall` never
returned — no output, board dead, recovered only by power-cycle. The same binary is correct under
QEMU, run through the identical controller.

Narrowed by bisection to a single instruction. `images/delin.dom` and `images/nop.dom` are the
same build except for **one 4-byte instruction at the same address**, with identical register
plumbing:

```
delin.dom:  .insn r 0x5b, 0x1, 0x3, a1, x0, x0     (delin a1)
nop.dom:    .insn i 0x13, 0x0, x0,  0(x0)          (addi x0, x0, 0)
```

That is the whole diff of the two disassemblies (`asm/delin-vs-nop.diff`).

| image | board |
|---|---|
| `nop.dom` | **returns** 9 (correct), 1812 cycles |
| `delin.dom` | **wedges** — no output, 2 attempts x 120 s |

Both return the correct value under QEMU. Each ran as the first domain of a clean boot. We ran
the `nop` variant as a size-matched control because we'd previously seen this platform change a
program's result when four instructions were added elsewhere, so a one-instruction delta needed
layout ruled out. It is ruled out.

## Why we think the fault is ours

`delin` is obviously implemented — our entry glue executes it several times in *every* domain
(`split(gp,…)`→`delin(gp)`, per global `split(t2,…)`→`delin(t2)`→`stc(t2, gp, i*16)`, and
`delin(sp)`), and the benchmarks that work all go through it.

What differs is the operand. The glue delins each cap-table entry **before** `stc`-ing it. So if
capability type survives a store/load round-trip, the entry a domain later loads is **already
NONLIN**, and our extra `delin` was a NONLIN→NONLIN no-op — i.e. we were asking for something
meaningless, and trapping on it may well be the correct behaviour.

Two further things point the same way:
- Our own emulator only tolerates this case because we patched it to. The patch comment says it
  treats `delin` as idempotent *"rather than faulting"* — so the un-patched behaviour was a fault.
- Removing the `delin` entirely fixes it: the derivation it was supposedly protecting
  (`B = A + N*N`, a `cincoffset` with rd != rs1) works fine without it, and does **not** consume
  the source. We had added it defensively on a theory that turned out to be wrong.

We've dropped it from our benchmark, which also brings that kernel back to being a faithful copy
of upstream CoreMark.

## The two questions

1. Is `delin` on an **already-NONLIN** capability meant to be legal (idempotent) or to trap?
   If it traps, this was purely our bug and the matter is closed.
2. Should a capability's **type** survive `stc` → `ldc` into a cap-table slot? Our emulator says
   the reloaded capability is still LIN, which contradicts what the glue did to it. If the RTL
   preserves the type and QEMU doesn't, then our emulator is the thing that needs fixing.

## The one thing possibly worth your attention

Whatever the answer to (1), the failure mode is a **full wedge** — M-mode appears to spin, the
board goes dead, and only a power-cycle recovers it — rather than a trap delivered to the domain
or a clean halt. If an illegal or meaningless capability operation is supposed to be catchable,
that might be worth a look. If a wedge is the intended outcome for this class, ignore this.

We can't single-step the domain ourselves (our harness detaches gdb to reach the Linux shell),
which is why this is a two-image repro rather than a trace.

## Running it

```bash
tar xzf capstone-delin-repro.tar.gz && cd capstone-delin-repro
# One image per clean boot (a second domain at the same entry VA within one boot
# hangs regardless, unrelated to this).
#   transfer images/ladder_perf_ctl and images/<delin|nop>.dom to the target
#   ./ladder_perf_ctl <image>.dom
# nop.dom prints a result line with retval=9; delin.dom prints nothing.
```

`src/` has the full build inputs if you'd rather rebuild than trust our binaries. The domain is
built `-O0`, 32 KiB code window, `-capstone-gp-captable`, shrink off, `-fno-jump-tables`, `+m`.

## Not part of this

After removing the `delin`, this benchmark still fails for a second, unrelated reason we're
bisecting on our side. That one is ours and isn't part of this question.
