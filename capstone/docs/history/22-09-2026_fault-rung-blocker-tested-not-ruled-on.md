# The fault-rung blocker, tested instead of ruled on (2026-09-22)

`H1-platform.md:159-160` blocks half the FPGA safety matrix on a ruling:

> Until the lead rules how a fault rung earns that record, the fault half of the FPGA matrix is
> blocked and the control half is not.

and gives the reason at `:155-158`:

> It has not returned on the bitstream of record. Its deliberate out-of-bounds load ends QEMU without
> a trap, so it cannot earn the recorded QEMU pass that the preflight's check 13 demands for every
> baked rung.

That reason had never been tested. It was worth testing before spending a lead ruling on it, because
if the premise were false the blocker would dissolve at no cost. **The premise is partly false and the
conclusion still holds — for a different and more actionable reason.**

## What was run

`trapctl` — the rung that exists to be this exact positive control, verdict `0x7A05` in `res[0]` —
built through `build-ladder-fpga.sh` and run under `run-ladder-perf-qemu.sh`, which executes *"the
SAME freestanding controller the board runs … exact parity with `run_ladder_perf_fpga.py`, minus the
hardware"*. That runner hands the domain a 4 KiB region and reads `res[0..2]`, so unlike the legacy
8-byte harness it has room for the verdict.

## Result

```
[CAPSTONE] Cap mem access OOB: insn = 000636db, pc = 10156064c, rs1 = x12,
           cursor = 101580000, addr = 101580000, size = 16, bounds = (10157ffb0, 10157fff0)
[CAPSTONE] domain halted by capability fault: cause = 5, pc = 0x10156064c, tval = 0x101580000
[CAPSTONE]   x14 = 7a05
```

Three readings, in order:

1. **The deliberate out-of-bounds load DOES fault.** Cause 5, with the bounds and the offending cursor
   printed. So *"ends QEMU without a trap"* is imprecise: the fault construction is correct and fires.
2. **The rung reaches its verdict.** `x14 = 7a05` is the pass sentinel, computed and held at the moment
   of the fault — every step before the handler works.
3. **The domain HALTS.** The in-domain handler does not convert the fault into a return, so no verdict
   is ever written out and the rung cannot report.

## Why it halts — this is the actionable part

Not a rung defect, not a build defect. Our own emulator says so, at
`capstone-qemu/target/riscv/cpu_helper.c:1869-1878`:

> A domain installs no ctvec (only the monitor does, for host S/U-mode traps), so the horizontal-trap
> path below cannot deliver this fault. … Returning control to the host domain-launcher instead of
> halting requires a **monitor-side fault-return path that does not exist yet** (no `__domasync`
> handler; `DOM_REENTRY_POINT` is a stub) — tracked as the follow-up in
> `design/domain-fault-delivery-proposal.md`.

So **no fault rung can earn a QEMU pass on the current emulator**, whatever its construction. The
conclusion in `H1-platform.md` is right. Its stated mechanism is not, and the difference matters:
nobody can fix this by writing a better fault rung.

**A third option therefore exists that was not on the table**: implement
`docs/design/domain-fault-delivery-proposal.md`, which already exists (11,863 B, 2026-09-19). That
would let fault rungs earn ordinary QEMU passes and unblock the fault half **by construction rather
than by ruling** — no rule bent, no preflight edited, no hand-written marker.

## Two documentation defects found by positive control, not by reading

The first two attempts at this test produced "the handler does not fire" from a binary in which the
handler **was never present**. Both were caught by building with and without the flag and comparing
hashes — the flag is live only if the two differ.

| attempt | flag used | result | why |
|---|---|---|---|
| 1 | `INTERP_DOMAIN_MTVEC=1` | binaries **byte-identical** | `DOMAIN_GLUE` defaults to `generated` (`build-ladder-domain.sh:22`); the flag is read only under `interp` (`:126`, `:140`) |
| 2 | `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1` | binaries **byte-identical** | same cause; also not the variable the build reads |
| 3 | `DOMAIN_GLUE=interp INTERP_DOMAIN_MTVEC=1` | binaries **DIFFER** | instrument live; this is the run reported above |

So:

- **`H1-platform.md:147` is incomplete.** *"export `INTERP_DOMAIN_MTVEC=1` into that build"* does
  nothing at the default glue. It needs `DOMAIN_GLUE=interp` beside it.
- **`trapctl_fpga_app.c:5` names a different variable again** — `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1`
  — which no build script reads.

Either instruction, followed exactly, silently produces a domain with no trap vector. And
`trapctl_kernel.h` predicts precisely that failure: *"Without it the glue installs no vector, the
deliberate fault wedges, and the rung reports nothing — **which would look exactly like the handler
failing**."* It looked exactly like that, twice.

**A flag whose absence is indistinguishable from the result it would produce needs a build-time
assertion, not a comment.** Recorded; not fixed here.

## What this does NOT establish

- **Nothing about silicon.** This is the emulator. Whether the board's monitor delivers a domain fault
  is a separate question and is not answered here.
- **It does not retire the ruling.** The lead still has to choose. It changes what the choice is
  between, and removes the option nobody should pick — waiting for a better-built fault rung.
