# S-12 no longer kills the board: the fault returns a code naming its own cause and site

## What changed

The SQLite silicon domain built with `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1` installs an
in-domain trap vector in the entry glue (`start-gp-captable-interp.S`), before cap-init. On the
board it produced, on the first draw:

    SQ: G/enter   ENT0:00000001  ENT1:00000001  ENT2:F643D21D
    SQ: H/return
    SQ: X/fail
    SQ: obs=4131639837          (= 0xF643D21D)
    slt did not run             rc=1

Decoded by the layout the glue itself documents:

    bits 31..28  0xF   -> this run TRAPPED; the word is not a result
    bits 27..22  25    -> mcause 25
    bits 21..0   0x03D21D -> (mepc - _start) >> 2

`_start` is at VA `0x10000`, so `mepc = 0x10000 + 0x03D21D*4 = 0x104874`, and in this image:

    10486c  stc           a4, 0x0(a5)
    104870  ldc           a4, 0x0(a0)
    104874  cincoffsetimm a4, a4, 0xb0     <- mepc

That is the S-12 window, and the fault site is now **reported by the domain itself** rather than
reconstructed from a last-writer-wins debug latch and a hand-computed DBAS subtraction — the step
that produced a retraction earlier in this campaign.

Three draws for three reports is worth noting against the trap-OFF baseline's 3/3 and 3/4 wedge
rates: the handler does not appear to reduce the fault rate, it just stops it being fatal. At n=3
that is an observation, not a measured rate.

## Why this matters more than the datum

Until now a capability fault inside a domain took the core with it: one boot yielded one bit, which
is why establishing the one-byte localisation cost roughly 35 boots. The fault is now survivable.
That changes three things at once:

1. **Many observations per boot** instead of one. Arms that previously needed four boots each can
   share a boot.
2. **The fault site and cause come back as data**, so classification no longer depends on the
   latch, on `HALT_MUX_READS=0`, or on the DBAS arithmetic.
3. **A returning fault can be repeated within one entry**, which is the precondition for measuring
   a rate rather than sampling a coin once per boot.

## What it does NOT show, and one caution

`slt did not run`: the fault aborts the query, so this build reports the fault instead of
completing the workload. It is fatal to the run, no longer fatal to the session. Under QEMU the
same image completes normally with `SLT-SUMMARY ... completed=1`, so the early return is a silicon
behaviour, not a build defect.

The handler is documented in the glue as trading fault-proofing for information: the two-instruction
version touched no memory, this one reads the frame and writes through the region capability, so a
fault caused by a bad `sp` or a bad region cap would re-fault and wedge. The downside is bounded by
the previous status quo, but a wedge on this build is therefore NOT evidence that the handler is
broken.

Also note this build is a **new baseline**: the image is 40 bytes larger and everything shifts by 96
bytes, so the fault site moves from `0x104814` to `0x104874`. The pinned base and all patched arms
of the localisation campaign belong to the trap-OFF build and are not interchangeable with it.
sha256 `b316decabbc9e04d…`, pinned at `~/fpga-artifacts/s12-base-trapon.dom`.
