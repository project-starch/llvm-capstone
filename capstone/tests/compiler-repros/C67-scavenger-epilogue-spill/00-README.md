# C-67 — in an epilogue with no free register, the scavenger keeps half a capability register and reloads it from the released frame

**A latent COMPILER bug, not yet seen in compiled C.** Found 2026-09-25 by an adversarial audit of the
live-source copy rule (C-32). It is independent of that rule: it reproduces with `dev`'s compiler
(`01ec8b0d322a`), and with the rule switched off on its branch.

## The shape

The function has a frame larger than 2 KiB, so releasing it needs a scratch register for the offset.
Every capability register is live at the return, so the scavenger has to spill one. At -O2, both
compilers emit:

    sd          a5, 0(sp)          # spill: the INTEGER half of c15 only
    ...
    lui         a5, 2
    addi        a5, a5, 48
    cincoffset  sp, sp, a5         # the frame is released
    ld          a5, 0(sp)          # reload: from the caller's side of sp, not from the spill
    cjalr       zero, 0(ra)

Two defects:
1. The spill is `sd`/`ld` of x15. c15's capability metadata is not saved, so a capability in c15
   comes back untagged.
2. The reload runs after `sp` has moved, so it reads another address than the one the spill wrote.

## Reach

The trigger needs no free register at all at the epilogue. Ordinary code always has caller-saved
registers free there. The two cases that could preserve every register are not reachable today:
- the Capstone clang ignores `__attribute__((interrupt))` (checked: `-Wunknown-attributes`);
- `preserve_all` was not checked.

So this is recorded as latent. The reproducer is MIR in which every register is live at the return
(`src/all-live-epilogue.mir`).

## Run it

    LLC=<llc> ./run.sh                                     # PRESENT, exit 0
    LLC=<llc> ./run.sh src/control-one-free-register.mir   # ABSENT, exit 1: c15 is not live, so
                                                           # the scavenger has a free register

Measured 2026-09-25: PRESENT and ABSENT, as above, with `dev`'s llc and with the llc of
`compiler/movc-live-source-copy`.

## What would fix it

The scavenger's spill of a GPCR register must save the whole capability (an `stc`, through the class
of the register actually being clobbered), and in the epilogue it must be restored before the frame
is released. Alternatively, the epilogue can avoid needing a register at all, for example by
releasing the frame in steps of at most 2047 bytes. Neither was tried.
