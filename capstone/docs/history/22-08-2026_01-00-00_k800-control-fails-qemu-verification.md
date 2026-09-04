# The default known-good control fails QEMU verification — and the controls file was already stale

**Date:** 2026-08-22
**Status:** UNEXPLAINED. One QEMU run with full output capture settles it; the lock was held by
another lane at the time.

## What happened

Preparing the S-10 acceptance ladder, `verify-and-stage-rung.sh` was run for three rungs. Two
passed and wrote content-bound markers; **the control did not**:

```
k800   QEMU FAIL: expected 'retval = 4'      no marker written
wb0    PASS   marker 63b5296361b9 == staged .dom
wb1    PASS   marker c526e0b11996 == staged .dom
```

`k800` is the row in `ref/known-good-controls.md` described as *"The default control"*, and the
one row with the most current evidence — 10/10 in a single boot on 2026-08-18.

**The oracle is not the problem.** `k800_compute()` is self-contained: four iterations each
incrementing `ok`, plus `pad[0] = 1`, returning `ok + pad[0] - 1u` = `4 + 1 - 1` = **4**. No
globals, no capability operations, no dependencies. The arithmetic is right.

So the failure is in what the domain did, not in what it should return — and the run reached the
QEMU stage rather than failing to build.

## Why it matters more than one rung

`known-good-controls.md` **declares itself stale in its own header**:

> *"Every row below reads `last verified 2026-08-06`, against bitstreams that have since been
> replaced at least three times … This matters because `preflight-board-run.sh` BLOCKS a run whose
> first rung is not listed here, so the gate keeps passing on evidence that is three bitstreams out
> of date: a gate whose condition is always satisfied."*

That was already a warning. This is the warning coming true: the file's most current row does not
reproduce today. **A control that cannot be verified is not a control**, and a boot led by one
looks valid and is not — the same failure mode as running a ladder with no positive control.

Three readings, and choosing between them without evidence would be the error this whole exercise
is about:

1. the row is stale in a way that matters, and `k800` genuinely no longer returns 4;
2. `k800` is board-only in some respect nobody recorded, so a QEMU failure is expected and the
   verify step is the wrong gate for it;
3. something changed under it — toolchain, glue, domain window — since 2026-08-18.

## Consequence for the S-10 acceptance run

**Blocked, on two independent things, neither of which is the board.**

1. **No verified control.** Not running Boot A behind a control that was just watched failing.
2. **C13 cannot be satisfied for `wb2`.** The gate demands a QEMU pass bound to the `.dom`'s
   sha256; `wb2` is documented board-only, because QEMU has no write buffer and its capability
   store is one atomic 16-byte-plus-tag operation, so the reorder under test **cannot occur
   there**. The positive control — the arm whose firing makes the boot non-void — is the one arm
   that can never earn a marker. There is no `PREFLIGHT_ALLOW_*` for C13.

Both are the project lead's to resolve. A second agent independently reached the same conclusion
on C13 and put it well: **two agents concurring is not authorisation.**

## A methodological failure of my own, recorded because the artifact caught it and the status line did not

The verification loop printed `exit=0` for all three rungs, **including the one that failed**. The
`exit=$?` came after a `| tail -6`, so it captured `tail`'s status rather than the script's — the
"never filter between a gate and its exit status" trap, committed while auditing someone else's
gates.

What caught it was the **artifact**: `k800` has no `.qemu-pass` marker, and the marker is bound to
the `.dom`'s sha256 so it cannot be produced by touching a file. **The artifact told the truth
where the status line lied.** That is the argument for content-bound markers over mtime, and it is
now demonstrated rather than argued.

The same `tail` also discarded the actual retval, which is why this note says UNEXPLAINED rather
than naming the value. The diagnostic cost of truncating output is not hypothetical.

## What settles it

One `verify-and-stage-rung.sh k800` with the full output captured — no pipe between the script and
the log. If it returns something other than 4, reading 1 or 3 applies and `known-good-controls.md`
needs the row corrected. If it returns 4 this time, the first run was disturbed and the disturbance
itself needs explaining, since another lane was running QEMU against the same `rootfs.ext2` write
lock in the same window.
