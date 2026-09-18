# All twenty CPython pymalloc defects, paired arms, QEMU, 2026-09-19

**40/40 arms passed.** Every defect reachable by the pymalloc port at the 3.13.7
pin completes silently in the spatial arm and faults at the labelled stale access
in the Sublet arm.

    matrix.tsv    the forty result lines
    inputs.json   the sha256 of the binary, the loader, QEMU and the compiler

Supersedes `../20260918-qemu/`, which ran the first eight of these against an
earlier binary. Its rows are a strict subset and it is kept for the trail.

`matrix.tsv`, not the serial logs. A capture is contaminated by construction —
kernel and driver banners carry account names — and is 76k characters of which
about forty matter.

## What was run

One binary, `defects.dom`, built through the port's `PY_CORPUS_SRC` seam from
`../../shared/defects.c`. The arm is chosen at **runtime** from the loader's mode
argument, so the two arms of a pair differ in exactly one thing:

| mode | what the allocator does on free | expected |
|---|---|---|
| spatial (0) | `pym_issue`/`pym_release` keep the block's alias | the stale access succeeds, the run **completes** |
| sublet (1) | each issue and release does `sublet_give` then `sublet_take` | the stale access is a revoked alias and **faults** |

`inputs.json` records `distinct_trace_inputs: 20` and `arms: 40` — twenty
different cases really were selected, not one case forty times under twenty
labels.

## The oracle, and what it cannot tell apart

The Sublet arm requires a fault at the **published** probe address, not merely a
fault. The domain prints its case marker (`0xcf19…|case`) and both probe
addresses at the moment the defect is about to happen; the runner reads the
expected PC out of that boot's own output and compares. Nothing is hardcoded —
and this run proves that matters: the probe sits at `0x101a0db1c` here and at
`0x101a0b0bc` in the eight-case run, because adding twelve cases moved it.

**What it cannot distinguish:** all forty faults land at the same PC, because all
twenty cases share one `read_probe`. The PC proves the fault is *at the stale
access*; it is the **marker** that says *which case* produced it, and the runner
reads the probe addresses only from the text following that marker. Neither alone
is sufficient and the runner requires both.

## The negative control

`../20260919-qemu-20-negative-control/` is the same binary — compare the
`defects.dom` hash in both `inputs.json` files — fed an input the domain refuses.
Neither arm then performs the defect, and **all four oracles reported FAIL**: the
spatial arms because nothing completed, the Sublet arms because no fault occurred
(`cause=0, pc=0`). A suite whose oracles cannot say FAIL proves nothing by saying
PASS.

Run it with `--negative-control`, which inverts the exit status: 0 means every
arm failed as it must.

## One arm had to be made to return

The first attempt at this run stalled. A boot wedged at kernel time 0.70 s,
before login, and because `run_guest` defaults to `timeout_multiplier=12` the
suite sat on it for eleven minutes until it was killed by hand. The runner now
bounds every boot (90 s login, 90 s command, multiplier 1) and treats a boot that
produced neither the completion marker nor a fault as **infrastructure**, exiting
75 rather than recording a FAIL that would read like a defect failing to
reproduce. A suite of forty arms cannot contain an unbounded one.

## Two things this run does NOT show

**Containment.** Every row reads `delivered: false`. This QEMU **halts** the
domain on a capability fault rather than delivering it, so the run shows the
fault is raised, not that the VM survives it. The PostgreSQL corpus demonstrates
delivery on a build carrying the local-trap-delivery change; this one does not.

**Twenty independent mechanisms.** Twenty *upstream reports* were reproduced, but
they reduce to **nine** allocator shapes — eight of the twenty are the same
free / reuse / stale read. That is a finding rather than a defect in the corpus:
one revocation mechanism covers a class upstream has been fixing one module at a
time for over a year. `matrix.tsv` carries the shape per row, and `../README.md`
tabulates which case holds which.

## Fidelity

Every case is **model-consumer / real-allocator**: `obmalloc.c` from the pinned
3.13.7 is real and unmodified but for the port's patches; the consumer is reduced
to the allocator calls the upstream defect makes, in the same order, because
reaching them in place needs a running interpreter and the port does not put one
in a domain. Each case directory's `PROVENANCE.md` quotes the upstream hunk and
says what was reduced.
