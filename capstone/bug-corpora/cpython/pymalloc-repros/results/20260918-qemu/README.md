# Eight CPython pymalloc defects, paired arms, QEMU, 2026-09-18

**16/16 arms passed.** Every one of the eight defects completes silently in the
spatial arm and faults at the labelled stale access in the Sublet arm.

    matrix.tsv    the sixteen result lines
    inputs.json   the sha256 of the binary, the loader, QEMU and the compiler

`matrix.tsv`, not the serial logs. A capture is contaminated by construction —
kernel and driver banners carry account names — and is 70k characters of which
about sixteen matter.

## What was run

One binary, `defects.dom`, built through the port's `PY_CORPUS_SRC` seam from
`../../shared/defects.c`. The arm is chosen at **runtime** from the loader's mode
argument, so the two arms of a pair differ in exactly one thing:

| mode | what the allocator does on free | expected |
|---|---|---|
| spatial (0) | `pym_issue`/`pym_release` keep the block's alias | the stale access succeeds, the run **completes** |
| sublet (1) | each issue and release does `sublet_give` then `sublet_take` | the stale access is a revoked alias and **faults** |

`inputs.json` records `distinct_trace_inputs: 8` — the eight arms of each mode
really did select eight different cases, rather than running one case eight
times under different labels.

## The oracle, and what it can and cannot tell apart

The Sublet arm requires a fault at the **published** probe address, not merely a
fault. The domain prints its case marker (`0xcf19…|case`) and both probe
addresses at the moment the defect is about to happen; the runner reads the
expected PC out of that boot's own output and compares. A relink cannot turn the
check into a tautology, because nothing is hardcoded.

**What it cannot distinguish:** all eight cases share one `read_probe` function,
so all eight faults land at the same PC. The PC proves the fault is *at the stale
access*; it is the **marker** that says *which case* produced it, and the runner
reads the probe addresses only from the text following that marker. Neither
alone is sufficient and the runner requires both.

## The negative control

`../20260918-qemu-negative-control/` is the same binary — byte-identical
`defects.dom` hash — fed an input the domain refuses. Neither arm then performs
the defect, and **both oracles reported FAIL**: the spatial arm because nothing
completed, the Sublet arm because no fault occurred (`cause=0, pc=0`). A suite
whose oracles cannot say FAIL proves nothing by saying PASS.

Run it with `--negative-control`, which inverts the exit status: 0 means every
arm failed as it must.

## Two things this run does NOT show

**Containment.** Every row reads `delivered: false`. This QEMU **halts** the
domain on a capability fault rather than delivering it, so the run shows the
fault is raised, not that the VM survives it. The PostgreSQL corpus demonstrates
delivery, on a build that carries the local-trap-delivery change; this one does
not, and nothing here should be read as a containment result.

**Eight independent mechanisms.** Eight *upstream reports* were reproduced, but
they reduce to five allocator shapes — cases 0, 1, 3 and 4 all come down to
free / reuse / stale read. That is a finding rather than a defect in the corpus:
one revocation mechanism covers a class upstream has been fixing one module at a
time. `matrix.tsv` carries the shape per row so the distinction is not lost.

## Fidelity

Every case is **model-consumer / real-allocator**: `obmalloc.c` from the pinned
3.13.7 is real and unmodified but for the port's patches; the consumer is reduced
to the allocator calls the upstream defect makes, in the same order, because
reaching them in place needs a running interpreter and the port does not put one
in a domain. Each case directory's `PROVENANCE.md` quotes the upstream hunk and
says what was reduced.
