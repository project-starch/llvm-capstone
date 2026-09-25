# All twenty defects again, one program per defect, on a rebuilt QEMU — 2026-09-25

**40/40 arms passed.** Every defect completes silently in the spatial arm and
faults at the labelled stale access in the Sublet arm, as in
`../20260919-qemu-20/`. This run exists because that one cannot speak for the
tree as it stands, for two reasons, and it closes both:

- **It ran the layout that no longer exists.** `../20260919-qemu-20/` booted ONE
  binary built from `shared/defects.c`, with the case selected by the trace
  input. `4d4d5af3464e` ("One program per defect") deleted that file and replaced
  it with twenty `case.c`. Nothing in the tree had been re-run against the
  corpus as it is now.
- **The emulator was rebuilt.** Both the 09-19 run and its 09-20 repetition used
  QEMU `a02f755a…`; the binary in the checkout now hashes `ce93cb32…`. No Sublet
  arm had run on it.

    matrix.tsv    the forty result lines
    inputs.json   the sha256 of each of the twenty domains, the loader and QEMU

## Exactly one variable moved

The twenty domain binaries and the loader are not new. They are the ones a
2026-09-20 run of the per-case layout used, recovered from that run's directory
and verified against its per-arm manifests: **20 of 20 domain hashes match, and
the loader matches.** That run reported 40/40 too. So between it and this one the
inputs are byte-identical and the emulator is not, which is the comparison worth
having and the reason nothing was rebuilt.

**`clang` is recorded as `null`, and that is the cost of it.** This host no
longer carries a built Capstone LLVM, so the compiler that produced these
binaries cannot be hashed. The runner now records an input it cannot hash as
absent rather than substituting another file's hash (`run-defects.py`), and the
`llvm-nm` the share preflight needs came from the system LLVM 18 — a reader, not
a compiler, and not an input to the artifact. What this run therefore cannot
re-establish is the compiler half of the provenance; everything it says about the
emulator stands on its own.

## What the forty rows say

| | spatial | sublet |
|---|---|---|
| passed | 20/20 | 20/20 |
| outcome | `completed = 1`, no fault on any arm | fault, cause 24 on every arm |
| fault PC | — | equal to the address that boot published, on all twenty |
| delivered | — | `false` on all twenty |

## The PC half of the oracle got sharper, and is still not per-case

`../20260919-qemu-20/README.md` states its own weakness: all forty faults landed
at ONE address, because one binary held all twenty cases behind one `read_probe`.
With one program per defect the probe sits at its own address in each: **15
distinct PCs over the twenty Sublet arms** — thirteen cases have an address of
their own, five share one, two share another. So the PC now separates most cases
as well as locating the instruction. It is still the **marker** that says which
case produced a fault, and the runner still requires both.

## The negative control

Run with `--cases 0 --negative-control` against the same binaries: **2/2 oracles
fired**, exit 0 — the spatial arm because nothing completed, the Sublet arm
because no fault occurred (`cause=0, pc=0`). A suite whose oracles cannot say
FAIL proves nothing by saying PASS.

The corpus's own checks were run the same day: `tests/check-corpus.py` reports 20
cases and 0 problems, and `python3 -m unittest discover -s tests` 22 tests OK.

## What this run still does NOT show

Unchanged from `../20260919-qemu-20/`, and not weakened or improved by it:

- **Containment.** Every Sublet row reads `delivered: false`. This QEMU halts the
  domain on a capability fault rather than delivering it, so the fault is raised;
  the VM surviving it is not shown.
- **Twenty mechanisms.** Twenty upstream reports, far fewer allocator shapes —
  `../README.md` tabulates which case holds which, and `matrix.tsv` carries the
  shape per row.
- **A running interpreter.** Every case is model-consumer / real-allocator: the
  allocator is the pinned 3.13.7 `obmalloc.c`, the consumer is reduced to the
  allocator calls the upstream defect makes. The corpus says nothing about the
  defects as reached through the interpreter.
