# Where a thing goes in this repository

## The rule, in one sentence

**What the build reads goes in the port. What is a measurement goes in the corpus. What is
a narrative goes in `docs/`.**

## The three homes

| Directory | Holds | Answers |
|---|---|---|
| `capstone/ports/<program>/` | `fetch-*.sh`, `patches/`, `port/`, `adapted/`, `tools/`, `build-*-silicon.sh`, `census-*.sh`, README | does this software build and run under capabilities |
| `capstone/bug-corpora/<program>/` | one directory per case, each with its own `run.sh` and recorded result | what does the hardware fail to catch |
| `capstone/benchmarks/` | beebs, coremark, rv8 | how much does it cost |

`benchmarks/` used to hold both kinds. For five of its eight entries "benchmark" was the
wrong word: they measure no time, they answer compatibility, and they are all shaped alike
inside. Splitting them was cheapest while only `sqlite` had to move; with the MicroPython,
JerryScript, WAMR and mruby ports in flight it would have cost five times as much.

## Three things that are decided, so nobody re-derives them

**A protection port lives apart from the compatibility port.** A port directory answers
whether the program runs under capabilities; the Sublet port of its allocators answers what
the discipline costs it, and the paper counts those lines. So `ports/<program>/sublet/` holds
the primitives and the patch that protect, applied on top only when asked for, and nothing in
`adapted/`, `patches/` or `port/` is Sublet's. The unprotected build must never read a file
from `sublet/`.

**`capstone/tests/capstone-test-env.sh` does not move. 341 files reference it.** It is the
one path in this repository that everything sources, and a tidy-up that relocates it for
symmetry costs more than every other move here put together.

**Case material never lives inside a port.** A port directory that grows a `cases/` or
`cve-repros/` is the thing this layout exists to prevent: it makes "does it run" and "what
did we measure" share a branch, a review and a diff.

## Moving anything

Rewrite the full path everywhere, then prove none is left with a search you have first
checked against a control string. Two cases need deciding by eye rather than substituting:
a changed leaf name still reached relatively (`../old-name/x`), and the path written
without its leading segment (`benchmarks/foo` for `capstone/benchmarks/foo`). Both
survived a move that had just reported itself clean; the second was 52 references in 33
files.

## Still open

`capstone/tests/` is fourteen directories plus twenty-three loose files, and the loose ones
are five unrelated kinds: release gates, drivers, scanners, baselines, and one stray test
input. Splitting `gates/`, `tools/` and `results/` out of it is the next move. It is held
back only because those scripts run daily, so it wants a window in which no lane is inside
them, not because the split is in doubt.
