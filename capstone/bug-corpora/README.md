# Bug corpora

Reproduction material for defects in third-party software, one directory per program.
Deliberately outside `ports/`, which holds the ports themselves and their build.

    sqlite/cve-repros/       the CVE rows, one directory per row
    postgres/mmgr-repros/    defects in consumers of PostgreSQL's memory contexts
    cpython/pymalloc-repros/ defects in consumers of CPython's small-object allocator

Each case is its own directory with a `run.sh` that builds, runs its control first, and
prints a verdict against what the case records. A run whose control fails exits 75 with NO
verdict rather than reporting a result, so an infrastructure failure can never be read as a
measurement.

The cross-program index of which defects standard CHERI cannot see, and why, is
`docs/ref/blindspot-cases/`. Silicon defects are not here at all: those live in
`tests/fpga-repros/`, one folder per issue.
