# Bug corpora

Reproduction material for defects in third-party software, one directory per program.
Deliberately outside `ports/`, which holds the ports themselves and their build.

    micropython/temporal/              16 cases + 14 excluded, use-after-free and
                                       lifetime order, with repros/ and evidence/
    micropython/spatial/                4 cases, overflow inside the GC heap
    micropython/untrapped-cma-objects/  the 9 measured rows both corpora feed;
                                       generated, `gen-overview.py --check` detects drift
    sqlite/cve-repros/                  the CVE rows, one directory per row

Each case is its own directory with a `run.sh` that builds, runs its control first, and
prints a verdict against what the case records. A run whose control fails exits 75 with NO
verdict rather than reporting a result, so an infrastructure failure can never be read as a
measurement.

The MicroPython driver lives here too, `micropython/repro-lib.sh`, negative-tested by
`repro-selftest.sh`. It calls the port's build script; the dependency runs that way and
never the other, so a port can be reviewed without its corpus.

The cross-program index of which defects standard CHERI cannot see, and why, is
`docs/ref/blindspot-cases/`. Silicon defects are not here at all: those live in
`tests/fpga-repros/`, one folder per issue.
