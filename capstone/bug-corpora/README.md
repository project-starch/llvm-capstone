# Bug corpora

Reproduction material for defects in third-party software, one directory per
program. Deliberately outside `benchmarks/`, which holds the ports themselves
and their build.

    micropython/temporal/              16 cases + 14 excluded, use-after-free
                                       and lifetime order
    micropython/spatial/                4 cases, overflow inside the GC heap
    micropython/untrapped-cma-objects/  the 9 measured rows both corpora feed;
                                       generated, `gen-overview.py --check`
                                       detects drift

Each case is its own directory with a `run.sh` that builds, runs its control
first, and prints a verdict against `RESULT.txt`. A run whose control fails
exits 75 with NO verdict rather than reporting a result.

The shared driver stays with the port it builds:
`benchmarks/micropython/repro-lib.sh`, negative-tested by `repro-selftest.sh`.

The cross-program index of which defects standard CHERI cannot see, and why,
is `agent-handoff/ref/blindspot-cases/`.
