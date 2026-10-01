# PostgreSQL original-layout Capstone inner-reuse qualification

Three complete PostgreSQL 17.5 `work.sql` processes passed the same 2,000-row
input and exact 22-row SQL-output oracle as the [Sublet
arm](../postgres-reuse-sublet-20260928/README.md), in one persistent
Capstone/Linux guest with 262,144 nodes. Each repetition used a fresh copy of
the pinned native 16-byte-MAXALIGN cluster. The original PostgreSQL memory
contexts reported 54,032 successful chunk handouts, 51,177 releases and
45,002 observed start reuses per process. All 32-bin histograms agree and
reconcile, with `error=0`.

The observer sits at the memory-context allocation boundary. It counts
explicit frees, moved reallocations, and implicit releases at context reset
and deletion. PostgreSQL's aligned-allocation wrapper delegates to those
operations and is counted once. An in-place realloc preserves its lifetime.
The index advances on successful new handouts, so the histogram describes
observed reuse distances, not a fixed-follow-up retirement cohort.

The [raw archive](capstone-raw.tar.gz) contains points, VM identity, commands,
full stdout/stderr and process records. The [SDK build
manifest](build-manifest.json) hashes every input object and the final image.
`python3 validate.py` checks the three full oracles, fresh-cluster declaration,
image hash, exit status, phases, node capacity and histograms. The build was
made from a committed, clean recipe revision.

This completes the Capstone original-layout denominator for PostgreSQL. The
CheriBSD spatial control has 54,004 handouts and 44,974 reuses; its small count
difference reflects platform allocation choices. The protected PoisonCap
backend has no complete oracle, so the PostgreSQL four-arm violin is still
blocked. The 45,002-versus-43,705 observed reuse counts in the two Capstone
arms alone do not establish a memory-cost or working-set advantage.
