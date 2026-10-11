# PostgreSQL Sublet inner-reuse qualification

Three complete PostgreSQL 17.5 `work.sql` processes passed the same
2,000-row input and exact SQL-output oracle in one persistent Capstone/Linux
guest. Each started from a fresh copy of the native 16-byte-MAXALIGN cluster
and used 262,144 provisioned nodes. The inner memory-context Sublet adapter
reported exactly 54,032 chunk handouts, 51,177 releases and 43,705
observed start reuses per process. All three 32-bin histograms agree,
reconcile exactly and have `error=0`.

The [raw archive](capstone-raw.tar.gz) contains the points, VM identity,
commands, full stdout/stderr and process records. The
[SDK build manifest](build-manifest.json) hashes every input object and the
final image. `python3 validate.py` rechecks the raw output, fresh-cluster
declaration, image hash, phases, node capacity and reuse histogram. This
is an actual complete PostgreSQL backend workload, not a trace replay.

The [CheriBSD spatial control](../postgres-reuse-spatial-20260928/README.md)
has the same SQL oracle but 54,004 inner handouts; platform-specific
allocator decisions differ slightly, so these raw counts are not a paired
fraction. The [Capstone original-layout control](../postgres-reuse-capstone-spatial-20260928/README.md)
now has a qualified inner histogram; the protected PoisonCap backend still
lacks one. Consequently this is not a
four-arm reuse plot or a claim that Sublet wins PostgreSQL memory behavior.
