# Bug corpora

Reproduction material for defects in third-party software, one directory per program.
Deliberately outside `ports/`, which holds the ports themselves and their build.

    sqlite/cve-repros/       19 CVE rows, one directory per row
    postgres/mmgr-repros/    2 defects in consumers of PostgreSQL's memory contexts
    cpython/pymalloc-repros/ 20 defects in consumers of CPython's small-object allocator
    sqlite/cve-repros/       the CVE rows, one directory per row
    cpython/pymalloc-repros/ defects in consumers of CPython's small-object allocator
    sqlite/cve-repros/     the CVE rows, one directory per row
    postgres/mmgr-repros/  defects in consumers of PostgreSQL's memory contexts
    postgres/mmgr-repros/  defects in consumers of PostgreSQL's memory contexts
    sqlite/cve-repros/       the CVE rows, one directory per row
    cpython/pymalloc-repros/ defects in consumers of CPython's small-object allocator
    ffmpeg/pool-repros/      defects in consumers of FFmpeg's AVBufferPool/AVRefStructPool
    httpd/apr-pool-repros/   1 defect in a consumer of APR's pools
    memcached/allocator-repros/ 2 defects in consumers of memcached's per-thread object cache

Each case is its own directory recording what the defect is and where it came from. The
file names differ by corpus: sqlite and postgres cases carry `before.c`, `oracle`,
`PROVENANCE.md` and `README.md`; pymalloc cases carry `case.c`, `case.json` and
`PROVENANCE.md`, with the corpus-wide description in its own `README.md`.

One sqlite row is a recorded non-reproduction rather than a case: `row15_cpython_subinterp_gil`
has a `NOTE.md` and no `before.c`, because the upstream issue turned out to concern interpreter
concurrency and not an SQLite C-API pointer lifetime. It is kept so the row is not silently
dropped from the table.

Cases are not run individually. Each corpus has one runner at its top level, which builds
and executes the whole set:

    sqlite/cve-repros/run-host-asan-repros.sh
    postgres/mmgr-repros/run-host-repros.sh
    cpython/pymalloc-repros/runners/cheribsd/run-defects.py
    cpython/pymalloc-repros/runners/capstone-domain/run-defects.py
    httpd/apr-pool-repros/runners/capstone-domain/run-defects.py
    memcached/allocator-repros/runners/capstone-domain/run-defects.py

The postgres and pymalloc runners execute a control before any case — ASan must see a plain
malloc use-after-free; CheriBSD must pass its ABI and bounds probes — and exit 75 with NO
verdict if it fails, so an infrastructure failure can never be read as a measurement. The
sqlite runner has no such control and does not use that convention.

Silicon defects are not here at all: those live in `tests/fpga-repros/`, one folder per
issue.
