# row15_cpython_subinterp_gil

Upstream: https://github.com/python/cpython/issues/101767
Table 3 row: 15
Class: stale-state
Essence: The upstream issue concerns interpreter concurrency and GIL ownership, not an SQLite C-API pointer lifetime.
Observed: no ASan report - see NOTE.md
