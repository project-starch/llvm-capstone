# PROVENANCE — row15 (CPython #101767 subinterpreter / GIL)

- Upstream : https://github.com/python/cpython/issues/101767
- Tier     : **N/A** (out of scope — no SQLite pointer lifetime)

The upstream issue concerns per-interpreter GIL ownership / subinterpreter
concurrency, not an SQLite C-API pointer-lifetime bug. There is no borrowed or
owned SQLite handle whose lifetime a capability could revoke — the failure mode is
interpreter concurrency / liveness. Out of scope for the temporal-safety corpus;
no `before.c` (nothing memory-unsafe in SQLite to reproduce).
