# 20261029-allocator-probe

Which defence ends a case on CheriBSD? Measured with `controls/abort-probe.c`, five stages in one
boot at the platform default.

| stage | result |
|---|---|
| malloc/free | completes |
| **double free** | **SIGABRT, nothing printed** -- the signature of the corpus's aborting cases |
| free of an interior pointer | SIGABRT, nothing printed |
| **read after free** | **completes, and reads the old byte back** |
| **write after free** | **completes** |

So the aborts are the allocator's own consistency check, and a plain use-after-free does not trap
at the default setting at all: the chunk is quarantined, not revoked, and the stale capability
stays valid until a sweep runs -- which in a process this short may never happen. The eager
configuration sweeps on every free, and 10 of the 14 cases that abort at the platform
default become capability faults there (11 aborting cases in plain-temporal-repros, of which
7 convert, and 3 in zend-alloc-repros, all of which convert); 4 still abort.

`matrix.tsv` is the per-stage reading. The transcript is not committed.
