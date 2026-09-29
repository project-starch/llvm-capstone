# row1_cpython_progress_uaf

Upstream: https://github.com/python/cpython/issues/142830
Table 3 row: 1
Class: UAF
Essence: A progress callback frees its heap context, then host-wrapper code reads that context after SQLite returns from the callback.
Observed: ERROR: AddressSanitizer: heap-use-after-free
