# native-detect (ASan) on upstream's own backing -- 2026-10-10

    bash capstone/bug-corpora/memcached/allocator-repros/runners/run-asan.sh <fresh outdir>

The port is now built with `MCP_STOCK_MALLOC=ON` (`ports/memcached/allocators/cmake/Allocators.cmake`):
each slab page and each cache.c object is its own host `malloc`, as in upstream memcached, using the
ledger stock CheriBSD already runs (`src/cheribsd/malloc-leases.c`). The earlier arm
(`../2026-10-09-native-asan/`) carved pages and objects out of one 64 MiB `aligned_alloc`, which an
upstream build does not do.

**Result: 8 SILENT, case 8 REPORTED** (`heap-buffer-overflow` in the scan, frame `mc_case_body`, 0
bytes after the 16384-byte rbuf object). Both controls reported in the same run (one byte past, and a
read after free of, a 1 MiB block).

**Case 8's reduction had a defect this exposed.** Its labelled probe read `rcurr + rbytes` -- one byte
past the object -- on BOTH arms, so the FIXED arm made an out-of-bounds read of its own. On the arena
the byte belonged to the next object and nothing noticed; on a per-object malloc the fixed arm reported
`heap-buffer-overflow`. The fixed arm now probes where the bounded scan stopped (`case.c`); the buggy
arm's probe is unchanged, and the other platforms' runners execute the buggy sequence only.
