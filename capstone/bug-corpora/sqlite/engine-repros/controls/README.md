# Controls for the virtual arms

Not cases. Each is a program in the corpus's own harness (`repro322_common.h`, `REPRO322_MAIN`),
built by `ports/sqlite/repro322/build-virtual.py --observe` against the SAME engine object as the
cases, and run before them. What each must do is the configuration's, in `tools/arms.json`:

| control | virtual-sqlite-memsys5 (stock) | virtual-sqlite-memsys5-pools (Sublet port) |
|---|---|---|
| `control_uaf_mem5` -- one sqlite3_malloc block, freed, read through its alias | complete | fault, in `control_read` |
| `control_bounds_mem5` -- a 16-byte block, written 1 KiB past its start | complete | fault, in `control_write` |
