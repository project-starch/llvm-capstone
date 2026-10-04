# fz08 -- fz08

Upstream fix `FZ08`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
FZ08 — SEGV in sqlite3VdbeCursorMoveto on a deeply nested CTE/sub-select.
 *
 * Collected by replaying SQLite's post-3.22.0 fuzz corpus against 3.22.0. Host ASan on
 * stock 3.22.0, SEGV on address 0x3 (a small non-zero address, i.e. a bad pointer rather
 * than a plain NULL):
 *
 *   #0 sqlite3VdbeCursorMoveto  sqlite3.c:76109    VdbeCursor *p = *pp;
 *   #1 sqlite3VdbeExec          sqlite3.c:82409
 *
 * The trigger is a single self-contained statement: a CTE named c whose body is a
 * VALUES/UNION, with min(-i) aggregates over further CTEs of the same name nested inside
 * sub-selects. The repeated shadowing of the name `c` at several nesting depths is what
 * drives the cursor bookkeeping wrong, so the nesting is NOT decoration and must be kept.
 *
 * Kept verbatim from the fuzz case (the inner text is machine-generated and resists
 * hand-minimisation; reducing the nesting makes it run clean).
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
