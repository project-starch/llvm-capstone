# fz02 -- fz02

Upstream fix `FZ02`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
fz02 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86):
 *     sqlite3Get4byte:29655, heap-buffer-overflow
 *
 * PROVENANCE OF THE TRIGGER. This entry carried only the database and an ASan
 * log; the SQL was never recorded, because fuzzcheck pairs every database with
 * every script in the corpus and the provenance kept only the database id. The
 * statement below was re-derived by replaying candidate statements one process
 * at a time against a FRESH copy of the image -- a cumulative script does not
 * work here, since the earlier statements rewrite the very corruption the later
 * ones need. It reproduces the recorded crash at the SAME source line.
 *
 * The image is embedded and served by repro322_memfs.c, an in-domain in-memory
 * FILESYSTEM: the pager needs a rollback journal for any write, and memvfs.c
 * serves the main database only and takes its buffer as an integer in the URI,
 * which on a capability machine is a pointer forged from a scalar.
 *
 * REACHABILITY: every statement's rc is printed. SQLITE_CORRUPT (11) on the
 * statement that should reach the defect means SQLite rejected the image first
 * -- a real negative, not a silent pass.
```
