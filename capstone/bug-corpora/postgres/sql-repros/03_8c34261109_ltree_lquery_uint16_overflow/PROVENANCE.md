# ltree_lquery_parse_overflow

Collected in round R4/R5 (PostgreSQL 17.5, NVD and upstream history).
Upstream identifier: `no-fix-recorded`.

The trigger is `trigger.sql`, run statement by statement against a real
server. Host oracle artefacts kept with the collection:

- `host-run-20261004.txt`
- `host.out`

## The threshold, measured on every build (2026-10-10)

The uint16 that wraps is the per-level `totallen` in `ltree_io.c:539-546`:
`LQL_HDRSIZE` once, then `MAXALIGN(LVAR_HDRSIZE + len)` per OR-variant. For a
1000-character variant that is 1008 bytes where `MAXIMUM_ALIGNOF` is 8 and
1024 where it is 16, over a level header of 16, so the first count that wraps
is 65 on the former and 64 on the latter.

Rather than trust that, each build was asked what a variant costs on it, in
band and in one statement -- the difference between `pg_column_size` of a
2-variant and a 3-variant lquery -- and then the count was bisected with one
backend per input:

| build | `MAXIMUM_ALIGNOF` as measured | per-variant | first count that faults | one below |
|---|---:|---:|---:|---:|
| host, `--enable-cassert` + ASan | 8 (2048 -> 3056) | 1008 | 65, heap-buffer-overflow | 64 clean |
| `cheribsd-revocation` purecap | 16 (2080 -> 3104) | 1024 | 64, SIGPROT | 63 clean |
| `spatial` | 16 (pg_config.h) | 1024 | 64, at `pc=0xe02023a8` | 63 clean |
| `sublet` | 16 (pg_config.h) | 1024 | 64, at `pc=0xe0202414` | 63 clean |

`trigger.sql` is 66 variants, so it crosses the boundary on all four. Two
things follow. The Capstone arms fault at 64 at the *same instruction* the
trigger faults at and complete at 63, so that instruction is firing on the
wrap and not on large lqueries in general -- which is what the shared faulting
address of cases 02 and 03 could not settle by itself. And the purecap guest's
installed `pg_config.h` claims `MAXIMUM_ALIGNOF 8` while its deployed ltree
behaves as 16; the measurement is what the arm does.

`CREATE EXTENSION ltree` on its own exits 0 in the purecap guest, so the
extension script is not what faults there. It does fault on `sublet`, which is
why the case runs against a fixture that already carries the extension; that
one is still open and belongs to the Sublet runtime.

This is also the correction of an error in this case's own record. The first
`control.sql` was 64 variants, which wraps wherever the alignment is 16 --
all three arms -- so it faulted, and the `sublet` row was withdrawn on the
reading that the arm's fault did not depend on the defect. It did. The control
is now 48 variants, 49168 bytes, and completes on all three arms.
