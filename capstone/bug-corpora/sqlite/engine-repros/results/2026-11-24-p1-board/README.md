# The sqlite engine-repros on a P1 board (2026-11-23/24)

Item 5's board reading (pre-registration "item 5 on a P1 board" and its amendments, `docs/ref/nested-allocator-workloads.md`).
Board 2, bitstream `caplifive_ids55-…-pagedtbl_d83d3a633` (sha256 `dc411e36…`, flashed and read back by the chain),
kit `bkit111-p1v-corpus-xzv`, one xz-staged boot per arm through `appvm-session.py`: pools 22:05-22:42, malloc
22:42-01:00. Same image bytes as `../2026-11-23-p1v-twin-p1geom/` (the prediction of record).

- The board recorded each arm as `vp2-<arm>` (`boardvm.record_arm`); the bitstream is the one above, not vp2.
- The boots printed `resolve_dirty=Y` and `resolve_pte=Y` (`@@PARAM`), `supervisor_slots=1` in the module's load line, and
  `flat table 1048576 records, 16777216 bytes, from CMA`.
- 63 of the 66 cells equal the twin in verdict, reason, cause, faulting function and offset. Malloc 19, 20 and 22
  differ in cause only (board 4, 4, 13; twin 24), inside the pre-registration's consequence-fault exception for those
  cells.
- Malloc 11 spins: the gate's watcher ended it at its 6,000 s bound (exit 137), read `NO-READING:infra` as on the twin.

Account paths in `inputs.json` are written `<hdd>/`. Raw console output is not committed.
