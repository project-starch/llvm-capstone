# The php engine-repros on a P1 board (2026-11-24)

Item 5's board reading (pre-registration "item 5 on a P1 board" and its amendments, `docs/ref/nested-allocator-workloads.md`).
Board 2, bitstream `caplifive_ids55-…-pagedtbl_d83d3a633` (sha256 `dc411e36…`), the runtime lane's board runner
`app-p1b`, one boot per arm. Same image bytes as `../2026-11-23-p1v-twin-p1geom/` (the prediction of record).

- `temporal-asan-visible/` (kit `bkit111-p1v-corpus-xzv`, 02:56-05:10): all four boots completed. Each read
  `/sys/module/capstone_vm/parameters/` as `ids=55`, `index_bits=25`, `node_flat_records=1048576`,
  `supervisor_slots=1`, `resolve_dirty=Y`, `resolve_pte=Y`. The sanity arm completed on every boot (6,375 revokes on
  object and exact) and the bound probe faulted as declared on object and exact. All 124 cells equal the twin in
  verdict, reason, cause and faulting function.
- `asan-blind`: the first attempt read nothing (amendment 7); it is re-run on bkit112.

The bundles' platform fields hash the `--qemu` and `--images` inputs, which the board does not run; the board's
identity (kit, firmware, kernel hashes) is each run's `board` record, which is not committed. Account paths are written `<hdd>/`. Raw console output is not committed.
