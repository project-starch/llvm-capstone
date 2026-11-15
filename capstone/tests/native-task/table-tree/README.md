# table-tree: the Phase 1 paged lifetime table (CAPSTONE_VM_TABLE_TREE), its twin gate

`make CV_FLAT_INDEX_BITS=25 CV_IDS=55 CV_TABLE_TREE=1` builds capstone_vm with the flat ids55 table's semantics and its
records in a paged tree that grows (Sublet Section 5, Phase 1; plan `docs/plans/2026-11-15-sublet-section5-rtl.md`).
A kit tells the build by `/sys/module/capstone_vm/parameters/table_tree` = 1. Layout and growth protocol: the comment
at `CAPSTONE_VM_TABLE_TREE` in `capstone_vm.c`; agreed with the rtl lane on 2026-11-15.

## The gate (QEMU twin, native route, node_flat_records=262144, node_batch_pages=16)

Twin kernels: perf/natio/linux-twin-a0ed with this capstone_vm.c and Makefile, built as tree, tree + CV_TEST_HOOKS,
and flat. Every arm carries the lifeops plugin (executed CDERIVE/CREVOKE/CSRETIRE/CSMINT by funct7); k5 is the
context's kind-5 (cause 30) escapes. mcf = the FRCDKMSa3 mcf test stage; ntp = tests/native-task/node-table-pool's
program on the board's dom_ flags.

| arm | build, parameters | predicted | read |
|---|---|---|---|
| tg-F | flat, mcf | 2/2, k5 0 | 2/2, k5 0, CDERIVE 60,282, CREVOKE 60,238 |
| tg-T | tree, mcf | 2/2, outputs = tg-F, 2 growths, k5 2, CDERIVE = tg-F + k5 | 2/2, result files identical to tg-F, table_growths 2, batches 3, k5 2, CDERIVE 60,284 (+2) |
| tg-M1 | hooks, test_tree_mutant=1 (a capacity over an unpublished page) | context refused by the self-check | tree_check_failures 1, the must-fire line, refused |
| tg-M1s | as tg-M1, test_tree_skip_check=1 | fails in the twin, k5 >= 1 | fails (cause 25 in user code) with k5 0: before any cause 30, not where predicted |
| tg-M2 | hooks, test_tree_mutant=2 (growth answers without a slot) | the repeat guard ends it: k5 5 | k5 5, grow_loops 1, the must-fire line, ended |
| tg-FAIL | hooks, test_grow_fail_at=2 (the first growth fails) | ends at its one cause 30 | k5 1, table_growths 0, ended |
| tg2-F-D1 | flat, ntp, pool=4 verify=1 | = ntp-D1 | PASS, hits 7, puts 9, zeroed 932640, violations 0 |
| tg2-T-D1 | tree, ntp, pool=4 verify=1 | PASS, no violations | PASS, hits 7, puts 9, violations 0; 1 growth, k5 1, CDERIVE +1 over flat |
| tg2-T-M1 | hooks, pool mutant 1 (stale root header) + verify | violations > 0 | violations 7, PASS |
| tg2-T-M1u | hooks, pool mutant 1, no verify | not PASS | NTP_DONE FAIL |
| tg2-T-M2 | hooks, pool mutant 2 (stale last record) + verify | violations > 0 | violations 7, PASS |

Re-execution is read two ways: the derive count over the flat arm equals the cause-30 count exactly, and the outputs
are identical. A resume that skipped the CDERIVE would leave rd stale.

The pre-built (no growth) version of the same build, commit 6640bb4fcab1, passed its own 10 arms with one arm
recorded as observed against its prediction.
