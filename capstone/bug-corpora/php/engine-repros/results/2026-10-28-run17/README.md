# 2026-10-28-run17

The current record: the merged 70-case corpus on the REDUCED seam (cdfbff0e4e57: the real-PHP image no longer links php_virtual_lower.c, the seam reports its counters on one route). Every verdict, reason and attribution is identical to run16's (560 cells, both scorings), as pre-registered; every arm's sanity run prints the seam's exit line, the arm-ordering gate passes, and no transcript carries the dropped "armed" line (run16's carried it in 292 of 296). Platform: run16's; the seam image and seam probe rebuilt from the reduced seam on the same SDK.

- pre-registration (the lane commit the run answers): `1891bea735e9d6779e06326091d22752706cc542`
- PHP image (php-real-seam.dom) sha256 `1ddce9792539e5ec70c3edb00d4b868c98508b8a7a7c0c969410fdf356f28bd4`
- QEMU `e8a89c014e5aa627b294cdf0b7d366610cf4334adc1f9d1579c498f33e4fee23`, launcher `b533a0689e31cb828caae954a64da770d0a12e9d27ab5ababc520e040ed9dcc2`, module `da3d0e5f9be7fc24528b2cf8f05d0976d6017c6ecdff13648a36f74808f8aa8d`, kernel `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b`; probes {"putenv": "5d65c6d7f446886b47f99789b73291dc9c42e17ca2dcbd4b7e328b3fade658f5"}
- judge: declared function (case.json fault_sites) in its file, with a protective cause (arms.json protective_causes); own fault licenses own reach; 2026-10-27; a catch through a forwarding routine's call path is the chain class, counted apart; 2026-10-28; protective causes [25, 28]
- one bundle per arm (`bundle-<arm>/verdicts.jsonl`, `inputs.json`), the two populations merged by `tools/merge-population-bundles.py`; `sensitivity/` holds the same records re-scored with `--score-oracle-crashes`
- every cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

### Default scoring (native crashers out of the denominator)

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 6 | 1 | 33 | infra 1, not-reached 2, out-of-denominator 28 |
| system allocator | 17 | 1 | 21 | infra 1, not-reached 2, out-of-denominator 28, unattributed 1 |
| nested | 30 | 1 | 9 | infra 2, out-of-denominator 28, unattributed 1 |
| nested exact | 35 | 2 | 3 | infra 2, out-of-denominator 28, unattributed 2 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 24 | infra 1, not-reached 2, out-of-denominator 27 |
| system allocator | 12 | 1 | 12 | infra 1, not-reached 2, out-of-denominator 27, unattributed 1 |
| nested | 25 | 1 | 0 | infra 2, out-of-denominator 27, unattributed 1 |
| nested exact | 25 | 1 | 0 | infra 2, out-of-denominator 27, unattributed 1 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 23 | infra 1, not-reached 2, out-of-denominator 24 |
| system allocator | 12 | 1 | 11 | infra 1, not-reached 2, out-of-denominator 24, unattributed 1 |
| nested | 25 | 1 | 0 | infra 2, out-of-denominator 24 |
| nested exact | 25 | 1 | 0 | infra 2, out-of-denominator 24 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 19 / 0 of 32
- system allocator: 2 of 19 / 10 of 32
- nested: 2 of 19 / 23 of 32
- nested exact: 2 of 19 / 23 of 32

### Sensitivity scoring (every case scored by fault site)

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 7 | 1 | 33 | infra 3, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 6 |
| system allocator | 30 | 1 | 21 | infra 2, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 5 |
| nested | 53 | 2 | 9 | infra 3, out-of-denominator 1, unattributed 4 |
| nested exact | 58 | 3 | 3 | infra 3, out-of-denominator 1, unattributed 5 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 24 | infra 3, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 5 |
| system allocator | 25 | 1 | 12 | infra 2, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 4 |
| nested | 48 | 2 | 0 | infra 3, out-of-denominator 1, unattributed 3 |
| nested exact | 48 | 2 | 0 | infra 3, out-of-denominator 1, unattributed 3 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 23 | infra 2, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 3 |
| system allocator | 25 | 1 | 11 | infra 1, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 2 |
| nested | 48 | 2 | 0 | infra 2, out-of-denominator 1 |
| nested exact | 48 | 2 | 0 | infra 2, out-of-denominator 1 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 19 / 0 of 32
- system allocator: 10 of 19 / 15 of 32
- nested: 19 of 19 / 29 of 32
- nested exact: 19 of 19 / 29 of 32
