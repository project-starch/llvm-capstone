# 2026-10-28-run15

The follow-up lane's measurement and the current record. These bundles are run15's transcripts re-judged under the third review's tree (bbfa6e77dfbf: the faulting frame classified like a chain frame, the launcher's exit line out of the application's output) and again under the fourth's (f9079b6f1e8e: a catch through the call path carries the attribution class `chain`, shown apart below); every verdict and reason is the one the original judge gave, pre-registered and checked cell for cell. Cell-identical to run14 (both scorings, both populations, 520 cells), on an adapter rebuilt from the tree whose launcher names its image when a run ends, so the putenv probe's exclusion of CRASH-084 rests on a transcript that names the recorded image. Platform: the 8 MiB-stack SDK with line tables for the musl objects, the launcher that records the frame chain and prints its image on exit, the putenv probe. run14 (`../2026-10-28-run14/`) is kept as the record it was; run13 could not be merged (its two boots recorded different probe lists).

- pre-registration (the lane commit the run answers): `009ca7b34b006c0cb51e8d16e6b15deb201bc45b`
- PHP image (php-real-seam.dom) sha256 `afd3e23961ee69522e6f03007e5db00584940dd29aa7cb2a8c39fd29bcda75ef`
- QEMU `e8a89c014e5aa627b294cdf0b7d366610cf4334adc1f9d1579c498f33e4fee23`, launcher `b533a0689e31cb828caae954a64da770d0a12e9d27ab5ababc520e040ed9dcc2`, module `da3d0e5f9be7fc24528b2cf8f05d0976d6017c6ecdff13648a36f74808f8aa8d`, kernel `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b`; probes {"putenv": "5d65c6d7f446886b47f99789b73291dc9c42e17ca2dcbd4b7e328b3fade658f5"}
- judge: declared function (case.json fault_sites) in its file, with a protective cause (arms.json protective_causes); own fault licenses own reach; 2026-10-27; protective causes [25, 28]
- one bundle per arm (`bundle-<arm>/verdicts.jsonl`, `inputs.json`), the two populations merged by `tools/merge-population-bundles.py`; `sensitivity/` holds the same records re-scored with `--score-oracle-crashes`
- every cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

### Default scoring (native crashers out of the denominator)

**all 65 cases** (65):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 6 | 1 | 31 | infra 1, not-reached 2, out-of-denominator 25 |
| system allocator | 15 | 1 | 21 | infra 1, not-reached 2, out-of-denominator 25, unattributed 1 |
| nested | 28 | 1 | 9 | infra 2, out-of-denominator 25, unattributed 1 |
| nested exact | 33 | 2 | 3 | infra 2, out-of-denominator 25, unattributed 2 |

**the CWE-416 cases** (49):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 21 | infra 1, not-reached 2, out-of-denominator 24 |
| system allocator | 10 | 1 | 11 | infra 1, not-reached 2, out-of-denominator 24, unattributed 1 |
| nested | 23 | 1 | 0 | infra 2, out-of-denominator 24 |
| nested exact | 23 | 1 | 0 | infra 2, out-of-denominator 24 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 19 / 0 of 30
- system allocator: 2 of 19 / 8 of 30
- nested: 2 of 19 / 21 of 30
- nested exact: 2 of 19 / 21 of 30

### Sensitivity scoring (every case scored by fault site)

**all 65 cases** (65):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 7 | 1 | 31 | infra 2, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 4 |
| system allocator | 28 | 1 | 21 | infra 1, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 3 |
| nested | 51 | 2 | 9 | infra 2, out-of-denominator 1, unattributed 2 |
| nested exact | 56 | 3 | 3 | infra 2, out-of-denominator 1, unattributed 3 |

**the CWE-416 cases** (49):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 21 | infra 2, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 3 |
| system allocator | 23 | 1 | 11 | infra 1, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 2 |
| nested | 46 | 2 | 0 | infra 2, out-of-denominator 1 |
| nested exact | 46 | 2 | 0 | infra 2, out-of-denominator 1 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 19 / 0 of 30
- system allocator: 10 of 19 / 13 of 30
- nested: 19 of 19 / 27 of 30
- nested exact: 19 of 19 / 27 of 30
