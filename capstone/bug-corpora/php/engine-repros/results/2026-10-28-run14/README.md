# 2026-10-28-run14

The follow-up lane's measurement, replacing run12 and run13 (run14 is cell-identical to run13; run13's two boots could not be merged because their platform records differed in the probe list). Platform: the 8 MiB-stack SDK with line tables for the musl objects, the launcher that records the frame chain, the putenv probe.

- pre-registration (the lane commit the run answers): `1ed7d8c601e247048b7314af4bdbf0a64dd49f20`
- PHP image (php-real-seam.dom) sha256 `afd3e23961ee69522e6f03007e5db00584940dd29aa7cb2a8c39fd29bcda75ef`
- QEMU `e8a89c014e5aa627b294cdf0b7d366610cf4334adc1f9d1579c498f33e4fee23`, launcher `6d4aa29f23901f449262c4cdfaa428956ef6799cb660e37f050cf7e324d68da6`, module `b378711c6ae8d3ae748219bb2629fb3bbe8074686ae4febef57d7595672dd966`, kernel `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b`
- judge: declared function (case.json fault_sites) in its file, with a protective cause (arms.json protective_causes); own fault licenses own reach; 2026-10-27; protective causes [25, 28]
- one bundle per arm (`bundle-<arm>/verdicts.jsonl`, `inputs.json`), the two populations merged by `tools/merge-population-bundles.py`; `sensitivity/` holds the same records re-scored with `--score-oracle-crashes`
- every cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

### Default scoring (native crashers out of the denominator)

**all 65 cases** (65):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 6 | 0 | 31 | infra 1, not-reached 2, out-of-denominator 25 |
| system allocator | 15 | 0 | 21 | infra 1, not-reached 2, out-of-denominator 25, unattributed 1 |
| nested | 28 | 0 | 9 | infra 2, out-of-denominator 25, unattributed 1 |
| nested exact | 33 | 0 | 3 | infra 2, out-of-denominator 25, unattributed 2 |

**the CWE-416 cases** (49):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 0 | 21 | infra 1, not-reached 2, out-of-denominator 24 |
| system allocator | 10 | 0 | 11 | infra 1, not-reached 2, out-of-denominator 24, unattributed 1 |
| nested | 23 | 0 | 0 | infra 2, out-of-denominator 24 |
| nested exact | 23 | 0 | 0 | infra 2, out-of-denominator 24 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 19 / 0 of 30
- system allocator: 2 of 19 / 8 of 30
- nested: 2 of 19 / 21 of 30
- nested exact: 2 of 19 / 21 of 30

### Sensitivity scoring (every case scored by fault site)

**all 65 cases** (65):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 7 | 0 | 31 | infra 2, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 4 |
| system allocator | 28 | 0 | 21 | infra 1, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 3 |
| nested | 51 | 0 | 9 | infra 2, out-of-denominator 1, unattributed 2 |
| nested exact | 56 | 0 | 3 | infra 2, out-of-denominator 1, unattributed 3 |

**the CWE-416 cases** (49):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 0 | 21 | infra 2, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 3 |
| system allocator | 23 | 0 | 11 | infra 1, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 2 |
| nested | 46 | 0 | 0 | infra 2, out-of-denominator 1 |
| nested exact | 46 | 0 | 0 | infra 2, out-of-denominator 1 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 19 / 0 of 30
- system allocator: 10 of 19 / 13 of 30
- nested: 19 of 19 / 27 of 30
- nested exact: 19 of 19 / 27 of 30

