# 2026-10-28-run10

Lane C's measurement: the seam on the libc's reclaim; against run7 no CAUGHT cell changed (only CRASH-139, MISSED to unattributed, and CRASH-092, watchdog to completed, moved, both explained in the plan document). Platform: the 1 MiB-stack SDK and the PHP image rebuilt from lane C's tree (the seam is a patch to that image), the launcher and module rebuilt from the same tree, and a different QEMU build than run7's (every hash below differs from run7's record; the plan document's lane C section says which pins), no frame chain, no platform probe.

- pre-registration (the lane commit the run answers): `8569fb7194dc5e436a089788ec3133fc1c851e64`
- PHP image (php-real-seam.dom) sha256 `a1aa15dae7ca5eb13056abeb2f9895d478859b3ce5ed26af0fed49c6a3acfc61`
- QEMU `e8a89c014e5aa627b294cdf0b7d366610cf4334adc1f9d1579c498f33e4fee23`, launcher `3d2eca00cc186992371ed906e22ae19f7ca55d824eed7e930e3b7f704fa3c65f`, module `15ad4c20e0f0a5d5f79f30258ef2e0db403559b3737be3d2405683cf0c458cee`, kernel `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b`
- judge: declared function (case.json fault_sites) in its file, with a protective cause (arms.json protective_causes); own fault licenses own reach; 2026-10-27; protective causes [25, 28]
- one bundle per arm (`bundle-<arm>/verdicts.jsonl`, `inputs.json`), the two populations merged by `tools/merge-population-bundles.py`; `sensitivity/` holds the same records re-scored with `--score-oracle-crashes`
- every cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

### Default scoring (native crashers out of the denominator)

**all 65 cases** (65):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 6 | 32 | infra 1, not-reached 2, out-of-denominator 24 |
| system allocator | 15 | 22 | infra 1, not-reached 2, out-of-denominator 24, unattributed 1 |
| nested | 28 | 10 | infra 2, out-of-denominator 24, unattributed 1 |
| nested exact | 32 | 4 | infra 2, out-of-denominator 24, unattributed 3 |

**the CWE-416 cases** (49):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 1 | 22 | infra 1, not-reached 2, out-of-denominator 23 |
| system allocator | 10 | 12 | infra 1, not-reached 2, out-of-denominator 23, unattributed 1 |
| nested | 23 | 1 | infra 2, out-of-denominator 23 |
| nested exact | 23 | 1 | infra 2, out-of-denominator 23 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 19 / 0 of 30
- system allocator: 2 of 19 / 8 of 30
- nested: 2 of 19 / 21 of 30
- nested exact: 2 of 19 / 21 of 30

### Sensitivity scoring (every case scored by fault site)

**all 65 cases** (65):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 7 | 32 | infra 1, not-reached 13, setup-fault 8, unattributed 4 |
| system allocator | 27 | 22 | infra 1, not-reached 2, setup-fault 10, unattributed 3 |
| nested | 49 | 10 | infra 2, setup-fault 2, unattributed 2 |
| nested exact | 53 | 4 | infra 2, setup-fault 2, unattributed 4 |

**the CWE-416 cases** (49):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 2 | 22 | infra 1, not-reached 13, setup-fault 8, unattributed 3 |
| system allocator | 22 | 12 | infra 1, not-reached 2, setup-fault 10, unattributed 2 |
| nested | 44 | 1 | infra 2, setup-fault 2 |
| nested exact | 44 | 1 | infra 2, setup-fault 2 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 19 / 0 of 30
- system allocator: 10 of 19 / 12 of 30
- nested: 19 of 19 / 25 of 30
- nested exact: 19 of 19 / 25 of 30

