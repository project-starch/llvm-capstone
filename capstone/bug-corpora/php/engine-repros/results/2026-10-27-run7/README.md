# 2026-10-27-run7

Lane B's measurement: the corpus at its own head, run7 (the plan document's lane B sections record runs 5 and 6 of the same tree agreeing with it cell for cell; only run7 is archived here). Platform: the 1 MiB-stack SDK, the launcher without a frame chain, no platform probe.

- pre-registration (the lane commit the run answers): `672a0deffaa2304fa590bd74a776b7eb7e399018`
- PHP image (php-real-seam.dom) sha256 `c7309c906a68a79da70a3aaed6f97be2ea1b5280087b5fc9a120db184554339f`
- QEMU `28b9ae0fb42adb9f97477ef9d1267dfcb411b7bf6d5753173444da09f894f42c`, launcher `d7d32d82a31e1aad8168aae00c9a66489a0198fd6ca1bb9ec80d386de2e06561`, module `d0d98d815558c959a06c45dee6a4f6736d0602fb59a3df6f156800cffb99379a`, kernel `e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b`
- judge: declared function (case.json fault_sites) in its file, with a protective cause (arms.json protective_causes); own fault licenses own reach; 2026-10-27; protective causes [25, 28]
- one bundle per arm (`bundle-<arm>/verdicts.jsonl`, `inputs.json`), the two populations merged by `tools/merge-population-bundles.py`; no sensitivity re-scoring over both populations exists for this run
- every cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

### Default scoring (native crashers out of the denominator)

**all 65 cases** (65):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 6 | 32 | infra 2, not-reached 1, out-of-denominator 24 |
| system allocator | 15 | 22 | infra 2, not-reached 1, out-of-denominator 24, unattributed 1 |
| nested | 28 | 11 | infra 2, out-of-denominator 24 |
| nested exact | 32 | 5 | infra 2, out-of-denominator 24, unattributed 2 |

**the CWE-416 cases** (49):

| arm | CAUGHT | MISSED | NO-READING |
|---|--:|--:|---|
| stock | 1 | 22 | infra 2, not-reached 1, out-of-denominator 23 |
| system allocator | 10 | 12 | infra 2, not-reached 1, out-of-denominator 23, unattributed 1 |
| nested | 23 | 1 | infra 2, out-of-denominator 23 |
| nested exact | 23 | 1 | infra 2, out-of-denominator 23 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 19 / 0 of 30
- system allocator: 2 of 19 / 8 of 30
- nested: 2 of 19 / 21 of 30
- nested exact: 2 of 19 / 21 of 30

