# 2026-11-11-run19

The current record: the 70 cases on the nested allocator layer after it moved to the one-operand CREVOKE (`heap_end(ptr)` in `ports/php/zend-alloc/sublet/zend-cap-mode.c`, where it called `heap_end(block, ptr)`), on the QEMU that implements that instruction. Against run18, in the default scoring, 274 of the 280 cells keep their verdict and reason; six move, on two cases, and both read the same on unmodified `dev` with `dev`'s QEMU (`control-dev/matrix.tsv`), so neither is the instruction change's. The sensitivity scoring moves the same cells and four more, on one case, which also reads the same on `dev`. The image holds 7 CDERIVE and 4 CREVOKE instructions, every CREVOKE of the one-operand form. The cross-arm gates pass; `revocation-sanity` completes on every arm and `nested-bound-probe` faults on the two that declare it.

- pre-registration: none was written for this run; it is a re-measurement of run18's configurations after a conversion, compared cell by cell
- PHP image (php-real-nested.dom) sha256 `a8a77eadd554cc8577e139654d634270d21052294bd0d75e9cecf05789a87949`
- QEMU `2d5a7107890e48475fada73439d90ffd5c7cff166eb1d547d7a9fc5f31722365`, launcher `0f8e0f903725d734817b3e30d79127c290f3480a641d4f6de8bc7f699d05c76e`, module `e47bca62addfc9356f8150755f2500b737912eb01088e91398e29a1ed7a4e180`, kernel `48eb725a40e448fa9ac85abb8b7f9fe8a4080d73ffd49c051bed111319d64da2`; probes {"putenv": "8eb1331676abd9c8e1a3d4a2da7ef64f23838301a8d8c138cec641f8f8864e3f"}; `-cpu` and module arguments as run18's (`run-virtual.py`), a pinned load
- SDK: 8 MiB stack, 16 MiB data, line tables in the musl objects, the corpus codegen gate (`-fno-builtin`)
- judge: declared function (case.json fault_sites) in its file, with a protective cause (arms.json protective_causes); own fault licenses own reach; a catch through a forwarding routine's call path is the chain class, counted apart; protective causes [25, 28]
- one bundle per arm (`bundle-<arm>/verdicts.jsonl`, `inputs.json`), the two populations merged by `tools/merge-population-bundles.py`; `sensitivity/` holds the same records re-scored with `--score-oracle-crashes`
- every table cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

**How the boots were taken, where it differs from run18.** The host's QEMU lock was shared with long sessions of other work, so the arms were not booted one per session. The stock arm of the asan-blind population is `run-realphp.py`'s own boot. The other three arms of that population ran in one boot, and the four arms of the temporal-asan-visible population in another: per arm exactly the runs `run-realphp.run_arm` builds (the sanity control, the putenv probe, the bound probe where declared, every case), each run its own `capstone-vexec` process with the arm's `PHP_CAP_MODE` in its own environment. The record of each boot was split by arm and judged by `run-realphp.py --from-record`. Those two boots ran side by side with one other boot of this work. The watchdog readings are run18's (CRASH-075 and CRASH-161 on four arms, CRASH-081 on stock; CRASH-092, whose stock timeout is from the boot that ran alone, is the case discussed here), so the load added no timeout. The symbolizer's source paths are recorded relative to the run's work directory. The populations hold 39 and 31 cases where run18's held 48 and 22: nine `oracle.json` files changed population after run18 (the release-destination correction); the 70 cases are the same.

## The cells that differ from run18

| case | arm | run18 | run19 | on unmodified `dev` |
|---|---|---|---|---|
| CRASH-092 (30) | stock, system allocator | NO-READING not-reached | NO-READING infra (the 40 s watchdog) | infra |
| CRASH-092 (30) | nested, nested exact | NO-READING infra (the 40 s watchdog) | **CAUGHT**, cause 25 in `_zval_ptr_dtor` at Zend/zend_execute_API.c:389, the declared function at ASan's own line | CAUGHT, the same site |
| CRASH-139 (41) | nested, nested exact | NO-READING unattributed (the layer refused the pointer by name, `code=3`) | **MISSED** (completes, as on stock) | MISSED |
| CRASH-153 (44), sensitivity only | all four | NO-READING unattributed (cause 24 in `_phpi_pop`) | NO-READING not-reached (completes) | not-reached |

`control-dev/matrix.tsv` is the `dev` column: the same image recipe built from unmodified `dev` (`9e4d529f...`, its own SDK and musl tree), booted on `dev`'s QEMU (`35529815...`), those three cases on the four arms, judged by the same runner. Its cells equal run19's. So the three cases changed between run18's platform and today's `dev`, before the instruction change. What changed them is not established here. CRASH-092's README note already says it "hangs, or completes short of its defect, on uninitialised stack contents"; CRASH-139's trigger frees a static string through a zval the engine builds on the C stack without initialising its reference count (its `trigger.php`), so whether the free happens depends on the same kind of stack contents. CRASH-139 is one of the three CWE-590 cases, and its MISSED is the one miss in the temporal table of the nested arms.

### Default scoring (native crashers out of the denominator)

Arms in this record: stock, system allocator, nested, nested exact. Not measured here: CheriBSD revocation.

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 6 | 1 | 33 | infra 2, not-reached 1, out-of-denominator 28 |
| system allocator | 17 | 1 | 21 | infra 2, not-reached 1, out-of-denominator 28, unattributed 1 |
| nested | 31 | 1 | 10 | infra 1, out-of-denominator 28 |
| nested exact | 36 | 2 | 4 | infra 1, out-of-denominator 28, unattributed 1 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 24 | infra 2, not-reached 1, out-of-denominator 27 |
| system allocator | 12 | 1 | 12 | infra 2, not-reached 1, out-of-denominator 27, unattributed 1 |
| nested | 26 | 1 | 1 | infra 1, out-of-denominator 27 |
| nested exact | 26 | 1 | 1 | infra 1, out-of-denominator 27 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 1 | 1 | 23 | infra 2, not-reached 1, out-of-denominator 24 |
| system allocator | 12 | 1 | 11 | infra 2, not-reached 1, out-of-denominator 24, unattributed 1 |
| nested | 26 | 1 | 0 | infra 1, out-of-denominator 24 |
| nested exact | 26 | 1 | 0 | infra 1, out-of-denominator 24 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 1 of 28 / 0 of 23
- system allocator: 5 of 28 / 7 of 23
- nested: 5 of 28 / 21 of 23
- nested exact: 5 of 28 / 21 of 23

### Sensitivity scoring (every case scored by fault site)

Arms in this record: stock, system allocator, nested, nested exact. Not measured here: CheriBSD revocation.

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 7 | 1 | 33 | infra 4, not-reached 13, out-of-denominator 1, setup-fault 7, unattributed 5 |
| system allocator | 30 | 1 | 21 | infra 3, not-reached 2, out-of-denominator 1, setup-fault 9, unattributed 4 |
| nested | 54 | 2 | 10 | infra 2, not-reached 1, out-of-denominator 1, unattributed 2 |
| nested exact | 59 | 3 | 4 | infra 2, not-reached 1, out-of-denominator 1, unattributed 3 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 24 | infra 4, not-reached 12, out-of-denominator 1, setup-fault 7, unattributed 5 |
| system allocator | 25 | 1 | 12 | infra 3, not-reached 1, out-of-denominator 1, setup-fault 9, unattributed 4 |
| nested | 49 | 2 | 1 | infra 2, out-of-denominator 1, unattributed 2 |
| nested exact | 49 | 2 | 1 | infra 2, out-of-denominator 1, unattributed 2 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| stock | 2 | 1 | 23 | infra 3, not-reached 12, out-of-denominator 1, setup-fault 7, unattributed 3 |
| system allocator | 25 | 1 | 11 | infra 2, not-reached 1, out-of-denominator 1, setup-fault 9, unattributed 2 |
| nested | 49 | 2 | 0 | infra 1, out-of-denominator 1 |
| nested exact | 49 | 2 | 0 | infra 1, out-of-denominator 1 |

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- stock: 2 of 28 / 0 of 23
- system allocator: 18 of 28 / 7 of 23
- nested: 27 of 28 / 22 of 23
- nested exact: 27 of 28 / 22 of 23
