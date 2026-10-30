# 2026-10-30-cheribsd

The `cheribsd-revocation` arm over all 70 cases: the real interpreter on **stock CheriBSD
purecap**, no nested allocator layer anywhere, so the only mechanism is the platform's -- `malloc`
bounds each object, `free` quarantines it, a later sweep revokes. Three boots of one binary: the
platform default (`runtime_revocation_default=1`, `every_free=0`), the same binary with revocation
off as the pair that attributes a fault, and a third at every-free that separates a quarantined
chunk from a genuine miss. Only the default decides a verdict.

- pre-registration (the lane commit this run answers): `23eeacd9807a137ffc4a168ea9656eb123cfe984`
- PHP binary (`ports/php/cheribsd/build.sh`, 5.0.0, ext/standard only) sha256 `b389e2399540f5e5449bff7c58c8112f73cba65be7639db941791dd1ad86f442`
- controls: `bounds-malloc` sha256 `9794786d5555ba70919c6d1bf68eb680c21b6de06032818b47f6f6d7a44706d5`; probes `{"putenv": "a6097b4fc984d7c41250f832e0182009d844883f186aa8be6fb4b7829b69c31d"}`
- platform, read from the tools by `run-cheribsd-cases.py` rather than described: QEMU `7.1.0` (the SDK's `qemu-system-riscv64xcheri`, no `-cpu` override), CHERI-LLVM `clang 17.0.0` (`CTSRD-CHERI/llvm-project` `7e122876ee01a7a82585f7674c94f12fc1cc7689`), firmware `bbl-riscv64cheri-virt-fw_jump.bin` (37520 bytes), the purecap kernel and a qcow2 overlay over the 9.5 GB purecap image
- **this is not the vehicle of `capstone/tests/cheri-php-blind`**, which runs a 7.2.0 CHERI-QEMU fork with `-cpu rv64,sstc=false`; that is why each bundle records its own, and why the two tables may be set side by side only on what they share
- revocation, with every sysctl read back: default `1`/`0`, off `0`/`0`, eager `1`/`1`
- judge: `bug-corpora/tools/verdicts.py` against `tools/arms.json` `cheribsd-default`, driven by `bug-corpora/php/tools/run-php-corpus.py`
- one bundle per scoring: `bundle-cheribsd-revocation/` is the default scoring, `sensitivity/bundle-cheribsd-revocation/` the same observations with the pristine gate lifted, written from the SAME boots
- every cell below is computed by `tools/result-tables.py` from these bundles; nothing is typed by hand

**The controls held.** `bounds-malloc` faulted in every boot, so bounds are enforced on this vehicle
and a MISS means something. The eager boot printed the request-shutdown marker for 31 of the 70
cases, so a fault's placement is a reading rather than the only answer an uninstrumented boot can
give. The `putenv` platform probe ran in both the default and the off boot with identical readings
(control retained, both empty-name calls refused with `EINVAL` and nothing stored).

**Provenance of the two generations.** The default and off boots here carry the `putenv` probe; the
eager record is reused from the first run of the same image earlier the same day, because it decides
no verdict by construction and costs 139 minutes. Against that first run, which had no
platform-precondition support at all, **69 of the 70 cells are identical in both scorings** and the
single change is CRASH-084, from `MISSED` to `out-of-denominator` -- which is the defect this run
exists to fix. Two independent boot generations agreeing on 69 cells is the reproducibility reading.

**CRASH-084 and the libc.** The case declares a platform precondition: its use-after-free needs
`putenv` to keep an empty-name string referenced from `environ`, which is where the freed string
dangles from. FreeBSD's `putenv` refuses an empty name with `EINVAL` and stores nothing, exactly as
musl's does, so the behaviour the defect needs does not exist on this platform either and what the
run did is not a reading of the protection. The first run of this arm scored it `MISSED` having
never checked -- a trigger that ran, completed and caught nothing, read as a miss. Both instruments
now put the same 28 cases out of the default denominator.


### Default scoring (native crashers out of the denominator)

Arms in this record: CheriBSD revocation. Not measured here: stock, system allocator, nested, nested exact.

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| CheriBSD revocation | 6 | not classified | 35 | inconclusive 1, out-of-denominator 28 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| CheriBSD revocation | 1 | not classified | 26 | inconclusive 1, out-of-denominator 27 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| CheriBSD revocation | 1 | not classified | 25 | inconclusive 1, out-of-denominator 24 |

("not classified": no catch in this record carries the `chain` attribution class. Either the record was judged before the class existed (2026-10-28), in which case its chain catches are carried as `function` and are not re-labelled, or its arms do not attribute by call path at all. The bundles do not say which, so neither is claimed here)

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- CheriBSD revocation: 0 of 19 / 1 of 32

### Sensitivity scoring (every case scored by fault site)

Arms in this record: CheriBSD revocation. Not measured here: stock, system allocator, nested, nested exact.

**all 70 cases** (70):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| CheriBSD revocation | 14 | not classified | 41 | inconclusive 13, infra 1, out-of-denominator 1 |

**the temporal cases (CWE-416, CWE-415, CWE-590)** (55):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| CheriBSD revocation | 8 | not classified | 32 | inconclusive 13, infra 1, out-of-denominator 1 |

**the CWE-416 cases** (51):

| arm | CAUGHT | of which by chain | MISSED | NO-READING |
|---|--:|--:|--:|---|
| CheriBSD revocation | 8 | not classified | 31 | inconclusive 11, out-of-denominator 1 |

("not classified": no catch in this record carries the `chain` attribution class. Either the record was judged before the class existed (2026-10-28), in which case its chain catches are carried as `function` and are not re-labelled, or its arms do not attribute by call path at all. The bundles do not say which, so neither is claimed here)

**CWE-416 catches by native population** (temporal-asan-visible / asan-blind):

- CheriBSD revocation: 6 of 19 / 2 of 32

