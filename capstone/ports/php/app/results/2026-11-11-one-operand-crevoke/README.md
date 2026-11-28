# PHP on the one-operand CREVOKE, 2026-11-11

The port rebuilt from the tree after the nested allocator layer moved to the one-operand
CREVOKE (`heap_end(ptr)` in `../../../zend-alloc/sublet/zend-cap-mode.c`), and run on the QEMU
that implements it. This is the **smoke record** of `../2026-10-27-requalification/` and the
PHP check of `../2026-10-27-common-entry/`, taken again; every cell reads as recorded there.
The corpus measurement of the same date is with the corpus
(`bug-corpora/php/engine-repros/results/2026-11-11-run19/`), the layer's own probes with the
layer (`../../../zend-alloc/sublet/results/one-operand-crevoke-2026-11-11/`).

## What ran

Two boots through `run-virtual.py`, every run its own `capstone-vexec` process. The scripts
are `../2026-10-27-requalification/workload/` and `workload/gate.php`, which is the PHP check
of the platform gate (`runtime/virtual/run.py --php`: a sort, a file written and read back
through the delegated syscalls, one line printed) as a file, since `run-virtual.py` splits a
run's arguments at spaces and cannot carry `-r '...'`.

`smoke-nested.json`: the measurement image (`realphp/php-real-nested.dom`, no default arm),
`PHP_CAP_MODE` named per run.

| run | modes 0 / 4 / 6 / 7 |
|---|---|
| `php -n -v` (mode 0 only) | exit 1, `PHP 5.0.0 (cli) (built: Nov 14 2023 22:13:20)` |
| `hello.php` | exit 0, `3,3,1,2 OK ca72431bf6bc7729c82dbb19ed7a9f2c` on all four |
| `churn.php` | exit 0, `churn 197900` (the exact sum) on all four |
| `sanity.php` (`run-realphp.SANITY_TEXT`) | exit 0, `sanity 573 200 17 abc 0,1,2,3,4,5,6,7,8,9` on all four |
| `sorts.php` | exit 0, `sorted 1,2,3` on all four |
| `gate.php` | exit 0, `PHP_VIRTUAL_OK sorted=1,2,3 md5=ca72431bf6bc7729c82dbb19ed7a9f2c file=virtual` on all four |

The layer's exit line shows the release doing its work: at modes 6 and 7 `retires` equals
`issues` and `live=0` (`churn.php`: `issues=349797 retires=349797 revokes=349797`); at modes
0 and 4 `retires=0 revokes=0`.

`smoke-entry.json`: the two images of the common entry
(`ports/common/application/build-virtual.sh php`, `CAPSTONE_VIRTUAL_LIFETIMES` 1 and 0), each
in a fresh output with the SDK the entry built for it.

| run | `php1` (nested, arm 6 baked in) | `php0` (stock `zend_alloc.c`) |
|---|---|---|
| `php -n -v` | exit 1, the version line, after `ZEND_CAP_MODE_DEFAULTED 6` | exit 1, the version line |
| `hello.php`, `churn.php`, `sanity.php`, `sorts.php` | exit 0, the lines of the table above, each after `ZEND_CAP_MODE_DEFAULTED 6` and `ZEND_CAP_MODE 6 exact=0 revoke=1` | exit 0, the same lines |
| `gate.php` | exit 0, the `PHP_VIRTUAL_OK` line | exit 0, the same line |
| `hello.php`, `sanity.php`, `gate.php` with `PHP_CAP_MODE=7` | exit 0, `ZEND_CAP_MODE 7 exact=1 revoke=1` and no `DEFAULTED` line | |

So the baked-in arm, its override and the stock image behave as `../2026-10-27-common-entry/`
records (`nomode.json` there).

## What did not run

The platform gate itself (`run.py --php IMAGE`, 44 checks). Its other 43 checks need the
contract, SQLite and mruby images of the platform lanes and are theirs; only its PHP check ran
here, as `gate.php`.

## Offline suites

`test-run-realphp.py`: all checks pass (no linter installed, which it reports).
`test-run-cases.py`: 23 checks, 0 failed. `bug-corpora/php/tools/run-php-corpus.py
--self-test`: every arm of the judge fired as declared.

## The platform, by content

`hashes.sha256`. Compiler `0bbb1f88...`; measurement SDK with an 8 MiB stack, 16 MiB data,
line tables in the musl objects and the corpus codegen gate (`-fno-builtin`); the entry's own
SDK for `php1` and `php0`; QEMU `2d5a7107...` with `run-virtual.py`'s `-cpu` string
(`x-capstone-exact-bounds`, `x-capstone-test-csrs`, 32 supervisor slots,
`x-capstone-addr=sv39`), module `e47bca62...` loaded pinned with `node_initial_pages=1024
supervisor_slots=32`; guest images `48eb725a...` / `5b5c047a...` / `62049800...`. PHP source:
the museum tarball by SHA-256, `prepare-sources.sh`, tree patches 0001 and 0002. The nested
image holds 7 CDERIVE and 4 CREVOKE instructions (`count-cap-insns.py`: `derive=7 revoke=4`),
every CREVOKE with rd and rs2 x0.

The first boot ran alone; the second ran beside two other boots of the same work.
