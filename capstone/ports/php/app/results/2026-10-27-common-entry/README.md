# PHP through the common entry and on the platform gate, 2026-10-27

The record behind "like the other applications": PHP built by
`ports/common/application/build-virtual.sh php` for both selector values, and the platform
gate `runtime/virtual/run.py` run with `--php` on each image. Same host, platform and tree as
`../2026-10-27-requalification/` (dev `0b0b3414dca5` + this branch, at `2443e310fbaa`), the
same adapter, module, emulator and guest images; `hashes.sha256` holds the full SHA-256 of
every file named here.

## Built through the entry

| `CAPSTONE_VIRTUAL_LIFETIMES` | image | what it is | `pipeline.json` |
|---|---|---|---|
| 0 | `php0/image/php.dom` `34fd5588…` | `zend_alloc.c` as released over virtual mallocng | `pipeline-php0.json`: `lifetimes 0`, `shared_sdk true` |
| 1 | `php1/image/php.dom` `5f83f4fe…` | the seam (patch 0003), arm baked in as mode 6 | `pipeline-php1.json`: `lifetimes 1`, `shared_sdk true` |

Each in a fresh output with the SDK the entry built for it (`libapplication-runtime.a`
`12726a71…` in both -- the same bytes, built twice), its tree from `prepare-sources.sh` under
`OUT/source` (patch set `31089150…`, the one `../2026-10-27-requalification/` names). Build
times on p13: 1 min 43 s and 1 min 48 s. `sqlite` and `mruby` were built through the same
entry for the gate's battery (`sqlite3.dom` `09e10f35…`, `mruby` `950a5adc…`), and
`contract.dom` `00428f27…` with the `php0` SDK.

## On the platform gate

`run.py --skip-recycling --php IMAGE`, module `node_initial_pages=1024 supervisor_slots=32`,
emulator `-cpu` extras `x-capstone-exact-bounds=true,x-capstone-test-csrs=true,x-capstone-supervisor-slots=32`:

| image | `run.py` | of which PHP |
|---|---|---|
| stock (`php0`) | **PASS, 44 of 44** (`run-php0.json`) | `php` exit 0, `php_output` the expected line |
| seam, arm 6 (`php1`) | **PASS, 44 of 44** (`run-php1.json`) | same |

The PHP check sorts (so patch 0002, PHP's qsort swap keeping its capabilities, is on the
gate), writes and reads a file through the delegated syscalls, and prints
`PHP_VIRTUAL_OK sorted=1,2,3 md5=ca72431bf6bc7729c82dbb19ed7a9f2c file=virtual`. The 42 other
checks are the battery as it was. The first run of this gate read 43 of 44 on both images,
`php_output` false: PHP's `echo` with several arguments writes each one, and the launcher's
`CAPSTONE_VM_ADD_OK` diagnostic (stderr, merged by `2>&1`) landed between `file=` and the
file's contents, where `file_get_contents` maps a page. `2443e310fbaa` builds the line first
and writes it once; the values were right in both runs.

## The baked-in arm

`nomode.json` (`run-virtual.py`, launcher mapping lines filtered as in the sibling bundle):
the `php1` image run with **no `PHP_CAP_MODE`** prints `ZEND_CAP_MODE_DEFAULTED 6` then
`ZEND_CAP_MODE 6 exact=0 revoke=1`, runs `hello.php` and a `sort()` to completion (exit 0,
`sorted 1,2,3`); the same image with `PHP_CAP_MODE=7` prints `ZEND_CAP_MODE 7 exact=1 revoke=1`
and no `DEFAULTED` line. The measurement images (`realphp/php-real-*.dom` of the sibling
bundle) carry no default and still refuse an unnamed arm.

## What this does not say

Nothing about the corpus or any defect: the gate's PHP check is one benign program. It says
PHP builds and runs the way the other applications do, under the same selector, SDK identity
and gate.
