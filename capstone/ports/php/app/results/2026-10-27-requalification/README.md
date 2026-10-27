# Re-qualification on dev `0b0b3414dca5`, 2026-10-27

The port, as merged into the framework branch, built from scratch and run on today's dev:
compiler, SDK, adapter and module from the branch tree, PHP from the tree
`app/prepare-sources.sh` prepares. This is a **smoke record**, not a corpus measurement: the
60 corpus scripts and their ASAN/pristine oracles were not available, so neither population
was re-run. What it establishes is that real PHP 5.0.0 runs on the current platform on every
arm, and it is the record behind the correction in `../2026-10-26/README.md`.

## What ran

One guest boot per image, through the port's own `run-virtual.py`, every run its own
`capstone-vexec` process; `PHP_CAP_MODE` selects the seam mode.

| run | stock image | seam image, modes 0 / 4 / 6 / 7 |
|---|---|---|
| `php -n -v` | exit 1, `PHP 5.0.0 (cli) (built: Nov 14 2023 22:13:20)` (`SOURCE_DATE_EPOCH` 1700000000) | same |
| `hello.php` (count, implode, strtoupper, md5) | exit 0, `3,3,1,2 OK ca72431bf6bc7729c82dbb19ed7a9f2c` | exit 0 on all four |
| `churn.php` (2000 rounds of array/string build, `unset`, free) | exit 0, `churn 197900` (the exact sum) | exit 0, exact, on all four |
| the runner's `revocation-sanity` script (`run-realphp.SANITY_TEXT`: ksort, sort, usort, 400 allocations, serialize round trip) | exit 0, `sanity 573 200 17 abc 0,1,2,3,4,5,6,7,8,9` | exit 0, same line, on all four |
| `sorts.php` (`sort(array(3,1,2))`) | exit 0, `sorted 1,2,3` | exit 0, same |

`smoke-stock.json` and `smoke-seam.json` are `run-virtual.py`'s records: per run, the exit
status read from the guest's `VM_EXIT` marker and the console output attributed to it. The
launcher's `CAPSTONE_VM_ADD_OK` mapping lines (one per mapping, hundreds per run) and kernel
console lines are removed from the committed copies, with the count removed recorded under
`filtered`; every other line and every exit status is as recorded. The
judge that read them required the full run set, those exits and those strings (the console
interleaves launcher and kernel lines into application output mid-line, so it matches on the
run's text with those fragments removed). Three earlier runs of the same images, before patch
0002, agreed cell for cell except `sorts.php`, which faulted; see below.

## The probe behind patch 0002

`swap-probe.c` swaps two `malloc`'d pointers the three ways a C program might -- as `int`s,
as `char`s, with `memcpy` -- then reads the tag of each and dereferences. `swap-probe.json`
is its record: the `int` and `char` swaps leave both tags 0 and the next load faults with
cause 24; `memcpy` keeps one tag (its scratch buffer is a `char[64]`, not pointer-aligned).
PHP 5.0.0's `_zend_qsort_swap` is the `int`-then-`char` case on the `Bucket *` array that
`zend_hash_sort` sorts, which is why every `sort`, `usort` and `ksort` faulted before
`app/patches/php-5.0.0-0002-zend-qsort-swap-keeps-the-capability.patch`, and why CRASH-022
and CRASH-078 in the 2026-10-26 result are that fault and not a detection. Compiled with the
same SDK as the images: `capstone-cc -O0 -std=gnu89 swap-probe.c`.

## The platform, by content

| component | identity |
|---|---|
| tree | `ports/php-framework` at `b663561a9a24` (dev `0b0b3414dca5` + the php-5 merge + this port's commits) |
| compiler | clang 22.0.0git built from the tree at `500f854f46a6` (its `llvm/` is identical at `b663561a9a24`), Release, assertions on |
| SDK | virtual profile: `libc-capstone-virtual.a` `c590ca0d…`, `libapplication-runtime.a` `12726a71…`, musl 1.2.5 with the tree's patches |
| images | `php-real-stock.dom` `a6d47e9b…`, `php-real-seam.dom` `ac8532fc…`, `probe-seam.dom` `087ea279…`, `swap.dom` `bbcc81f8…` |
| adapter | `capstone-vexec` `8378b036…`, `capstone_vm.ko` `dff9fafd…` (paged table, built against the `linux-custom` 6.1.0 build the gate lanes use; `MODVERSIONS` off) |
| emulator | capstone-qemu `b290f565` (dev's pin), binary `28b9ae0f…`, `-cpu` as `run-virtual.py` names it: `x-capstone-u-mode`, `x-capstone-exact-bounds`, `x-capstone-test-csrs`, 32 supervisor slots; module `supervisor_slots=32 node_initial_pages=1024` |
| guest | Image `e833dfb1…`, `fw_jump.elf` `4dd4a2ad…`, `rootfs.ext2` `62049800…` (the gate lanes' images-v7 set) |
| PHP source | `prepared-tree-PROVENANCE.txt`: the museum release tarball by SHA-256, `CFLAGS=-std=gnu89 ./configure --disable-all`, tree patches 0001 and 0002 |

Hashes are the first 16 hex digits of SHA-256. The collaborator's 2026-10-26 runs used the
same QEMU source pin and cpu string but their own builds of QEMU and the guest images.

## What this does not say

Nothing about the corpus: no case was run, no verdict was produced, `derive-verdicts.py` has
nothing to re-judge here. CRASH-022 and CRASH-078 are measurable for the first time with patch
0002 and have no reading yet. The real-PHP runner (`run-realphp.py`) was exercised only by its
offline suite (`test-run-realphp.py`, all checks pass) and through `SANITY_TEXT` here.
