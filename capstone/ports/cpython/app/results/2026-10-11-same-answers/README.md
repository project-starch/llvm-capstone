# CPython 3.13.7: protected vs stock vs native, 2026-10-11

The CPython row of the pre-registration
`capstone/docs/plans/2026-10-11-protected-same-answers.md` (pushed as 9bc6f9adf1c9). The
prediction: `python -m test` run sequentially in one process gives the same outcome for each
test on the protected image (`CPY_SUBLET=1`, patch 0014) as on the stock virtual image. Both
arms are compared with native CPython 3.13.7 built from the same tarball.

## Result

This section is the first run, as pre-registered. F1-F3 were then fixed and every module
rerun, sharded: see "Rerun after the fixes, sharded" below, where protected equals stock on
every finished test.

**The prediction is not met. Both Capstone arms die with a domain fault before the list ends,
and neither exits as native does (139 against 0).**

- **protected** dies in module 6 of 15, at `test_str.StrTest.test_codecs`. On stock the same
  test raises `MemoryError`: the test is recorded as an ERROR and the process continues. On
  protected, pymalloc's failed allocation reaches patch 0014's `block_issue` as NULL and faults
  (finding F1). This is the one per-test difference between protected and stock.
- **stock** dies in module 9 of 15, at
  `test_json.test_recursion.TestCRecursion.test_endless_recursion`. The fault is in patch 0008's
  trashcan path (finding F2), and both images carry that patch. The protected arm never got that
  far.
- **Tests both arms ran:** protected ran 1149 tests, all of which stock ran too. 1148 have the
  same outcome. The one that differs is `test_codecs`: an ERROR on stock, a process death on
  protected.
- **Protected against native where stock equals native:** 0 differences.
- **Not compared:** 331 tests that stock ran were never reached on protected. Modules 10-15
  (`test_gc`, `test_weakref`, `test_collections`, `test_itertools`, `test_sort`, `test_re`) ran
  on neither arm.
- **Repeats:** each Capstone arm was booted twice (see Deviations). The repeat gave the same
  test ids, the same outcome for every test, and the same fault pc offset within the image.

## Per module, first boot of each arm

The cell format is pass/failure/error/skipped/process-died, counting test ids. A failing
subtest is folded into its test. Native is the matched build; the default build is identical
test for test.

| module | native | stock | stock status | protected | protected status |
|---|---|---|---|---|---|
| test_dict | 114/0/0/0/0 | 113/0/0/1/0 | passed, 1 min 36 s | 113/0/0/1/0 | passed, 1 min 57 s |
| test_list | 65/0/0/0/0 | 64/0/0/1/0 | passed, 2 min 7 s | 64/0/0/1/0 | passed, 2 min 27 s |
| test_set | 623/0/0/0/0 | 623/0/0/0/0 | passed, 41 min 22 s | 623/0/0/0/0 | passed, 43 min 27 s |
| test_tuple | 36/0/0/0/0 | 36/0/0/0/0 | passed, 2 min 52 s | 36/0/0/0/0 | passed, 3 min 23 s |
| test_bytes | 292/0/0/5/0 | 290/0/0/7/0 | passed, 32 min 23 s | 290/0/0/7/0 | passed, 33 min 36 s |
| test_str | 130/0/0/6/0 | 126/1/1/8/0 | failed (1 error, 4 failures), 16 min 44 s | 12/0/0/1/1 | **fault** in test_codecs |
| test_float | 49/0/0/1/0 | 48/0/0/2/0 | passed, 2 min 23 s | not reached | |
| test_long | 42/0/0/2/0 | 42/0/0/2/0 | passed, 22 min 13 s | not reached | |
| test_json | 181/0/0/3/0 | 113/0/0/1/1 | **fault** in test_endless_recursion | not reached | |
| test_gc | 51/0/0/1/0 | not reached | | not reached | |
| test_weakref | 135/0/0/2/0 | not reached | | not reached | |
| test_collections | 114/0/0/1/0 | not reached | | not reached | |
| test_itertools | 139/0/0/5/0 | not reached | | not reached | |
| test_sort | 21/0/0/0/0 | not reached | | not reached | |
| test_re | 161/0/0/4/0 | not reached | | not reached | |
| **total** | 2153/0/0/30/0 of 2183 | 1455/1/1/22/1 of 1480 | exit 139 | 1138/0/0/10/1 of 1149 | exit 139 |

Native "pass" includes unittest's expected failures; none occurred.

## Every per-test difference

**Protected against stock**, over the 1149 tests both ran:

| test | native | stock | protected |
|---|---|---|---|
| test_str.StrTest.test_codecs | pass | error (`MemoryError` at test_str.py:2320, building a string of 1,112,064 code points) | process died: domain fault cause=24 (F1) |

**Stock against native**, over the 1480 tests both ran. These are the port's differences.
Protected agrees with stock on each of them that it reached.

| test | native | stock | protected | reason |
|---|---|---|---|---|
| test_bytes.ByteArrayTest.test_obsolete_write_lock | pass | skipped | skipped | No module named '_testcapi' |
| test_bytes.BytesTest.test_from_format | pass | skipped | skipped | No module named '_ctypes' |
| test_dict.DictTest.test_repr_deep | pass | skipped | skipped | requires _testcapi |
| test_list.ListTest.test_repr_deep | pass | skipped | skipped | requires _testcapi |
| test_str.StrTest.test_codecs | pass | error | process died | stock: `MemoryError`; protected: F1 |
| test_str.StrTest.test_formatting_c_limits | pass | skipped | not reached | No module named '_testcapi' |
| test_str.StrTest.test_formatting_huge_precision_c_limits | pass | skipped | not reached | need _testcapi module |
| test_str.StrTest.test_raiseMemError | pass | failure | not reached | 4 subtests: `sys.getsizeof(char * 42)` is 8 bytes larger than the test's struct_size + data; e.g. 107 != 99 for 'a' |
| test_float.IEEEFormatTestCase.test_serialized_float_rounding | pass | skipped | not reached | needs _testcapi |
| test_json.test_recursion.TestCRecursion.test_endless_recursion | pass | process died | not reached | F2 |

`_testcapi` and `_ctypes` are not built into the image. `--disable-test-modules` and the
static build leave `_testcapi` out, and `CPY_TEST_CAPI=1` adds only `_testinternalcapi` and
`_testlimitedcapi`. The `test_raiseMemError` size difference is the object layout with 16-byte
pointers: the test hard-codes the 64-bit CPython struct sizes. Why `test_codecs` runs out of
memory on Capstone was not established in this run. The image records `arena_bytes`
67108864, and `runtime/virtual/vm.h` caps one mapping at 256 MiB; neither limit was shown to
be the cause.

## Findings

**F1: protected faults where stock raises `MemoryError` (patch 0014).**
`Objects/obmalloc.c` with 0014 applied ends `pymalloc_alloc` with
`return block_issue(state, bp, nbytes);`. Upstream returns `bp` itself. On this path `bp`
comes from `allocate_from_new_pool`, which returns NULL when `new_arena` fails. Upstream's
NULL makes `_PyObject_Malloc` fall back to the system allocator, which then raises
`MemoryError`. With 0014, `block_issue` computes `POOL_ADDR(bp)` with a `cincoffset` on the
untagged NULL. QEMU logs `cincoffset with an UNTAGGED rs1 -- pc=... rs1=x10 val=0x0`, and the
launcher reports `domain fault cause=24`. Both boots faulted at the same image offset 0x15c088
(vaddr 0x16c088, in `_PyObject_Malloc`). The debug-linked `link/python.dom` of the same build
symbolizes that offset to `block_issue` (obmalloc.c:1832), inlined into `pymalloc_alloc`
(:2294). This is a correctness difference introduced by the protection, recorded and not
patched.

**F2: stock (and every image carrying patch 0008) faults in the trashcan.**
`_PyTrash_thread_deposit_object` calls
`_PyGCHead_SET_PREV(_Py_AS_GC(op), tstate->delete_later)`. With patch 0008 that becomes
`_PyGCWord_MAKE(prev, bits)`, and the code is a `cincoffset` on `prev`. When nothing has been
deferred yet, `delete_later` is NULL (untagged), and the first deposit faults (cause=24,
`pycore_gc.h:228`, `object.c:2822`). Both stock boots faulted at image offset 0x15610c
(vaddr 0x16610c, `_PyTrash_thread_deposit_object`), in `test_endless_recursion`, whose deep
deallocation reaches the trashcan. The protected image carries the same patch but died earlier.

**F3: `xml.etree.ElementTree.Element('a')` faults on both images (`_elementtree.c`).**
`_set_joined_ptr` (`_elementtree.c:67`, called from `element_init` :455) applies `JOIN_OBJ` to
`self->text`, which masks the pointer through `uintptr_t & ~1`. The compiled code loads the
field with an integer `ld` and dereferences the masked integer, and `_Py_NoneStruct` is
reached without a capability (cause=24). Each arm's boot runs a one-line probe before the
suite, and the probe exits 139 on both arms in all four boots. The first timing boot used
`--junit-xml`, which builds its report with ElementTree, and it faulted there before writing
any result. That is why the Capstone arms are recorded from the `-v` log (Deviations).

Nothing here was patched.

## What ran

- **Source.** `git archive` of 9bc6f9adf1c9 (`capstone/`, `compiler-rt/lib/builtins`),
  committed on the build host as the snapshot e801018c. Both image manifests record
  `recipe_revision`/`runtime_revision` e801018c and `runtime_dirty: true`. After the builds,
  `git status --porcelain` in the snapshot listed exactly one untracked file:
  `capstone/ports/cpython/app/__pycache__/survey-cpython-capstone.cpython-314.pyc`. The build
  wrote it itself when the survey script was imported. `git diff HEAD` was empty. A fresh
  `git archive 9bc6f9adf1c9` of `capstone/` and `compiler-rt/lib/builtins` compares equal to
  the snapshot (`diff -r`, excluding `.git` and `__pycache__`). No source file differed.
- **Tarball.** `Python-3.13.7.tgz` sha256
  6c9d80839cfa20024f34d9a6dd31ae2a9cd97ff5e980e969209746037a5153b2, as `upstream.json` pins it.
  The Capstone builds used the kit's `inputs/` copy. The native builds used a download of the
  same file with the same hash, made before the kit was complete.
- **Images.** Built as
  `CPY_TEST_CAPI=1 [CPY_SUBLET=1] build-virtual.sh cpython <fresh dir>`, with `JOBS=24` and
  `CPY_JOBS=16`, in 223 s each, in parallel. Both used the toolchain
  `compiler_sha256` 4c6135feafae20c2..., runtime archive fc719e2acf7a922a..., and builtins
  archive a72ed0bf31407f08....

  | | stock | protected |
  |---|---|---|
  | `cpython.dom` sha256 | 6760ff8773f450b044fbdce3ab6432cde3988c5c51aff25142eb4b1af61fec13 | 36cececaab98fb19aee0b905331a768c63c1a19341addc24539c881639cbeeea |
  | manifest `nested` | none | cpython |
  | patches applied | 0001-0005, 0007-0013, 0015, 0016 (14) | the same plus 0014 (15) |
  | `heap`, `profile`, `stack_bytes`, `arena_bytes` | virtual, virtual, 8388608, 67108864 | same |

- **Platform.** One hart, exact bounds, `run-staged.py` from the snapshot, with the kit's
  `kit-sublet`:

  | file | sha256 |
  |---|---|
  | qemu-system-riscv64 (QEMU 8.1.1) | af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d |
  | Image | e833dfb12bccf8a39bd6a81e8e42e4622f0e76bcc6cc09624a1fbe7198c18a4b |
  | fw_jump.elf | 44cd6cbc2d225c2fea6c6000ea0b9eab83bc34e769a7c95c036cf17b6fe0f116 |
  | rootfs.ext2 | 62049800ed0fa8827959c0060bbd74f0435aacb577dd521e14a14a099d167845 |
  | capstone-vexec | 3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4 |
  | capstone_vm.ko | a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512 |

- **Staging.** One staged disk per boot. It holds the launcher, the module, and
  `/mnt/vm/<arm>/cpython.dom`, with `/mnt/vm/<arm>/lib/python3.13` beside it, as this port's
  README requires so that posix_spawn children find the stdlib from `sys.executable`. The
  stdlib is the build's own source `Lib/`, which compares equal to the tarball's `Lib/`. The
  stage adds the build's `_sysconfigdata__linux_.py` and an empty `lib-dynload`, then compiles
  everything with the native 3.13.7 (`compileall -f -d /mnt/vm/<arm>/lib/python3.13`). Five
  deliberately malformed files under `test/` do not compile. The two arms' stdlib trees differ
  only in `_sysconfigdata__linux_.py` (build paths and 0014's `-I` in `CONFIGURE_CPPFLAGS`)
  and in the bytecode's embedded paths. Stage tree hashes and gate-script hashes are in
  `outcomes.json` under `stage`. A smoke boot checked that
  `sys.executable` is `/mnt/vm/stock/cpython.dom` and that a `-I` child it spawns finds
  `sys.prefix` `/mnt/vm/stock`.
- **Guest command**, as root, `HOME=TMPDIR=/tmp`, `CAPSTONE_EXEC_DIAGNOSTICS=1`, cwd `/tmp`:

      PYTHONHOME=/mnt/vm/<arm> /mnt/vm/capstone-vexec /mnt/vm/<arm>/cpython.dom -m test -v \
        test_dict test_list test_set test_tuple test_bytes test_str test_float test_long \
        test_json test_gc test_weakref test_collections test_itertools test_sort test_re

  The header reads "Run 15 tests sequentially in a single process". No `-u` was given, so
  resource-gated tests skip on every arm alike. No `-j`, `-r` or `--rerun` was given. After
  each boot, `capstone-vexec --stats` reported `live_domains`, `live_regions`, `live_bytes` and
  `tag_pages` all 0. On the first boots, `nodes_high_water` was 33534 for stock and 1824725
  for protected.
- **Native.** Two builds of the same tarball on the x86_64 host (gcc 15.2.0, glibc 2.43):
  - *matched*: `configure --disable-ipv6 --without-ensurepip --without-computed-gotos`;
  - *default*: plain `configure`. This is what `prepare-cpython-capstone.sh` builds as its
    build-python, a full in-tree interpreter that runs the list.

  Both were built with `make -j32` and run from the build tree as
  `./python -m test -v --junit-xml <file> <the 15 tests>` by an unprivileged user. Regrtest
  took 17.6 s on the matched build and 17.4 s on the default build. The two agree test for
  test. Native is the matched build.
  The Capstone-specific options were left out of native: `--disable-test-modules`, static
  modules, `-D_Py_FORK_EXEC_POSIX_SPAWN`, and the config.site answers. A first matched build
  with `--disable-test-modules` skipped all of `test_bytes` ("No module named
  '_testlimitedcapi'"), so native keeps its test modules.
- **Outcomes.** The Capstone arms' outcomes are parsed from regrtest's `-v` lines on the serial
  console. On both native logs, the same parser produces the 2183 test ids and outcomes of the
  native `--junit-xml`, one for one.

## Wall times

| arm | regrtest in guest | boot total |
|---|---|---|
| native (matched / default) | 17.6 s / 17.4 s | |
| stock, boot 1 / boot 2 | 7412 s / 7384 s (to the fault in module 9) | 7447 s / 7419 s |
| protected, boot 1 / boot 2 | 5296 s / 5357 s (to the fault in module 6) | 5334 s / 5391 s |

Per module, protected took 1.04-1.22 times as long as stock on the five modules both
completed.

## Deviations from the instructions

1. **`CPY_TEST_CAPI=1` on both images.** Without `_testlimitedcapi`, regrtest skips all of
   `test_bytes`, which would have made one of the 15 rows empty. The switch adds
   `_testinternalcapi` and `_testlimitedcapi`, the same on both arms.
2. **No `--junit-xml` on the Capstone arms**, because of F3. The `-v` log is the record.
3. **Two boots for each Capstone arm (N = 2 rather than 1).** The timing boot measured `test_dict`
   at 97 s against 0.25 s natively. `test_set` then took 41 min against 1.7 s natively, and the
   extrapolation came close to the 6 h timeout of the first pair. A second pair with a 24 h
   timeout was therefore started 48 min after the first, on separate locks and work dirs. Each
   repeat matches its first boot test for test, so the table reports the first boots.
4. **Two earlier timing boots of the stock image** on `test_dict` alone. One used
   `--junit-xml` and faulted (F3). The other passed: 114 tests, 1 skipped, 97.3 s.
5. **Arms in parallel boots**, each with its own lock file and work directory, plus
   `compileall` over the `test` package as well as the stdlib.

## Files

- `outcomes.json`: per-test outcomes for native, stock and protected, with skip messages and
  folded subtests. It also holds module statuses and counts for all six runs (native matched
  and default, both boots of both Capstone arms), the repeat comparisons, every difference
  with the number compared, and the stage hashes.

## Rerun after the fixes, sharded (2026-10-11, second run)

At the project lead's request, F1, F2 and F3 were fixed in the port. Both images were rebuilt
from one snapshot, and all 15 registered modules were rerun on both arms. To shorten the wall
time, the run was split into shards: one boot per module, with the slow modules split further
by TestCase class. That is a deviation from "sequential in one process", recorded below. The
per-test record is `outcomes-sharded.json`.

### Result

**With the fixes, protected gives the same outcome as stock on all 2183 test ids native ran.
Every test finished on both arms.** No boot faulted on either arm.

- **Protected against stock:** 0 differences among the 2183.
- **Protected against native where stock equals native:** 0 differences.
- **Stock against native:** 21 differences, listed below. Protected
  has the same outcome as stock on each of them. They are the port's differences.

### The fixes

Each fix is separate and touches one patch file. All 16 patches apply to the 3.13.7 tarball
with `--fuzz=0` and no offsets, with 0014 (protected) and without it (stock).

| finding | file | change |
|---|---|---|
| F1 | `patches/cpython-3.13.7-0014-pymalloc-sublet-lifetimes.patch` | `pymalloc_alloc` returns NULL before `block_issue` when `allocate_from_new_pool` found no arena, as upstream returns its NULL, so `_PyObject_Malloc` falls back and reports `MemoryError`. `block_issue` has one other caller, `pymalloc_realloc`, which passes `arena_inside(ao, p)` for a live block and never NULL. |
| F2 | `patches/cpython-3.13.7-0008-gc-links-as-pointers.patch` | `_PyGCHead_SET_PREV` with a NULL `prev` stores the flag bits alone (upstream's `0 \| flags`) instead of offsetting NULL. `_PyGCHead_PREV` returns NULL for a word with no address part instead of stripping flags from it. That word is the end of the trashcan's `delete_later` chain, which `_PyTrash_thread_destroy_chain` reads back. |
| F3 | new `patches/cpython-3.13.7-0017-elementtree-join-flag-keeps-pointer.patch` | `JOIN_OBJ` and `JOIN_SET` in `Modules/_elementtree.c` become inline functions that subtract or add the flag on the pointer itself, so the result keeps the pointer's capability. They do arithmetic only when the flag is set or is being set, so NULL (`element_gc_traverse` after a clear) passes through. |

Patch sha256: 0008 bed188e9199787e5..., 0014 6c2c170ef57e4988..., 0017 1115042c431f9296...
The originals were 0008 675e023473daaaa6... and 0014 84ca683d3977ecca....

**Native checks** (x86_64, gcc 15.2.0, the matched configure options):
- **F3:** the tarball plus 0017 passes `test_xml_etree` and `test_xml_etree_c` with 428 run and
  12 skipped, the same counts as the unpatched native build.
- **F2:** the tarball plus every patch except 0014, built with
  `CFLAGS=-D_Py_GC_WORD_AS_POINTER` (the pointer representation that 0008 selects on Capstone,
  confirmed in `sysconfig` `CFLAGS`), passes `test_gc`, `test_json`, `test_weakref`,
  `test_xml_etree`, `test_xml_etree_c`, `test_dict`, `test_list` and `test_set`: 1603 run,
  18 skipped. That includes `test_trashcan`, `test_trashcan_threads` and both
  `test_endless_recursion`. On x86 an offset NULL does not fault, so this checks that the fix
  keeps behaviour, not that it removes the fault.
- **F1:** 0014 uses Capstone instructions and does not build natively. It is checked on the
  image only.

**On the images:**
- **F3:** each boot ran `ET.Element('a')` with text and tail set and printed
  `ETREE-OK b'<a>x</a>y'`, with exit 0, in all 60 boots.
- **F2:** `test_json…test_endless_recursion` (C and Py), `test_gc…test_trashcan` and
  `test_trashcan_threads` pass on both arms.
- **F1:** protected's `test_codecs` raises `MemoryError` as stock's does (ERROR), and the shard
  runs to its end.

### What ran

- **Snapshot.** `git archive 9bc6f9adf1c9` plus the three patch files, committed on the build
  host as 44dc50b1. `git status` afterwards showed only the build's own `__pycache__`.
- **Images.** `CPY_TEST_CAPI=1 [CPY_SUBLET=1] build-virtual.sh cpython <fresh dir>`, built in
  233 s each, in parallel.
  - *stock* `cpython.dom`: 5f761dfe44a3bebffc81f4f0726945550919698070fd7860727d9d03742fd867,
    with 15 patches (0017 added) and `nested: none`.
  - *protected* `cpython.dom`: 675f609db6ceca12d3aa4409855d17806d79bf0dab48cb8d8ee5e007484cbd14,
    with 16 patches and `nested: cpython`.

  The compiler, runtime and builtins archive hashes and the stack and arena sizes are as in
  the first run. The platform files and the staging are as in the first run, the stage hashes
  are in `outcomes-sharded.json`, and the guest command is the same with `-m PATTERN ...`
  added.
- **Shards.** 29 planned per arm. Each class's estimate is its native junit time multiplied by
  the module's guest/native ratio from the first run (1800 where the first run had no ratio),
  plus 0.5 s per test. Classes are packed into shards of at most 1200 s estimated. Tests whose
  class exceeded that went out one test per unit, and doctests were kept together per module.
- **Coverage.** Each shard's `-m` patterns were run through the native interpreter's
  `--list-cases`. The union is the 2183 native test ids, each exactly once. On both arms the
  parsed logs hold all 2183 ids with the native multiplicities.
- **Boots.** 58 boots, one per arm and shard, each with its own lock and work dir, launched
  detached at 11:56 UTC on one host next to 4 other QEMU processes. The 52 that hold no
  threaded weakref test finished by about 12:36 UTC. The longest took 2371 s
  (`protected/test_long-10`); the shortest regular shard took 345 s. Running 58 VMs at once
  made each about 3 times slower than a lone VM: `test_dict` took 270 s against 91 s in the
  first run.
- **Shard 21b.** In shard `test_weakref-21`, the threaded test
  `test_threaded_weak_valued_setdefault` runs first and was still running after 40 min, with
  about 55 tests queued behind it. Those 55 (`ReferencesTestCase.*` and four MappingTestCase
  tests) were therefore started again as `test_weakref-21b` on each arm, and finished in
  240-254 s. Their outcomes come from 21b. Only the setdefault test is taken from shard 21.

### Per module, sharded run

Cells are pass/failure/error/skipped/process-died/not-finished. Native is the matched build
from the first run.

| module | native | stock | protected |
|---|---|---|---|
| test_dict | 114/0/0/0/0/0 | 113/0/0/1/0/0 | 113/0/0/1/0/0 |
| test_list | 65/0/0/0/0/0 | 64/0/0/1/0/0 | 64/0/0/1/0/0 |
| test_set | 623/0/0/0/0/0 | 623/0/0/0/0/0 | 623/0/0/0/0/0 |
| test_tuple | 36/0/0/0/0/0 | 36/0/0/0/0/0 | 36/0/0/0/0/0 |
| test_bytes | 292/0/0/5/0/0 | 290/0/0/7/0/0 | 290/0/0/7/0/0 |
| test_str | 130/0/0/6/0/0 | 126/1/1/8/0/0 | 126/1/1/8/0/0 |
| test_float | 49/0/0/1/0/0 | 48/0/0/2/0/0 | 48/0/0/2/0/0 |
| test_long | 42/0/0/2/0/0 | 42/0/0/2/0/0 | 42/0/0/2/0/0 |
| test_json | 181/0/0/3/0/0 | 180/0/1/3/0/0 | 180/0/1/3/0/0 |
| test_gc | 51/0/0/1/0/0 | 46/0/0/6/0/0 | 46/0/0/6/0/0 |
| test_weakref | 135/0/0/2/0/0 | 134/0/0/3/0/0 | 134/0/0/3/0/0 |
| test_collections | 114/0/0/1/0/0 | 114/0/0/1/0/0 | 114/0/0/1/0/0 |
| test_itertools | 139/0/0/5/0/0 | 137/1/1/5/0/0 | 137/1/1/5/0/0 |
| test_sort | 21/0/0/0/0/0 | 21/0/0/0/0/0 | 21/0/0/0/0/0 |
| test_re | 161/0/0/4/0/0 | 160/2/0/3/0/0 | 160/2/0/3/0/0 |
| **total** | 2153/0/0/30/0/0 | 2134/4/3/42/0/0 | 2134/4/3/42/0/0 |

### Every difference from native (stock = protected in every row)

| test | native | stock and protected | reason |
|---|---|---|---|
| test_bytes.ByteArrayTest.test_obsolete_write_lock | pass | skipped | No module named '_testcapi' |
| test_bytes.BytesTest.test_from_format | pass | skipped | No module named '_ctypes' |
| test_dict.DictTest.test_repr_deep | pass | skipped | requires _testcapi |
| test_list.ListTest.test_repr_deep | pass | skipped | requires _testcapi |
| test_float.IEEEFormatTestCase.test_serialized_float_rounding | pass | skipped | needs _testcapi |
| test_str.StrTest.test_formatting_c_limits | pass | skipped | No module named '_testcapi' |
| test_str.StrTest.test_formatting_huge_precision_c_limits | pass | skipped | need _testcapi module |
| test_gc.GCCallbackTests.test_collect_garbage | pass | skipped | requires _testcapi.with_tp_del |
| test_gc.GCTests.test_garbage_at_shutdown | pass | skipped | requires _testcapi |
| test_gc.GCTests.test_legacy_finalizer | pass | skipped | requires _testcapi.with_tp_del |
| test_gc.GCTests.test_legacy_finalizer_newclass | pass | skipped | requires _testcapi.with_tp_del |
| test_gc.GCTests.test_trash_weakref_clear | pass | skipped | requires ContainerNoGC extension type |
| test_weakref.ReferencesTestCase.test_cfunction | pass | skipped | No module named '_testcapi' |
| test_re.ReTests.test_regression_gh94675 | pass | skipped | test requires multiprocessing |
| test_re.ReTests.test_locale_caching | skipped (needs en_US.iso88591) | failure | the guest's `setlocale` accepts the name, so the test runs, and `re.L` case folding of b'\xc5'/b'\xe5' does not match; upstream marks the same as a "musl libc issue" (bpo-46390) and skips it only on Emscripten/WASI |
| test_re.ReTests.test_locale_compiled | skipped (same) | failure | same |
| test_str.StrTest.test_codecs | pass | error | `MemoryError` building a string of 1,112,064 code points |
| test_str.StrTest.test_raiseMemError | pass | failure | `sys.getsizeof` 8 bytes over the test's 64-bit struct sizes (4 subtests) |
| test_itertools.SizeofTest.test_product_sizeof | pass | failure | 144 != 136: object size with 16-byte pointers |
| test_itertools.TestBasicOps.test_tee_del_backward | pass | error | `MemoryError` |
| test_json.test_recursion.TestPyRecursion.test_highly_nested_objects_decoding | pass | error | `SystemError: error return without exception set` (new, see F4) |

**F4, both arms, not investigated (a port difference, not a protection difference).**
The test expects `RecursionError` from
`json.loads('{"a":' * 100000 + '1' + '}' * 100000)` under `support.infinite_recursion()`, with
the pure-Python decoder, and natively it passes. On both images it ends in
`SystemError: error return without exception set`. Some C path returns NULL without an
exception on the deep recursion. The sequential first run never reached this test.

### The three threaded weakref tests

`MappingTestCase.test_threaded_weak_valued_consistency`, `test_threaded_weak_valued_pop` and
`test_threaded_weak_valued_setdefault` each run 100,000 dictionary operations while a second
thread calls `gc.collect()` every 5 ms. Natively they take 1.8 s, 0.8 s and 0.2 s. On one
emulated hart they take hours, and the time varies from boot to boot. The boot cap of the
planned shards is 4 h. At 13:21 UTC, consistency and pop were therefore started again on each
arm in boots with a 24 h cap (`test_weakref-17L`, `test_weakref-19L`). Such a boot supplies an
outcome only where the 4 h shard has none; otherwise it is recorded as a repeat.

| test | stock | protected |
|---|---|---|
| setdefault | pass, shard 21, 51 min 10 s | pass, shard 21, 56 min 44 s |
| pop | pass, shard 19, 3 h 5 min; repeat in 19L: pass, 4 h 17 min | 4 h shard hit the cap; pass in 19L, 4 h 50 min |
| consistency | 4 h shard hit the cap; pass in 17L, 8 h 27 min | 4 h shard hit the cap; pass in 17L, 8 h 40 min |

The times are regrtest's own ("passed in ..."). No fault was logged in any of these boots. A
check at 13:20 UTC found QEMU at 100% CPU in the six 4 h boots, and a check at 18:20 UTC found
it at 100% in both 17L boots. The 17L boots ended at 21:51 UTC (stock) and 22:04 UTC
(protected), exit 0, both with `Ran 1 test` and `OK`.

### Deviations in this run

1. **Sharded instead of sequential in one process.** Each shard is sequential in its own
   process. Cross-test interactions inside one interpreter are not the same as in a full
   sequential run, and the 58 parallel boots shared one host.
2. **Shard 21b** re-ran 55 tests that were queued behind a long threaded test (above). The
   4 h shard 21 later ran them too, with the same outcome on both arms (`repeats` in
   `outcomes-sharded.json`).
3. **Native was not rerun.** The oracle is the first run's matched native build. The fixed
   patches do not apply to it, because the native oracle is the unpatched release.
4. **24 h-cap repeats of two threaded tests** (17L, 19L), because their 4 h shards hit the cap
   or were expected to (above).
