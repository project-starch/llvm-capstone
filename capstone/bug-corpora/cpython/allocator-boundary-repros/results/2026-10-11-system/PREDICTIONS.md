# Cases 19-21 on the virtual arms: predictions

Registered 2026-10-11, before either image below was built and before any case ran on it. A result
that differs is a finding; this file keeps the prediction that was made.

## What the run is for

Cases 19, 20 and 21 are this corpus's three temporal defects whose memory comes straight from
libc malloc, never from a pymalloc pool. On the virtual profile that is musl mallocng
(`capstone/runtime/virtual/heap.c`), which bounds each object to its request and retires its
lifetime on free (`rotate()` on every slot, the mmap-sized ones included). They are the CPython row's
system-allocator cell. On `2026-10-10-sublet-isa` one of the three was a reading:

| case | 2026-10-10 virtual-malloc and virtual-nested-pools | why it was not a reading |
|---|---|---|
| 19 | NO-READING, cause 25 in `fnv` | `fnv` is not a fault site: host ASan reported `siphash13` |
| 20 | CAUGHT, cause 25 in `memcpy` | |
| 21 | NO-READING, cause 28 in `PyFunction_NewWithQualName` | a store below the stack: the image's stack is 1 MiB |

## What changes, and why each change is not fitted to the answer

1. **The image's stack is 8 MiB** (`build-virtual.sh` for cpython, `build.py --stack-bytes`).
   Both 2026-10-10 images carry 1 MiB in `.capstone_domreq` (0x100000; `capstone-vexec` maps the
   main stack from that field). CPython 3.13.7 release builds set `Py_C_RECURSION_LIMIT` to 10000
   (`Include/cpython/pystate.h`), a limit sized for the 8 MiB main stack Linux gives a process; host
   ASan ran under that 8 MiB. Case 21's trigger re-enters `Parse` from its own handler, so the
   interpreter recurses until its guard or the stack ends, and with 1 MiB the stack ended first.
   The change applies to every CPython build, not to case 21.
2. **The runner runs a case's `negative_control.py`** when a fault is reached and no fault site
   names it, the rule `postgres/sql-repros/shared/run-arm.py` already applies with its
   `control.sql`. The control counts only when it prints `NEGATIVE-CONTROL no defect performed`,
   does not print `SELFTEST-FAILED`, ends with `ABR RETURNED` and does not fault. A control that
   faults leaves the case unattributed. Case 19 has one (committed in fb8461483f40, before any
   virtual run): the same upstream test file through the same path, with `__hash__` no longer
   releasing the buffer. 20 and 21 have none.
3. **No fault site is added.** Case 19 faults in `fnv` because this build hashes bytes with FNV:
   `pyconfig.h` defines `HAVE_ALIGNED_REQUIRED 1` and leaves `Py_HASH_ALGORITHM` undefined, and
   `Include/pyhash.h` then selects `Py_HASH_FNV` where the x86-64 ASan build selects
   `Py_HASH_SIPHASH13`. That explains the site, but it was read after the fault was seen, so it
   is not used to attribute it. The negative control is.

## Predictions

Both arms, `--only 19,20,21`, on the Sublet platform (QEMU af37cc32, module a1c6cb6b, launcher
3975314a). pymalloc plays no part in these three, so the two arms are predicted identical.

| case | virtual-malloc | virtual-nested-pools |
|---|---|---|
| 19 | CAUGHT, cause 25 in `fnv`, by control | the same |
| 20 | CAUGHT, cause 25 in `memcpy`, by function | the same |
| 21 | CAUGHT, cause 25 in `little2_toUtf8`, by function | the same |

What would falsify each:

* 19: the negative control faulting (then the `fnv` fault is not the defect's), or failing its
  self-test on this interpreter (then there is no reading).
* 20: anything else; nothing it depends on changed except the stack.
* 21: a cause-28 fault on the stack again (8 MiB is not enough on Capstone frames, a finding about
  the stack and not about the heap), a fault outside `little2_toUtf8`, or the trigger ending
  without a fault (the stale read did not happen, or reached memory still live).

The workload and the two pymalloc controls (`uaf-block`, `bounds-block`) must pass as
`tools/arms.json` says before any row counts.

## Amendment, 2026-10-11, before any output of the run was read

The run of 9239e39689e9 started on p13 at once, and while its images were building I found
d3a500f411cb on `corpus/cpython-controls-all-arms`, which this branch did not hold: case 19's
trigger runs the whole upstream file, and on the physical sublet arm that file faults at three
distinct sites (gh-142664, this case; gh-143195, a use-after-free in `memoryview.hex(sep)`;
gh-92888's regression test). The whole-file control neutralises gh-142664 only, so it is expected
to fault at one of the other two sites. The rule registered above would then read "the control
faults too" as "the `fnv` fault is not the defect's", which does not follow: a fault at a
different site in a different test says nothing about this one.

So, before any case output of that run was read, this branch merges d3a500f411cb (the probes and
the control scoped to gh-142664's six tests) and changes the rule for a case that declares a
`control_probe` (case 19: `probe-142664.py`): the probe must fault in the trigger's function at
the trigger's offset, and the scoped control must then run clean. Cases without one keep the rule
above.

The run already under way used the whole-file control and keeps its output as recorded. Cases
19, 20 and 21 are then rerun with the amended runner on the same two images, so every row comes
from one runner revision. The merge changes no build input (nothing under `capstone/ports`,
`capstone/runtime` or `capstone/host`), so the images built from 9239e39689e9 are the images this
commit would build.

Prediction for case 19 under the amended rule, both arms: `probe-142664.py` faults with cause 25
in `fnv` at the trigger's offset, `negative_control.py` runs its six tests clean, and the row is
CAUGHT by control. Falsifiers: the probe not faulting, or faulting elsewhere (then the whole-file
fault in `fnv` is not shown to be gh-142664's), or the scoped control faulting.

## Second amendment: case 21's timeout

On the run of 9239e39689e9 case 21 ended `NO-READING:infra`, runner timeout, on both arms: the
trigger was still running after the runner's 300 s, with no fault (so not the 1 MiB run's stack
overflow either). Natively on p13 the same trigger takes 1.7 s on the build's own 3.13.7 (it ends
on SIGSEGV, rc 139, the stale read hitting the unmapped 512 KiB buffer) and 2.1 s on the ASan
build (heap-use-after-free, READ of size 1). Case 21 is rerun alone with `--timeout 3600` on the
same images and runner. A timeout is not a reading in either direction, so lengthening it does
not choose the answer; the prediction for case 21 above stands unchanged.

## CheriBSD, registered 2026-10-11 before the interpreter was built

The CheriBSD column of 2026-10-06 to -09 was measured on another machine with an si_code reporter
and a mech-control program that never reached the repository, on a binary whose hash was
reconstructed afterwards (`results/20261006/README.md`). Neither the binary nor the helpers exist
on any machine this lane can reach, so the arm is rebuilt from the repository:

* the interpreter from `capstone/ports/cpython/app/cheribsd/build.sh`, which now passes
  `-cheri-tgot-tls` (the SDK's own purecap configuration carries it) and builds and stages the
  extension modules (`make python sharedmods`, `lib-dynload`). Before, it built `python` alone,
  so `_pickle`, `pyexpat` and `array` were missing: case 20's `pickle` would have fallen back to
  pure Python, where the defect cannot happen, and case 21 would have stopped at its import. The
  recipe now refuses a build without `_pickle`, `pyexpat`, `array` and `_struct`. Three things
  do not compile for purecap and are configured out, each named in the recipe: `select.kqueue`
  (it keeps a kevent's ident and udata as integers of `uintptr_t`'s size, 16 bytes here),
  `_multiprocessing` (no integer format code of `void *`'s size) and `_ctypes` (no libffi in the
  sysroot). None of the three cases uses them. The build of this recipe on p13 staged 57 modules
  beside the interpreter, binary `c9bec41630d4b707...`;
* `capstone/bug-corpora/tools/cheribsd/sicode.c`, a fault reporter preloaded in front of the
  interpreter, and its self-test, which runs under the same preload and must report
  `PROT_CHERI_BOUNDS` for a read past an allocation and `PROT_CHERI_TAG` for a read through a
  freed pointer after a forced revocation pass;
* the quarantine probe of `postgres/mmgr-repros/results/20261008-cheribsd`, preloaded beside it.
  The runner refuses a guest where the workload ends without the probe's line.

Guest: the purecap image e7470361 on p13, booted from snapshot, with revocation as the platform
ships it; `run-cheribsd.sh` refuses `runtime_revocation_default` other than 1 and records the whole
`security.cheri` subtree before and after.

How a row is read for the table's CheriBSD column, which credits use-after-reallocation
protection: the stale pointer must not have reached a reallocated object. A row earns it when
either

* the process ends on SIGPROT with `PROT_CHERI_TAG` (a sweep revoked the stale capability before
  the access), or
* the process ends without a fault and the probe reports every free quarantined
  (`quarantined_after_free` = `frees`) and `reused_while_quarantined=0`: the freed object could
  not be handed out again before a sweep, and a sweep would have untagged the stale capability.

Any other ending (another signal, a bounds fault, a free that did not enter the quarantine, no
probe line on a run that ended normally) earns nothing and is reported as it is.

Predictions:

| case | ending | probe | UAR credit |
|---|---|---|---|
| 19 | no fault: the stale reads come within the same call as the release, before any sweep | all frees quarantined, none reused | yes |
| 20 | no fault, for the same reason | the same | yes |
| 21 | SIGPROT, `PROT_CHERI_TAG`, as the 2026-10-08 run recorded; presumably a sweep runs between the 512 KiB block's free and the read, but the mechanism is not what is predicted, the ending is | none (the process dies before the probe's destructor) | yes |

A use-after-free is caught, as opposed to rendered harmless, only in the third.

## CheriBSD, first run (0bac348cd567): two runner defects, fixed before the rerun

Guest controls all passed: binary c9bec41630d4 on host and guest, `revocation_default=1`,
`every_free_default=0`, the workload `EXP-OK cpython 552` with the probe reporting 4,878 frees, all
quarantined, none reused, no sweep; the self-test `PROT_CHERI_BOUNDS` and `PROT_CHERI_TAG`.

| case | ending | what it showed |
|---|---|---|
| 19 | rc 1, `ModuleNotFoundError: No module named 'test'` | not reached: the stdlib zip leaves out the `test` package, and the upstream file imports `test.support` |
| 20 | rc 0, no fault | the probe line recorded was `allocs=0`: it was `timeout`'s |
| 21 | SIGPROT, `si_code=2 (PROT_CHERI_TAG)` | as predicted |

The two defects, both the runner's:

* it did not stage the `test` package, and it had no reach marker, so an import failure ended
  like a quiet run. Under the rule above, case 19 would have been credited on a run that never
  reached its defect. The runner now stages the release's `Lib/test` beside the zip and wraps
  each trigger in the virtual runner's launcher (`ABR BEGIN`, then `ABR RETURNED`, `ABR EXIT` or
  `ABR RAISED <type>`). An ending on `ImportError` or `ModuleNotFoundError` is marked not reached,
  and a row without `ABR BEGIN` is not started;
* the preload also loads into `timeout`, which allocates nothing, exits after the interpreter and
  reports last. The runner now records only probe lines from a process that mapped the shadow,
  that is, one that allocated.

Added to the rule: a row earns the CheriBSD column only if its trigger began and did not end
on an import error. The three cases are rerun on the same guest image, binary and helpers. The
predictions stand as registered.
