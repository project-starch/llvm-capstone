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
