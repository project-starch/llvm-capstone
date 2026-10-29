# Provenance

**Upstream fix:** `gh-151403`, `5810def6dd` on the 3.13 branch (back-port of `6679ac07d8` on main), after our `v3.13.7`
tag. **Consumer:** the freed block's pymalloc pool. **CVE:** `NO VERIFIED CVE`.

## The same defect exists twice in this tree, on purpose

`../pymalloc-repros/16_gh-151403_fork_exec_fspath` is this defect as a C model consumer: a sequence against the real pymalloc with a
labelled probe, so a fault can be attributed to one instruction. This entry is the same defect
reached in place: the reproducer from upstream issue #151403 (the fix adds no test), with one argument appended: main's fork_exec takes 22 arguments, 3.13.7's takes 23 (allow_vfork).

Neither supersedes the other. The C model is stronger for attribution; this one is stronger for
liveness, because the defect is reached in the whole interpreter. They are not two defects and must
not be counted twice.

## Liveness at the pin, measured

measured 2026-10-29 on the pinned 3.13.7 ASan build (--with-address-sanitizer, gcc 15.2, p13) with PYTHONMALLOC=malloc and ASAN_OPTIONS=detect_leaks=0: ASAN:heap-use-after-free in a 328-byte region. Running the trigger is the proof. Case 28's trigger reported its recorded 328-byte heap-use-after-free in the same build.

## Which allocator owns the object

**nested.** `allocator_layer` = `L1-pymalloc-pool`, `allocator_consumed` = `yes`.
measured 2026-10-29: ASan reported a 328-byte region, at or under the 512 B SMALL_REQUEST_THRESHOLD, and the heap report does not survive PYTHONMALLOC=pymalloc on a second 3.13.7 build with ASan and pymalloc both on (CFLAGS/LDFLAGS -fsanitize=address, because configure's --with-address-sanitizer forces pymalloc off): under PYTHONMALLOC=pymalloc the run completes and raises the expected TypeError with no report: the stale read lands inside a pymalloc block

## Negative control

`negative_control.py` keeps the trigger's traffic and makes the offending access valid: a second reference keeps the Evil item alive while args is cleared.
It ran to its end with no report on the same ASan build, and prints `NEGATIVE-CONTROL no defect performed`.
