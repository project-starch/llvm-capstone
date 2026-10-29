# Provenance

**Upstream fix:** `gh-153539`, `081e0beae2` on the 3.13 branch (back-port of `fa0ec86ab6` on main), after our `v3.13.7`
tag. **Consumer:** the freed block's pymalloc pool. **CVE:** `NO VERIFIED CVE`.

## The same defect exists twice in this tree, on purpose

`../pymalloc-repros/19_gh-153539_textio_tell_reentrant_decoder` is this defect as a C model consumer: a sequence against the real pymalloc with a
labelled probe, so a fault can be attributed to one instruction. This entry is the same defect
reached in place: the body of upstream's regression test CTextIOWrapperTest.test_reentrant_seek_during_tell (Lib/test/test_io.py at the fix) on the C io classes, with addCleanup and asserts written as plain code.

Neither supersedes the other. The C model is stronger for attribution; this one is stronger for
liveness, because the defect is reached in the whole interpreter. They are not two defects and must
not be counted twice.

## Liveness at the pin, measured

measured 2026-10-29 on the pinned 3.13.7 ASan build (--with-address-sanitizer, gcc 15.2, p13) with PYTHONMALLOC=malloc and ASAN_OPTIONS=detect_leaks=0: ASAN:heap-use-after-free in a 41-byte region. Running the trigger is the proof. Case 28's trigger reported its recorded 328-byte heap-use-after-free in the same build.

## Which allocator owns the object

**nested.** `allocator_layer` = `L1-pymalloc-pool`, `allocator_consumed` = `yes`.
measured 2026-10-29: ASan reported a 41-byte region, at or under the 512 B SMALL_REQUEST_THRESHOLD, and the heap report does not survive PYTHONMALLOC=pymalloc on a second 3.13.7 build with ASan and pymalloc both on (CFLAGS/LDFLAGS -fsanitize=address, because configure's --with-address-sanitizer forces pymalloc off): under PYTHONMALLOC=pymalloc the run completes, tell() returns, and there is no report: the stale read lands inside a pymalloc block

## Negative control

`negative_control.py` keeps the trigger's traffic and makes the offending access valid: the decoder's getstate() does not re-enter seek(); the same seek(0) runs after tell() returns.
It ran to its end with no report on the same ASan build, and prints `NEGATIVE-CONTROL no defect performed`.
