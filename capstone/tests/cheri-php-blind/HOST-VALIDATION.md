# Host ASAN validation of the allocator-pattern rows

Run `./host-validate.sh` (needs only gcc + libasan; no CHERI vehicle).

The rows claim to model the corpus's 31 ASAN-blind defects. That claim is
testable with the detector we already understand, and this is the result:

| row | stockalloc (real PHP) | patched (corpus patches) | |
|---|---|---|---|
| a1_uaf_read | OK | ASAN:heap-use-after-free | BLIND->CAUGHT |
| a2_uaf_write | OK | ASAN:heap-use-after-free | BLIND->CAUGHT |
| a3_uaf_recycled | OK | ASAN:heap-use-after-free | BLIND->CAUGHT |
| a4_uaf_zval_member | OK | ASAN:heap-use-after-free | BLIND->CAUGHT |
| a5_double_efree | ASAN:heap-use-after-free | ASAN:heap-use-after-free | control |
| a6_uaf_after_shutdown | ASAN:heap-use-after-free | ASAN:heap-use-after-free | control |
| b1_oob_read_slack | OK | ASAN:heap-buffer-overflow | BLIND->CAUGHT |
| b2_oob_write_slack | OK | ASAN:heap-buffer-overflow | BLIND->CAUGHT |
| b3_oob_read_past_block | ASAN:heap-buffer-overflow | ASAN:heap-buffer-overflow | control |
| b4_header_underflow | OK | OK | supplementary |
| c1_stack_after_return | ASAN:stack-use-after-return | ASAN:stack-use-after-return | control |
| z0_sanity | OK | OK | sanity |

**Six rows reproduce the ASAN-blind flip** -- four temporal (cache-hidden) and
two spatial (rounding-hidden) -- which is the property that defines the 31.
**Three are positive controls** that ASAN catches whatever the allocator does.

`a5` is worth noting: in the *stock* arm ASAN reports a use-after-free even
though neither `efree` called `free()`. That is faithful, not an artefact --
`shutdown_memory_manager` drains the cache and frees the block twice, and the
second loop iteration then reads `ptr->pNext` out of freed memory. It is exactly
where the pristine PHP 5.0.0 build crashes (`Zend/zend_alloc.c:490`,
`REMOVE_POINTER_FROM_LIST`), confirmed under gdb against a native build.

`b4` is invisible to ASAN in both arms, because the underflow lands in the
`zend_mem_header` inside the malloc'd block. It is reported as supplementary,
not as one of the 31.

`c1_stack_after_return` is caught in both arms, correctly: it never touches
`emalloc`, so the allocator variable does not apply to it. It is in the set
because CWE-562 is one of the 31's classes (CRASH-100) and because it is the one
place CHERI is expected to do *worse* than ASAN -- a dangling stack capability
stays tagged and in bounds, and CheriBSD does not revoke stack capabilities on
return.

## Three row-design errors this validation caught

1. **`b1`/`b2` originally derived the over-read offset from the live
   `REAL_SIZE()`.** Under `-DPHP_PATCHED_ALLOC` that is the exact object size,
   so the loop `for (i = n; i < rs; i++)` ran zero times: the treatment arm
   performed no over-read at all and could not be caught. Both arms now take the
   offset from `STOCK_REAL_SIZE`, so the access is identical and only the
   allocator varies.
2. **`ROW_MAIN` was not variadic**, so a body containing a top-level comma
   (`size_t n = 26, rs = ...`) silently failed to compile -- macro arguments
   respect parentheses, not braces.
3. **`c1` originally obtained the dangling pointer with `return buf;`.** gcc
   hardens a returned local by substituting NULL, so the row was a null-deref,
   not a stack use-after-return: ASAN reported `SEGV on unknown address
   0x000000000000`. CHERI *does* trap a null dereference, on the tag, so the row
   would have credited CHERI with catching a class it cannot catch. The address
   is now smuggled out through a `volatile` global, and the row asserts
   `DANGLING-NONNULL=1` and reads back the clobber byte (`0x30`) rather than its
   own `'S'` (`0x53`) -- i.e. it proves the frame was actually reused.

The host classifier's ASAN kind list is now the same list, in the same order, as
the corpus's own `scripts/run-corpus.py:33`; before that it lacked
`stack-use-after-return` and reported a correct detection as a bare `ASAN`.

Both were found only because every row has a positive control and the table is
read for the expected flip rather than for an absence of errors.
