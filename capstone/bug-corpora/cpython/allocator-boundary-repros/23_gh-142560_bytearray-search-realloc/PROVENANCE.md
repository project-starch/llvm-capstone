# Provenance

**Upstream fix:** `gh-142560`. **Consumer:** the freed block's pymalloc pool.
**CVE:** `NO VERIFIED CVE`.

## The same defect exists twice in this tree, on purpose

`../pymalloc-repros/10_gh-142560_bytearray_search_realloc` is this defect as a C model consumer: a sequence against
the real pymalloc with a labelled probe, so a fault can be attributed to one
instruction. This entry is the same defect as upstream's own reproducer through
the real interpreter.

Neither supersedes the other. The C model is stronger for attribution; this one
is stronger for liveness, because the defect is reached in place. They are not
two defects and must not be counted twice.

## Why this one had to exist

The C models cannot be built for the `cheribsd-revocation` arm. `corpus.h`
locates the faulting instruction with three inline-asm blocks, and their `"r"`
constraints cannot hold a 128-bit capability, so a purecap build stops with

    error: couldn't allocate input reg for constraint 'r'

The PoisonCap build works only because it selects purecap variants under
`#ifdef PYMALLOC_POISONCAP`; turning that off falls back to the non-purecap
assembly. Rather than change a shared header that twenty measured cases depend
on, this arm is measured through the interpreter.

## Liveness at the pin, measured

measured on the pinned 3.13.7 ASan build with PYTHONMALLOC=malloc and ASAN_OPTIONS=detect_leaks=0: ASAN:heap-use-after-free in a 2-byte region. Running the trigger is the proof.

## Which allocator owns the object

**nested.** `allocator_layer` = `L1-pymalloc-pool`, `allocator_consumed` = `yes`.
measured: ASan reported a 2-byte region, at or under the 512 B SMALL_REQUEST_THRESHOLD, and the heap report does not survive PYTHONMALLOC=pymalloc
