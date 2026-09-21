# httpd bucket-allocator defect corpus

Consumer-side lifetime defects on storage from Apache's **two stacked**
allocators. `apr_bucket_free` pushes a small node onto the bucket allocator's
own LIFO freelist and stops; a large one goes to `apr_allocator_free`, APR's
size-bucketed list, whose default configuration frees nothing. Neither level
reaches `malloc`, so a stale bucket pointer sits behind two layers a
malloc-level tool cannot see.

    00_1c7a70c9d9_wrong_bucket_alloc_across_connections/  the wrong connection's allocator
    01_d2a1cf5f8c_buckets_outlive_backend_allocator/      not flushed before it went
    02_106d0761c0_brigade_on_request_pool_past_eor/       holder on the pool that dies first
    03_d9c2352952_buckets_from_pool_that_dies_first/      payload from the pool that dies first
    04_c81adad105_brigade_not_cleaned_before_reuse/       held across a connection handback
    05_60919177e8_read_from_destroyed_brigade/            read after the brigade was destroyed
    06_4930450013_bucket_outlives_its_brigade/            contents outlive the holder
    07_edc450c8ac_subrequest_pool_private_data/           private data on the shorter pool

| shape | cases |
|---|---|
| foreign or dead allocator | 0, 1 |
| a pool outlived by its holder or its contents | 2, 3, 6, 7 |
| a recycled allocator serving its next user | 4 |
| the holder used after it was destroyed | 5 |

Eight cases, four shapes. The triage that selected them from 118 candidates, and
the seven it rejected with reasons, are in
[`docs/ref/httpd-bucket-allocator-defects.md`](../../../docs/ref/httpd-bucket-allocator-defects.md).

## What is distinctive here

The first two shapes have no counterpart in the other corpora. **What is wrong
is the identity of the allocator**, not a pointer into one: `apr_bucket_free`
reads `node->alloc` and pushes onto *that* list, so a node freed through the
wrong allocator lands on a freelist whose owner does not own the block — and the
next allocation from that list hands it out.

## The contract

The layout and the `case.json` fields are the corpus contract in
[`cpython/pymalloc-repros/SCHEMA.md`](../../cpython/pymalloc-repros/SCHEMA.md),
referenced rather than copied. Where this corpus differs:

* **`native-fix-differential`** replaces the protection axis: the pair differs
  by whether the upstream fix is applied.
* **Every protection arm is declared and not written.** The allocators build
  freestanding and this corpus runs against them, but no domain workload does,
  so `spatial`, `sublet` and both `poisoncap` arms carry `"status": "not
  written"` and the gap is visible rather than absent.
* **`native-detect`** is not merely unwritten but tautological: neither level
  reaches `malloc`, so ASan has no event. Valgrind against APR's own annotations
  is the arm that could discriminate.
* **`case.c` uses `APRB_CASE`**, this corpus's allocator seam, where the
  contract's checker looks for `PYC_CASE`. The rule it means — a case declares
  the number its directory carries, and the driver refuses a fixture that names
  another — is implemented.

## Running

    bash ../../../ports/apr/build-buckets-census.sh   # builds the allocators
    bash runners/run-native.sh [outdir]

One program per case, each run twice, control first; an infrastructure failure
exits 75 with no verdict. Each case interposes `free()` and prints
`freed_to_malloc`, so "nothing reaches malloc" is measured per case rather than
inherited from the census.

## Limits

These are reductions. Real is the pair of allocators, byte for byte, with the
bucket node geometry transcribed verbatim and checked against upstream's header
on every build. Reduced is everything else — brigades, filters, connections,
requests — to the holder and the storage. No case has been shown to fire in a
running httpd, and **this corpus has no CVE**: the two that candidates cited,
CVE-2010-1623 and CVE-2011-3192, are both denial of service by memory
consumption and were rejected for that reason.
