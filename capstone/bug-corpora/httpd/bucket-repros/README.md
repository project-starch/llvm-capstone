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
* **Three protection arms are measured: `spatial`, `sublet`, `cheribsd-revocation`.**
  The PoisonCap arms carry `"status": "not written"`: there is no PoisonCap
  build of APR, and the gap is visible rather than absent.
* **`native-detect`** is not merely unwritten but tautological: neither level
  reaches `malloc`, so ASan has no event. Valgrind against APR's own annotations
  is the arm that could discriminate.
* **`case.c` uses `APRB_CASE`**, this corpus's allocator seam, where the
  contract's checker looks for `PYC_CASE`. The rule it means — a case declares
  the number its directory carries, and the driver refuses a fixture that names
  another — is implemented.

## Running

The port is `ports/apr/pools` with `-DAPRP_BUCKETS=ON`, which carries
apr-util's bucket allocator on top of the pool allocator it is a client of.
`shared/build-cases.sh` invokes the port's one-source seam once per case:

    shared/build-cases.sh native <out>            then runners/run-native.sh
    shared/build-cases.sh capstone-domain <out>   then runners/capstone-domain/
    shared/build-cases.sh cheribsd <out>          then runners/cheribsd/

The native arms are the fix differential, one program per case run twice,
control first; an infrastructure failure exits 75 with no verdict. The
[domain runner's manual](runners/capstone-domain/README.md) and the
[CheriBSD manual](runners/cheribsd/README.md) say what each arm must do and
how the negative control must make every oracle say FAIL before a PASS is
believed.

## What the three systems see, measured 2026-09-22

| arm | what acts | result |
|---|---|---|
| `spatial` | bounds and tags; a pool node and a bucket piece keep their alias across the free lists | 8 / 8 complete |
| `sublet` | the pool port's release of a node, reached directly or through the bucket allocator's lend | **8 / 8 fault** at `apr_defect_read`, cause 24 |
| `cheribsd-revocation` | libc's quarantine and revoker, on, verified in the guest | 8 / 8 complete; the control beside them faults |

Which event each case ends at, and therefore which mechanism catches it:

| cases | the lifetime ends at | under Sublet |
|---|---|---|
| 2, 3, 5, 7 | a pool destroy: the node is released and reissued to the next pool | the pool port's release revokes the node; the holder's alias dies with it |
| 0, 1, 6 | a bucket allocator's destroy: its blocks go back to APR and are reissued | the same release, reached through the block the pool port lent; every piece dies with its block |
| 4 | the connection handback -- **reuse without free**, taxonomy class 3: the lender hands its connection and allocator to the next request and nothing is freed | the reduced consumer declares the handback as the allocator's epoch, the lender's operation under the discipline; the blocks go back and are reissued, and the old brigade's alias dies with them |

Case 4 is the corpus's one **reuse-not-free** case, and the one that
separates the systems by kind rather than by degree. Upstream's handback
(`ap_proxy_release_connection` → `connection_cleanup`) puts the backend
connection on the reslist and leaves its bucket allocator as it is; the rule
that nothing of the old request may outlive the handback is a copying
discipline (`ap_proxy_buckets_lifetime_transform`, then cleanup), not an
allocator event, and the defect (`c81adad105`) is that cleanup coming after
the release. No free happens, so no free-triggered mechanism -- libc's
revoker, PoisonCap, ASan -- has anything to act on, by construction. Under
the Sublet discipline a lender that reuses without freeing revokes at the
point of reuse, and the reduced consumer, which models the lender, does:
it ends the allocator's tenancy at the handback (destroy and recreate, the
operation `connection_cleanup` would perform), in both arms. That is the one
step the reduced sequence takes that upstream does not, its PROVENANCE.md
says so, and it is what the class-3 row of the taxonomy means by the
Capstone column: the boundary is enforceable once it is declared.

The bucket allocator's own recycling -- `apr_bucket_free` filing a small node
on the freelist and `apr_bucket_alloc` reissuing it -- is exercised by no
case here: the eight upstream defects all end at a pool or an allocator
destroy. It is the port's fixture suite instead
(`ports/apr/pools/security-tests/qemu/run-buckets.py`, seven fixtures, both
modes), which is where a stale read through a filed node, a stale write into
its next holder and a double free are shown to fault at the labelled site.

Records: `results/20260922-qemu/`, its negative control beside it, and
`results/20260922-cheribsd/`.

## What CheriBSD's revoker sees

Measured twice, and the answer is the same. On 2026-09-21 the eight cases ran
against the census's freestanding build with `free()` interposed and counted:
all sixteen arms completed and printed their native verdict, every arm with
`freed_to_malloc=0`. On 2026-09-22 they ran through the port's CheriBSD build,
the same allocators under the platform's `malloc`, and all eight completed
again with the run's revocation control faulting at `apr_defect_read` in the
same guest. The ABI probe prints `CHERI_ABI pointer_bytes=16
runtime_revocation=1`, read from CheriBSD's own `malloc_revoke_enabled()`;
the guest default is preserved.

Storage that never reaches `malloc` never enters the quarantine the revoker
sweeps. This is the informative negative for the two-level shape -- the
mechanism exists and is switched on, one level below where the lifetime
ends.

## Limits

These are reductions. Real is the pair of allocators, byte for byte, with the
bucket node geometry transcribed verbatim and checked against upstream's header
on every build. Reduced is everything else — brigades, filters, connections,
requests — to the holder and the storage. No case has been shown to fire in a
running httpd, and **this corpus has no CVE**: the two that candidates cited,
CVE-2010-1623 and CVE-2011-3192, are both denial of service by memory
consumption and were rejected for that reason.
