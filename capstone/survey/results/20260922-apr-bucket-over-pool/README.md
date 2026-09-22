# The nested boundary is mostly invisible, and measurably not always

`apr-bucket` on apr-util 1.6.3, the revision
`capstone/ports/apr` pins for this allocator. Its level 0 is the APR pool
rather than the system allocator, which makes it one of the two genuinely
nested levels in the set and the only one whose answer is not 100 per cent.

## The two levels of one program, static rung, repetition 1

| level | level 0 | allocations | reuses | before backing release | distinct |
|---|---|---:|---:|---:|---:|
| `pool` | libc | 3,249,222 | 3,241,109 | **100.0 %** | 8,113 |
| `apr_bucket` | **apr_pool** | 440,022 | 440,009 | **90.9 %** | **13** |

Forty thousand of 440,009 bucket-node reuses happen **after** the block
holding them has gone back to the pool. Every other level recorded in this
campaign reads essentially 100 per cent, which fairly invites the suspicion
that the question cannot come out any other way. Here it does, on the one
level the question was invented for, and the instrument is therefore
answering rather than agreeing.

Thirteen addresses carry 440,022 nodes. The pool level in the same run
reads 100.0 per cent over 3.2 million allocations and 8,113 addresses, so
the two allocators of one program disagree about the quantity the survey
measures. Identical on all three repetitions.

## Two things the wiring had to get right first

**A block is keyed on its data start.** A memnode's `first_avail` has moved
by the time the block is released, so the record and the forget could not
have named the same address. Both can compute
`(char *)node + APR_MEMNODE_T_SIZE`, and `apr_pools.c:418` sets exactly
that on every handout, so the stamp covers the nodes carved from the block
and the forget finds it again.

**`list->blocks` is a chain.** The bucket allocator links new blocks
through `next`, and one `apr_allocator_free` releases all of them.
Forgetting only the head would have left the rest stamped live and counted
later reuses as preceding a release that had already happened, which would
have pushed this row's 90.9 per cent up towards the 100 the other rows
show. The two chain sites walk `next`. The large-node path deliberately
does not, because that memnode's `next` belongs to the allocator and not to
this list, and walking it there would forget blocks that are still live.

## Why a completed arm is itself evidence

The wiring uses the recorded shape of `a1hook.h` rather than an interposed
malloc, because what sits under this level is the pool. The
do-it-yourself entry points abort instead of forwarding, so if the
instrument had been placed where the system allocator serves it, the run
would have died at the first call. An arm that completes says level 0
really was the pool.

## Provenance

| | |
|---|---|
| apr-util | 1.6.3, the revision the port pins for this allocator |
| APR | 1.7.6 here, against the port's 1.7.4 |
| workload | `ab`, static rung, 20,000 requests, three repetitions |
| arms | `shipped` as the oracle, `hooked` for the pool level, `buckets` for this one |
| wiring | `aprhook-buckets.inc`, level `apr_bucket` over `apr_pool` |
| companion pass | `experiments/a1/results/apache/20260922T031110Z` |
| compiler | gcc 13.3.0 |
| raw | `raw/`, hashed in `raw/SHA256SUMS` |

Every arm served 20,000 complete requests and the shipped arm's report does
not differ.

## Limits

One rung. APR itself is 1.7.6 here rather than the port's 1.7.4, which is a
revision difference in the level below this one; apr-util, which owns the
allocator this row measures, is the port's revision.
