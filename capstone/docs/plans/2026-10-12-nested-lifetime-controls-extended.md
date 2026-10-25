# Parent, sibling and successor controls at three more nested allocators: pre-registration, 2026-10-12

Committed and pushed before any harness is built or run. A result that differs from what is
written here is a finding, recorded as such; this file is not edited afterwards.

## The question

`2026-10-11-nested-lifetime-controls.md` (53325bba) ran the three controls at one nested allocator
per application. Its results name three allocators whose lifetime patch half it did not exercise:
SQLite's lookaside (`sqlite3_malloc` never uses it), Wireshark's wmem `block_fast` (its free is a
no-op, so its only release is the reset) and memcached's `cache.c` objects. This run applies the
same three controls, with the same definitions and the same stock-against-protected design, to
those three allocators. Everything in the earlier plan's "The question" and "Platform and builds"
sections applies unchanged, except the branch: images are built from a snapshot of
`ports/nested-controls-extended`, based on 88d0960bf9ae.

## Allocators and the release each control uses

| Program | Nested allocator, release operation | Parent reference used in control 1 | Corpus the cases join |
|---|---|---|---|
| SQLite 3.22.0 | lookaside slots (`sqlite3DbMallocRaw` on a connection with lookaside enabled, released by `sqlite3DbFree`) | the connection's lookaside buffer, through the reference the lookaside allocator keeps | `bug-corpora/sqlite/capi-repros` |
| Wireshark 4.6.8 | wmem `block_fast` (`wmem_alloc` on a `WMEM_ALLOCATOR_BLOCK_FAST` allocator, released by `wmem_free_all`) | the block_fast block (chunk) reference the allocator keeps | `bug-corpora/wireshark/wmem-repros` |
| memcached 1.6.45 | `cache.c` objects (`cache_alloc`, released by `cache_free`) | the reference to the object's storage that the cache keeps and from which patch 0006 derives the child | `bug-corpora/memcached/allocator-repros` |

Where the lookaside or the cache hands out an object from its backing allocator instead of its
own storage (lookaside full or disabled, cache list empty), the control arranges that A comes from
the nested allocator and records how this was checked.

**block_fast has no single-object release.** `wmem_free` on it does nothing and `wmem_free_all`
releases every child at once. Control 2 (sibling) therefore has no meaning there and is reported
as not applicable, not as a pass. Controls 1 and 3 and the stale access use `wmem_free_all` as
the release.

## Predictions

On the protected build, for each of the three allocators:

- control 1 (parent): completes, the read returns what the allocator itself left there;
- control 2 (sibling; lookaside and cache.c only): completes, B's contents intact;
- control 3 (successor): C's start address equals A's; access through C completes;
- stale access through A: rejected with cause 25 (revoked lifetime) at the labelled probe.

On the stock build: the same controls complete, and the stale access through A completes and
reads C's data.

Each control runs in at least two separate boots per build, as in the earlier run. An allocator
whose controls cannot be driven by 10:00 local on 2026-10-12 is reported as not measured, with the
reason. It does not count as a pass.
