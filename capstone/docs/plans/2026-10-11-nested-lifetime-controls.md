# Parent, sibling and successor controls at the nested allocators: pre-registration, 2026-10-11

Committed and pushed before any harness is built or run. A result that differs from what is written
here is a finding, recorded as such; this file is not edited afterwards.

## The question

The security table shows that each nested allocator's lifetime patch rejects stale accesses. It
does not show that the patch leaves permitted accesses alone. Three controls answer that, in the
real code of each ported nested allocator, driven through the allocator's own API (not an ISA test):

1. **Parent stays usable.** Allocate child A, release A through the allocator, then read and write
   A's former range through the allocator's own reference to the containing block, pool, page or
   arena (the reference the allocator itself uses to manage that storage). Must succeed.
2. **Sibling stays usable.** Allocate A and B from the same block, release A, then read and write B
   through B's reference. Must succeed, and B's contents must be intact.
3. **Successor stays usable, the stale reference does not.** Release A, allocate C until the
   allocator hands out A's start address again, and show that it did (C's address equals A's). Then
   read and write through C: must succeed. Then, in a separate run, read through the old reference
   to A: must be rejected.

The stale access is also run on the **release build** (no lifetime patch) as the control for the
control: there it must complete, reading C's data, so that the rejection on the protected build is
attributable to the patch.

## Applications and the allocator each control drives

| Program | Nested allocator, release operation | Parent reference used in control 1 | Corpus the cases join |
|---|---|---|---|
| SQLite 3.22.0 | memsys5 (`sqlite3_malloc`/`sqlite3_free` over `SQLITE_CONFIG_HEAP`) | memsys5's pool (`mem5.zPool`) | `bug-corpora/sqlite/capi-repros` |
| PostgreSQL 17.5 | memory-context chunks (`palloc`/`pfree`) | the chunk's block, through the context | `bug-corpora/postgres/c-repros` or `mmgr-repros` |
| CPython 3.13.7 | pymalloc (`PyObject_Malloc`/`PyObject_Free`) | the block's pool (`POOL_ADDR`) | `bug-corpora/cpython/pymalloc-repros` |
| mruby 4.0.0-rc2 | GC object slots (`mrb_obj_alloc`, released by the sweep) | the slot's heap page | `bug-corpora/mruby/gc-slot-repros` |
| Perl 5.36.3 | SV heads (`newSV`, released by `SvREFCNT_dec`/`plant_SV`) | the head's SV arena | `bug-corpora/perl/release-differential` |
| FFmpeg 9.0.1 | `AVBufferPool` entries (`av_buffer_pool_get`, release by `av_buffer_unref`) | the pool entry's own data reference | `bug-corpora/ffmpeg/pool-repros` |
| Wireshark 4.6.8 | wmem `block` allocator (`wmem_alloc`/`wmem_free`) | the wmem block (chunk header) | `bug-corpora/wireshark/wmem-repros` |
| memcached 1.6.45 | slab chunks (`slabs_alloc`/`slabs_free`) | the slab page | `bug-corpora/memcached/allocator-repros` |

Where an allocator offers several release operations, the control uses the one its lifetime patch
revokes at, and says which.

## Platform and builds

Both arms on the virtual profile of `2026-10-11-protected-same-answers.md`: QEMU af37cc32, launcher
3975314a, module a1c6cb6b, exact bounds, one hart. Protected = the application's lifetime patch as
in the security table; release = the same source without it. Images are built from a snapshot of
this branch (`ports/nested-lifetime-controls`, based on 382231d7cd9d, which carries CPython's patch
0014 fix and Perl's patch 0011). Each run records: the llvm-capstone revision, image sha256,
configuration, the source location of every access (file:line of the probe), the addresses of A, B
and C, and the observed outcome (completed / fault with cause and pc).

## Predictions

On the protected build, for every application:

- control 1 (parent): completes, the read returns what the allocator itself left there;
- control 2 (sibling): completes, B's contents intact;
- control 3 (successor): C's start address equals A's; access through C completes;
- stale access through A: rejected with cause 25 (revoked lifetime) at the labelled probe.

On the release build: the same three controls complete, and the stale access through A completes
and reads C's data.

An application whose allocator cannot be driven this way tonight is reported as not measured,
with the reason. It does not count as a pass.
