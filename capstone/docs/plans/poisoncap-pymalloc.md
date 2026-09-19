# PoisonCap pymalloc integration

Work branch: `ports/10-poisoncap-pymalloc`, based on the verified FFmpeg
PoisonCap integration and current `dev`. Reuse the published compiler, QEMU,
CheriBSD image and platform controls; keep platform sources outside Git.

## Lifetime policy

Use the existing CPython 3.13.7 extraction and block-lifetime hooks. Preserve
pymalloc's size-class selection, free-list order, pool reassignment and arena
retention. Keep allocator authority in separate records and strip poison and
VM authority from issued bounded client pointers. A tagged, exact current
client capability is required for free/realloc; address lookup alone is not
an authorization check.

Free poisons the whole block, completes a synchronous kernel sweep, clears
poison and zeros the stored poison capabilities, then returns manager authority to pymalloc for its in-band free-list
write. This avoids moving the free list or silently allowing stale pointers
to access its metadata. Reclassification and empty-arena release invalidate
the corresponding backing storage. This is a trusted allocator integration;
it does not establish isolation of hostile nested managers.

In-place realloc preserves the old block contents in a capability-preserving
metadata snapshot before invalidation and restores them before issuing new
client authority. Snapshot allocation failure returns NULL before invalidating
the old lease. Moved realloc preserves the previous requested length and only
releases the original allocation after successful allocation and copying.
Raw requests use a separate bounded reusable backing region, as in the Sublet
extraction, with alignment sufficient for CHERI compressed bounds.

## Acceptance gates

1. Build native, Capstone domain, standard CheriBSD and PoisonCap variants.
2. Preserve native extraction/oracle tests and default recorder behavior.
3. Run purecap ABI/bounds and published PoisonCap live/poison/reuse controls.
4. Pair spatial/protected cases for free, reuse, stale free, in-place realloc,
   bounds, raw fallback, pool reclassification and repeated address reuse.
5. Check capability-bearing moved/in-place realloc, failure preservation,
   all small size classes, large sizes, unwritten storage reuse across an
   unrelated sweep, and real empty-arena release.
6. Replay the same complete native recording in both modes, comparing logical
   operations, completion and event checksum. Account for ABI-dependent
   arena and metadata differences separately.

The first recording uses the existing standard-library driver with one round
and one JSON item. It is a complete captured workload, not a truncated prefix
of the larger trace. Record workload parameters, source/binary/platform hashes,
failed attempts, sweep/poison/clear/zero/copy counts and the selected reuse-round count.
The initial synchronous policy is deliberately conservative. Its sweep count
is not a lower bound for PoisonCap and QEMU elapsed time is not hardware cost.
A longer workload and a different sweep/quarantine policy are separate steps.
