# SQLite normalized memory campaign

This protocol is fixed before the new campaign. Earlier capacity-selected runs
are exploratory and are not pooled with it. No timing or security claim follows
from this campaign.

Primary pairs: Capstone original-layout memsys5 / Sublet memsys5; CheriBSD
original-layout memsys5 / PoisonCap memsys5 with corrected full-queue revocation.
The original CheriBSD control retains the fork's application ABI fixes and bounds
allocation returns, but uses the upstream inline free-list layout. The published
adapter-spatial layout is a diagnostic, not the original-layout denominator.

Use SQLite 3.22.0's complete speedtest1 main workload, deterministic phase oracles,
the same explicit SQLite feature options and -O0, lookaside disabled, 64-byte
atoms, and an 8 MiB configured original heap (129055 allocatable atoms). Sublet
gets that same allocatable pool and separately charged tables. Platform port
patches, compiler/ABI and VFS remain different and must be recorded.

One fresh process runs one warmup and 16 measured size-1 units. Each unit opens
and closes a fresh database; allocator and quarantine state survive between units.
Run three fresh-process repetitions per arm. Qualification precedes repetition;
failed qualification blocks a confirmatory four-arm comparison. A second profile
is four size-1 units, one size-4 burst, eight size-1 units. Report every failure;
do not substitute a smaller burst silently.

Preselected plots:

1. Peak selected allocator footprint H, protected/original within each platform,
   with absolute live/quarantine/metadata bytes alongside. H uses simultaneously
   held rounded block spans plus dedicated allocator metadata. This is neither
   requested live bytes nor RSS and excludes kernel capability-node storage.
2. H at unit completion and within-unit peak across repeated work, showing warmup,
   all repetitions and failures. This distinguishes retention from live work.
3. Cumulative distinct arena bytes covered by allocations and each unit's distinct
   coverage, normalized by pool size. This is address footprint, not resident
   working set. Allocation-start reuse is descriptive, not a retirement CDF.
4. Post-burst H recovery, only if all arms complete the fixed burst profile.

Instrument allocator events, not periodic wall-clock samples. Bitsets use block
indices and hold no capabilities. Report observer storage separately. Fail if
an index is outside the declared maximum or allocator ledger conservation fails.
Retain raw output, actual compiler argv, input/binary hashes, guest identity,
return status and all 32 phase oracles per completed unit. Timing output from
speedtest1 is ignored. Plots must identify incomplete or noncomparable arms.

These experiments can refute the proposed advantage: Sublet's per-atom tables
may dominate H even when it reuses addresses sooner. No implementation is declared
superior solely from a lower quarantine or smaller address footprint.
