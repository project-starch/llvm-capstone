# PoisonCap reference policy

The primary comparison follows the published allocator policy. Do not add
application-specific early sweeps to improve address reuse, or tune quarantine
thresholds after seeing the memory curves. These choices change the quantity
being compared. Keep previous adapter results as explicitly named controls.

## Published evidence

[PoisonCap v1](https://arxiv.org/html/2605.13210v1), sections 2.1, 4.6 and 5.5,
describes quarantine followed by revocation, and uses SQLite MEMSYS5 as its
nested-allocator implementation. It says that SQLite uses the Cornucopia
quarantine threshold. It does not publish an FFmpeg or mruby allocator port.
Its SPEC evaluation uses libc allocation and the shadow-bitmap revoker;
that configuration does not define another application's nested policy.

The reference artifact is
[`731c5720d1a6321d436a8d7b85bf3aa2e4971fe0`](https://github.com/Yuecheng-CAM/Ajs38RO4Vz-p/tree/731c5720d1a6321d436a8d7b85bf3aa2e4971fe0).
Its archive SHA-256 is
`c5e8b07d4f70ed8cc1ecf04193ceacc5266553a2cbd233e6fa51fb8f0d190b9c`.
The values below were checked against files read directly from that archive,
not the modified platform build directory.

| Boundary | Published trigger | Source |
|---|---|---|
| SQLite MEMSYS5 | On free, after insertion and poisoning: held bytes at least 16 MiB **and** quarantined bytes at least one quarter of held bytes | `sqlite/src/mem5.c:574–622` |
| SQLite queue capacity | Before inserting another entry into an already full 4,096-entry queue, drain the existing queue | `sqlite/src/mem5.c:467–597` |
| Outer libc allocator | Minimum allocated heap 8 MiB, default fraction 1/4; asynchronous revocation enabled, every-free override disabled | `cheribsd/lib/libc/stdlib/malloc/mrs/mrs.c:107–112,344–360,800–825` |

For MEMSYS5, held means rounded live allocation spans plus quarantined spans,
not the reserved arena size, cumulative allocation volume, or process RSS.
The percentage test has a **minimum held-size gate**, not a 16 MiB quarantine
limit. The queue-capacity path is independent of that gate. The unused
`QUARANTINE_EPOCH_DELAY` define does not implement an additional epoch delay.

The audited MEMSYS5 file SHA-256 is
`d8cb4c0e688350992c480b7956b5b6d47496700447ded6386e4c02e12d338df8`;
the libc `mrs.c` SHA-256 is
`650cd89e7ed97aeda044092b59feb40df2e9eb8aa21b6833a73fce5c9a2f56f1`.
Build-time defines and runtime overrides must also be recorded; a source
default alone does not establish the effective policy of a binary.

## Runtime correctness repairs

The reconstructed runtime already contains a local libc fix that clears
retired poison before handing an allocation to its new owner. That patch
does not change quarantine thresholds. Preserve it in the source/build audit;
the working runtime is not an unchanged copy of the artifact.

Enabling outer revocation exposed a separate libc ABI defect in FFmpeg:
`cheri_poison_set_version` declared its destination register as an input,
using an uninitialized C variable. In the shipped libc this overwrites `a0`
after `posix_memalign` has placed its successful zero return there. FFmpeg
therefore reports an allocation failure despite receiving a valid allocation.
The [one-line output-constraint correction](patches/cheribsd-poison-version-output.patch)
changes that operand to `"=C"`. With the same toolchain and libc sources,
the rebuilt `.text` has the same length and differs in six bytes. A
12-allocation alignment/size check fails on the old libc and passes on the
corrected libc with revocation enabled. The FFmpeg spatial Xvid diagnostic
then completes. This is an ABI correction, not a faster reclaim policy.

The longer resize input exposed incomplete retirement of old poison in libc.
The existing fix clears out-of-band poison metadata but leaves stored poison
capabilities in the payload. The kernel's `fupoison` examines their tag and
poison bit, so a later sweep can revoke the new owner's still-live pointer
before that owner initializes its buffer. The
[payload-retirement correction](patches/cheribsd-retire-poison-payload.patch)
also zeros the region before handing it out. Apply it after the preserved
[metadata-retirement patch](patches/cheribsd-retire-poison-metadata.patch).
The [allocation/reuse check](patches/poisoncap-retirement-check.c) aborts with
the metadata-only library; with both retirements it completes and retains all
64 live allocation tags across an explicit sweep. The complete 150-frame
FFmpeg resize diagnostic then matches its reference output. This adds
initialization work but no backing allocation or early quarantine drain;
these experiments do not compare timings or physical resident working sets.

SQLite's published nested `clear_region` also overwrites payload without an
explicit poison-metadata retirement. With outer defaults on, its old binary
faults during index creation. The
[nested retirement correction](patches/sqlite-poison-retirement.patch)
clears metadata before the existing initialization, preserving allocation
sizes, free-list geometry and thresholds. The preparer includes this repair
for future builds. Six new complete 17-unit processes now pass with outer
defaults enabled, paired with the unchanged archived Capstone controls.

Use the same corrected libc in both FFmpeg arms and record its loaded path
and hash. The new library is staged per campaign; do not replace a base disk
while another guest is running. The separate mruby campaign retains its
recorded libc in both arms. Neither a runtime correctness repair nor a
transferred port may be described as an unchanged-artifact measurement.

The mruby transfer also derives each poison operand with exact `RVALUE`
bounds, following MEMSYS5's bounded poison operand. The old adapter used
the wider page-manager pointer. Record this correction alongside the trigger
change; differences from the historical adapter are not a policy-only A/B
experiment. The new within-platform spatial/temporal pair uses the same
prepared source, binary, page geometry and observer.

## Published implementation versus corrected comparator

The artifact's full-queue path calls `quarantine_flush()` **without**
`quarantine_revoke()`. It also ignores the revoker's return value. Preserve
that exact implementation as a published-policy reproduction control.
The protected comparator keeps the same numeric thresholds but revokes
before a full-queue drain and stops on revoker failure. Call it
**PoisonCap, published thresholds with full-queue correction**. It is not
an unchanged-artifact result and must not silently replace that control.

The existing [SQLite policy-path measurement](results/sqlite-322-memory-20260927/README.md)
already distinguishes them: the published path completed six full-queue
drains with zero explicit revocations; the corrected path completed six
drains with six revocations. The later normalized SQLite results use the
corrected path. Their 8 MiB arena cannot satisfy the 16 MiB held-size gate:
the 4,096-entry capacity trigger, rather than the percentage trigger,
drives their explicit revocation. This matters to memory interpretation.

## Scope of existing application figures

| Application | Measured inner policy | Outer policy | Default-comparison status |
|---|---|---|---|
| SQLite | Published thresholds plus the documented full-queue correction | libc revocation disabled in normalized memory/reuse campaigns | Corrected nested reference with isolated outer allocator; not an unchanged-artifact reproduction |
| mruby | Reclaim poisoned slots when no reusable GC heap slot remains | libc revocation enabled | Our pressure-triggered adapter; not the published nested policy |
| FFmpeg | Sweep before reissuing an unswept poisoned block; selective RefStruct snapshots | libc revocation disabled | Our eager-reuse adapter; not the published nested policy |

Successful application oracles validate execution, not equivalence of these
policies. The measured FFmpeg equality in release-to-reissue gaps applies to
its eager adapter only. Snapshot bytes are adaptation costs, not intrinsic
PoisonCap costs. Do not infer a common-default ranking from the combined plot.

## Contract for replacement measurements

Use the published MEMSYS5 thresholds, with the separately disclosed
full-queue correction, as the reference for newly implemented nested ports.
Label them **published SQLite policy transferred to this allocator**; there
is no author-provided FFmpeg/mruby default to reproduce. Do not call free-slot
exhaustion or a requested pool entry a threshold event.

Quarantined entries stay unavailable until a qualifying sweep. In FFmpeg,
pool selection must skip those entries and allocate other backing. Merely
changing the revoker call while allowing the same slot to escape is invalid.
Persistent RefStruct state and destructor callbacks need a correct lifetime
path; any forced teardown/pressure sweep must be counted and disclosed as a
port-specific extension, not hidden in the default-policy curve.

Preserve the published outer allocator defaults in both CheriBSD arms.
Record effective sysctls, allocator environment and libc/build hashes. A
diagnostic with outer revocation disabled remains separately named; do not
promote it as default if enabling that default exposes a platform fault.

Provision equal capacities within each comparison pair before measuring.
Do not shorten the published quarantine to fit the old 4 MiB FFmpeg arena
or 2,048-record table. Record reservation and actually occupied backing
separately. Count live rounded spans, quarantined spans, reusable backing,
metadata, and sweeps by cause (percentage, queue-full, pressure, teardown).
Fresh runs must pass complete application-output oracles and account for
the actual policy path. Extra allocations induced by quarantine are an
outcome to report, not a reason to force all arms to have equal allocation
counts. Keep old raw results and record new campaigns under new identities.

The [replacement mruby/FFmpeg campaign](results/published-policy-20260928/README.md)
now validates 48 complete processes and plots the new policies separately.
The historical three-application figures retain their original data and
explicit custom-policy/outer-disabled labels. SQLite's repaired protected
17-unit qualification and all six repeated outer-default processes pass.
Their inner ledgers and reuse histograms reproduce the historical isolated-outer
controls exactly; the fresh raw evidence records the enabled outer policy.
