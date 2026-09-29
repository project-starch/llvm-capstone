# SQLite: normalized allocator memory and address footprint

SQLite 3.22.0's complete `speedtest1 main` workload is measured with an
original-layout allocator and its protected variant on each platform. The
[build manifest](builds.json) passes the [comparability gate](../../check-build-comparability.py):
explicit SQLite feature options, application/driver `-O0`, workload, atom count
and lookaside configuration match. The original-layout CheriBSD control retains
necessary CHERI application ABI fixes and bounds allocation returns; it does
**not** inherit PoisonCap's external links or quarantine. This is a different,
more appropriate denominator than the earlier adapter-spatial control.

The four arms are Capstone original, Capstone + Sublet, CheriBSD original,
and PoisonCap with the explicitly corrected full-queue revocation path.
PoisonCap's published full-queue path freed queued blocks without invoking
revocation. These results name the corrected implementation and preserve its
patch; they are not presented as an unmodified-artifact measurement.

## What the plots establish

- [Protection cost](figures/01-protection-cost.pdf): selected peak allocator
  footprint, protected/original within each platform, alongside absolute bytes.
  Lower is better. This includes the large Sublet tables and does not support
  an unconditional memory-overhead win for Sublet.
- [Repeated-work retention](figures/02-retained-memory.pdf): exact within-unit
  peak and the state after closing each fresh database. Sublet returns all
  allocated pool spans immediately; its tables remain reserved. PoisonCap
  retains a varying quarantine between units.
- [Address footprint](figures/03-address-footprint.pdf): cumulative and per-unit
  allocated arena coverage, plus protected/original expansion. Sublet follows
  the original allocator's address pattern; PoisonCap expands into more of the
  arena. Both plateau in this finite repeated workload. This is **not RSS,
  physical working set, cache occupancy, or a retirement-delay CDF**.
- [Burst diagnostic](figures/04-burst-diagnostic.pdf): four size-1 units, a
  size-4 burst, then eight size-1 units, without resetting the allocator.
  Failed bursts remain visible; the figure does not extrapolate their recovery.

The useful claim is about **address reuse and available pool capacity**, with
an explicit metadata tradeoff. It is not that Sublet always consumes less memory.
These observations concern one SQLite workload and cannot be generalized to
all application ports.

## Repeated-work result

All **12/12** repeated-work attempts complete: four arms × three processes ×
17 whole workload units, giving **6,528 matched phase oracles**. The three
repetitions give identical unit-level allocator observations in every arm.
Four separate one-unit qualifications also pass.

| Arm | Peak selected H | Peak H / original | Cumulative allocated address bytes | Address bytes / original |
|---|---:|---:|---:|---:|
| Capstone original | 1,404,831 | 1.00× | 1,280,960 | 1.00× |
| Sublet | 6,567,768 | **4.68×** | 1,280,960 | **1.00×** |
| CheriBSD original | 1,406,879 | 1.00× | 1,283,008 | 1.00× |
| PoisonCap corrected | 6,005,919 | **4.27×** | 5,113,792 | **3.99×** |

The ratios use each platform's own original allocator, not cross-OS absolute
values as denominators. H peaks exclude the warmup and use all 16 subsequent
units. Cumulative address coverage includes the warmup. Sublet uses about 75%
less arena address coverage than PoisonCap in this experiment, but its selected
H peak is about 9% larger. PoisonCap reaches its cumulative footprint plateau
by unit 1; this experiment does not show indefinitely increasing memory use.

After each database close, Sublet has zero live or quarantined pool bytes.
PoisonCap's post-close quarantine varies from 9,728 to 2,497,280 bytes across
measured units. Sublet's 5,292,184 selected metadata bytes remain charged even
when the entire allocatable pool is reusable. This counterexample is retained
in both the absolute and normalized footprint plots.

## Burst result

| Arm | Completed / attempted burst processes |
|---|---:|
| Capstone original | 3 / 3 |
| Sublet | 3 / 3 |
| CheriBSD original | 3 / 3 |
| PoisonCap corrected | 0 / 1; two repetitions blocked by failed qualification |

All successful processes complete the 13-unit profile and all phase oracles.
PoisonCap completes the four preceding size-1 units, then fails during phase
190 of the size-4 unit with SQLite `out of memory`. The last sample before
that phase has 5,241,600 quarantined bytes and 1,323,264 free bytes. This is an
application allocation failure, not the published kernel panic seen in earlier
experiments. No post-failure recovery curve is invented.

Sublet's peak allocated pool span grows from 1,275,584 to 4,195,008 bytes in
the burst and returns to 1,275,584 in the very next size-1 unit. Its selected
metadata remains charged throughout. This demonstrates successful reuse after
the burst at equal **usable pool capacity**, with the larger total Sublet
reservation already disclosed above. It does not demonstrate a smaller total
budget requirement.

## Method and accounting scope

The [preselected protocol](../../sqlite-memory-plan.md) defines one warmup and
16 measured complete workload units, with three fresh-process repetitions per
arm. Each unit opens and closes a new database while keeping allocator and
quarantine state. A burst's first attempt is its qualification; failure blocks
its remaining repetitions. There are no confidence intervals inferred from
three identical deterministic executions. All observations and repetitions
are retained in [unit-ledger.csv](unit-ledger.csv) and
[phase-ledger.csv](phase-ledger.csv).

Every arm has 129,055 allocatable 64-byte atoms (8,259,520 bytes). Original
allocators use an 8 MiB configured heap; Sublet receives an equally sized usable
pool plus separately allocated metadata. The legacy monitor rounds region
bounds to pages, so the Sublet adapter explicitly caps its atom count. This
avoids accidentally giving Sublet extra allocatable blocks. Lookaside is off.
All completed units must match all 32 native phase row counts and hashes, not
just the final return code. Native reference oracles are in [oracles.json](oracles.json).

The measured disjoint pool identity is `C + Q + F = P`: rounded active block
spans, quarantined spans, reusable free spans, and allocatable pool capacity.
Every reported sample checks this identity by enumerating the free lists.
Nonnegative phase IDs sample just before that phase; phase `-1` samples after
database close. The peak counter resets at each unit boundary and records all
allocation/free events during that unit. Samples also verify monotonic address
coverage and that no observer table overflow occurred.
The primary selected footprint is `H = C + Q + M`. Its peak is recorded on
allocator events, not by summing separate peaks. `M` here is explicitly:

| Arm | Selected metadata M |
|---|---|
| Capstone original | control bytes + `sizeof(mem5)` |
| Sublet | configured table buffer + `sizeof(mem5)` |
| CheriBSD original | control bytes + `sizeof(mem5)` |
| PoisonCap | control bytes + page-rounded external links + static queue + `sizeof(mem5)` |

This **selected allocator ledger** excludes small static support objects,
unused outer-buffer alignment, domain/image/stack storage, outer libc metadata,
kernel revocation state and capability nodes. It is not complete process memory.
In particular, the 33-byte original heap tail, Sublet's 200-byte unused table
reservation and rounded arena tail are outside H. No fragmentation ratio is
reported: original requested object bytes are not yet instrumented. `C` already
includes allocation rounding. `H` declining also does not imply pages are
returned to the OS: the entire arena stays reserved for the process.

For context, requested arena/table reservations are 8 MiB for each original
allocator, about 12.924 MiB for Sublet's pool plus table allocation, and about
9.113 MiB for PoisonCap's heap, rounded links and static queue. These are neither
minimum successful capacities nor physical backing measurements. The burst
comparison holds **usable pool capacity** equal, not total reserved bytes.

The observer uses three integer-index bitmaps totaling 49,152 bytes, plus scalar
counters and observation code, outside the application allocator. It holds no
application capabilities and is excluded from H in every arm. Allocation-start
reuse counts are available but are not interpreted as retirement-side reuse
probabilities. Timing output from `speedtest1` is discarded.

## Platform and reproduction

Both Capstone arms in the primary campaign use the same QEMU binary and
`CAPSTONE_REV_NODES=4194304`. At its default 65,536 nodes the legacy SQLite
Sublet host exhausts identities during repeated work. The automatic collector
in the new supervised runtime does not operate on this legacy path. Raising
the provisioned capacity permits the finite application-pool experiment; it
**does not establish bounded total memory or hardware scalability**. No timing,
node-memory or sweep-cost conclusion is made. Independent processes use fresh
legacy guests because starting a second domain after large-region cleanup
faulted in the old monitor. Units within a process share one boot and allocator.

The PoisonCap guest uses the published kernel/libc/SDK, with outer libc
revocation disabled before SSH starts (`security.cheri.runtime_revocation_default=0`)
and `_RUNTIME_REVOCATION_DISABLE=1` in each test process. Explicit nested
revocation remains enabled in the protected source; return errors are fatal.
Guest libc identity and the actual runtime setting are preserved in
[provenance](provenance/cheri-identity.json).

[builds.json](builds.json) records actual compilation/link arguments from
byte-identical reproductions of the measured binaries, source/driver/compiler
hashes, platform differences and baseline definitions. The rebuild also explicitly stages the matching 3.22.0 header; this
reproduces the measured Capstone binaries byte for byte. The two extra `SQLITE_*`
harness constants in the CheriBSD rebuild are inert there; adding them reproduced
both measured binary hashes exactly. Application and driver share one translation
unit at `-O0` in all arms. Capstone support code and the platform libc/VFS remain
different. The gate is a necessary comparability check, not proof of identical
allocation demand across ABIs.

Fetched upstream and artifact sources remain outside the repository. The
[patches](patches/capstone-base.patch) reconstruct the measured variants: apply
the platform base patch to the official amalgamation, then the arm's observation
and protection patch. Apply the arm's driver patch to official `speedtest1.c`.
The published fork's different source ID and CHERI compatibility edits are
explicitly preserved; matching version strings alone is not the source claim.

For new runs, use [prepare-sqlite-memory.py](../../prepare-sqlite-memory.py) with
explicit upstream, adapted Capstone, published PoisonCap and official-driver
paths under `$CAPSTONE_TMP_ROOT`, then [build-sqlite-memory.py](../../build-sqlite-memory.py).
Source `capstone/tests/capstone-test-env.sh` first and explicitly select compiler,
linker, Buildroot, guest compiler and CHERI SDK/sysroot. These tools reuse the
existing port builders; they do not introduce another VM manager. The common
observer now emits short print calls to avoid the Capstone long-varargs issue;
the measured CheriBSD patch retains the equivalent single-call output form.

Measured commands and binary identities are in [runs.json](runs.json); complete
compressed output is under `raw/`. To verify and regenerate the figures:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/study/check-build-comparability.py \
  capstone/experiments/study/results/sqlite-normalized-memory-20260927/builds.json
python3 capstone/experiments/study/plot-sqlite-normalized.py \
  capstone/experiments/study/results/sqlite-normalized-memory-20260927
```

The plotter requires matplotlib and rejects missing or corrupt ledgers,
nonmonotonic event counters, duplicate output, incomplete successful runs,
allocation failures mislabeled as successful, and completed-unit oracle drift.

The [excluded-attempt ledger](excluded-attempts.json) retains setup failures,
corrupt initial telemetry, the default-capacity resource failure and passing
default-capacity controls. They are separate from the primary four-arm plots.
The [short-print equivalence check](provenance/observer-print-equivalence.json)
compares every phase ledger and result oracle against the measured CheriBSD
output after rebuilding through the reusable source/build tools.

[Node-capacity controls](provenance/node-capacity-controls.json) verify exact
ledger/oracle equality for the successful default-capacity controls and their
large-capacity counterparts. [Source reconstruction](provenance/reconstruction.json)
checks every archived source and driver patch against the manifest hashes.

The [external artifact manifest](archive.json) identifies the complete scratch
source/build/run archive and its checksum. No benchmark sources are vendored
in this repository.
