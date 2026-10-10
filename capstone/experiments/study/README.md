# Paired application memory studies

## Current review entry point

Start with the [six-application reuse figure](results/reuse-six-applications-20260928/README.md)
and its PNG, PDF, CSV and provenance. SQLite, mruby, FFmpeg, PostgreSQL,
CPython and Perl have checked four-arm inner-reuse evidence; Perl's boundary is
its SV-head allocator ([adapter](../../ports/perl/sv-heads/README.md)).
The [PostgreSQL](results/postgres-reuse-four-arm-20260928/README.md),
[CPython](results/cpython-reuse-four-arm-20260928/README.md) and
[Perl](results/perl-reuse-four-arm-20260928/README.md) archive validators
recheck 12 processes each. The current figure pools all three repetitions in
each arm and retains the observed-reuse-only scope. The earlier
[five-application figure](results/reuse-five-applications-20260928/README.md)
remains as published.

The campaign descriptions below preserve earlier policy and qualification
states. Their then-unresolved CPython/PostgreSQL blockers are superseded by
these archives. Historical figures must retain their recorded policy labels;
none supplies an equal-scope total-memory or physical-working-set comparison.

## Campaign history and reproduction

The [published-threshold campaign](results/published-policy-20260928/README.md)
validates 48/48 fresh mruby/FFmpeg processes and provides checked reuse and
selected-memory plots. Six new outer-default SQLite runs paired with six
archived Capstone controls bring the checked figure set to 60 processes.
The [policy audit](poisoncap-policy.md) fixes the
reference thresholds, outer defaults and disclosed runtime repairs. Historical
figures below retain their custom-policy labels; they are not this campaign.

The [adapted FFmpeg FATE four-arm qualification](results/ffmpeg-fate-four-arm-20260928/README.md)
passes 24/24 complete decoder processes on two known input cases. A narrow
CheriBSD L2-superpage lookup fix removes the published kernel's poison-probe
panic; both CheriBSD arms use the same patched kernel. Lease-reuse bins remain
identical across all arms, while the PoisonCap temporal adapter uses transient
snapshot backing. This selected component does not establish a total-memory
advantage.

The [three-application reuse CDF preview](results/cross-application-reuse-preview-20260928/README.md)
places the qualified SQLite, mruby and FFmpeg nested-allocator results on one
paper-width figure with the same all-issues denominator and three independent
process repetitions per arm. It is derived from existing campaigns; the other
three application rows remain unqualified.

The [complete-interpreter CPython qualification](results/cpython-objects-qualification-20260928/README.md)
adds nine accepted application processes across the Capstone spatial/Sublet
pair and CheriBSD spatial control. The protected PoisonCap interpreter is
built but hits a reproducible published-kernel VM-map panic, so CPython is not
yet a fourth complete memory panel.

The [normalized SQLite memory campaign](results/sqlite-normalized-memory-20260927/README.md)
adds matched original-layout controls, three repeated-work runs per arm, event
ledgers and address-footprint plots. It reports a reuse advantage for Sublet
and a countervailing table cost; see the explicit selected-memory scope.
The [four-arm release-gap follow-up](results/sqlite-reuse-gaps-20260927/README.md)
now measures exact same-start reuse timing inside the complete SQLite application:
12/12 long runs match all SQL phase oracles; Sublet's histogram matches its
original bin for bin, while corrected PoisonCap shifts reuse to longer gaps.
The [FFmpeg whole-decoder lease-gap follow-up](results/ffmpeg-reuse-gaps-20260927/README.md)
adds a second nested-allocator boundary: 36/36 full 1/4/16-stream processes
match the frame oracle. All four arms have exactly equal pool lease-gap bins,
while the selective PoisonCap temporal adapter targets 116.155 MiB of
cumulative payload spans with poison, clear and copy operations in the
16-stream workload. This is not measured physical memory traffic or total footprint.
The [mruby GC-slot campaign](results/mruby-gc-memory-20260927/README.md) adds
a third complete-application boundary. All 24 upstream AO-render processes
at widths 8 and 16 match independent binary PPM oracles. Sublet and the PoisonCap spatial
control have identical slot-gap histograms; the explicit PoisonCap temporal
adapter shifts reuse to longer gaps and peaks at nine GC page groups versus
six in its spatial control. Sublet's page metadata and post-render retention
remain visible countercosts.

The [application figure design](../../docs/plans/application-memory-figures.md)
defines three main figures from complete application executions: paired memory
cost, address reuse, and sustained-work/burst behaviour. The
[SQLite paper-layout preview](results/sqlite-normalized-memory-20260927/paper-layout/README.md)
reformats existing checked data at manuscript width; it adds no measurements
and keeps missing full-memory instrumentation explicit.

This directory organizes known application benchmarks using the existing
persistent-guest runners. `catalog.json` records candidate suites, source pins,
application scope and qualification gaps. `study.py` fixes the workload matrix,
emits runner points from checked local artifacts, and reports the complete
attempt denominator. It does not implement another VM manager or vendor suites.

The general catalog has six applications and eight candidate suites. The
mruby AO case now has a repeated four-arm memory measurement through the
shared guest runners. Previous discovery results are not renamed as standard
benchmarks. The [research and execution plan](../../docs/plans/application-benchmark-study.md)
explains benchmark selection, measurement layers and the rollout.

For the nested-lifetime comparison, use the [Sublet/PoisonCap study
design](../../docs/plans/sublet-poisoncap-memory-study.md). It identifies the
published SQLite source, matched platform controls, policy audit findings,
cost plots and missing application integrations. The existing default-CheriBSD
pair remains a separate reference; it is not the PoisonCap spatial control.
The [application memory follow-up](results/memory-followup-20260927/README.md)
contains full-SQLite budget attempts and a selective FFmpeg adapter control.
It preserves kernel panics and port faults as unresolved outcomes, rather
than treating failed attempts as lower bounds on required memory.
The [full-application campaign contract](../../docs/plans/application-memory-campaign.md)
defines one experiment family and measurement schema for every admitted case:
fixed-live churn, live-set scaling, burst recovery, and fixed-budget progress.
Existing pilot data are discovery inputs, not a pre-registered confirmatory
campaign.

## Comparisons

| Arm | System allocator | Nested allocator | Process revocation policy |
|---|---|---|---|
| `capstone` | Shared SDK first-fit heap (`level0.c`) | Original application implementation | Existing Capstone runtime |
| `capstone-sublet`, profile `nested` | Same first-fit heap | Application's Sublet integration | Existing Capstone runtime |
| `capstone-sublet`, profile `outer-malloc` | Sublet malloc | Original application implementation | Existing Capstone runtime |
| `cheribsd-revocation-on` | Default CheriBSD malloc | Original application implementation | Explicitly enabled |
| `cheribsd-revocation-off` | Same binary and malloc | Same implementation | Explicitly disabled |

Each plan has exactly four arms and exactly one profile. The nested profile
addresses the nested-allocator paper question. The outer-malloc profile is
a separate comparison continuing the earlier experiments. Neither means
disabling the Capstone ISA. The first-fit heap is the SDK's arena allocator, not jemalloc;
cross-platform differences include allocator policy and OS/runtime differences.

Until 2026-10-10 `study.py` also planned a `nested-poisoncap` comparison
(`capstone`, `capstone-sublet`, `poisoncap-spatial`, `poisoncap-temporal`) from a
separate `catalog-poisoncap.json`, and `applications/comparison-build.py` built
the PoisonCap mruby and FFmpeg binaries. Both were removed with the PoisonCap
adapters. The campaigns above that used them stay as recorded, and the plot
scripts and `applications/cheribsd-run.py` keep the PoisonCap parsing that
re-validates their raw archives.

CheriBSD uses one kernel, libc, application binary and inputs for both arms.
Only `_RUNTIME_REVOCATION_ENABLE=1` versus `_RUNTIME_REVOCATION_DISABLE=1`
changes. The shared runner starts each study process with a clean environment,
keeps kernel defaults unchanged, records the effective environment, and checks
`malloc_revoke_enabled()` at every measurement phase. Other allocator-policy
overrides are rejected. The legacy `cheribsd-default` arm still inherits default
policy without these switches. [Implementation at the measured source pin](https://github.com/CTSRD-CHERI/cheribsd/blob/88f39900c32928d807dba245fba138808c666f34/lib/libc/stdlib/malloc/mrs/mrs.c#L1382).

## One plan, existing runners

Source the project environment before test/build/run commands. Plans, fetched
sources, build trees, local bindings and raw output belong under
`$CAPSTONE_TMP_ROOT`. Curated evidence and benchmark definitions belong here.

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/study/study.py list
python3 capstone/experiments/study/study.py plan \
  --profile nested --suites sqlite-speedtest1 cpython-pyperformance mruby-upstream \
  --repeat 3 --seed 20260927 --out "$STUDY_ROOT/plan.json"
```

The default variant executes upstream work counts. For size/retention/epoch
experiments, pass `--matrix matrix.json`. The matrix must explicitly cover every
selected suite case; each case maps variant names to work parameters:

```json
{
  "mruby-upstream/bm_so_lists.rb": {
    "small": {"list_size": 1000, "iterations": 30},
    "reference": {"list_size": 10000, "iterations": 300}
  },
  "mruby-upstream/bm_so_mandelbrot.rb": {
    "reference": {"width": 600, "height": 600}
  },
  "mruby-upstream/bm_ao_render.rb": {
    "reference": {"work": "upstream-default"}
  }
}
```

Nondefault parameters need a recorded workload adaptation. Changing the matrix
creates a different plan. Freeze the qualified plan and bindings before the
main campaign; do not choose the largest favorable endpoint after plotting.
Keep discovery and confirmatory campaigns separate.

## Qualified local bindings

A binding maps `suite/case@variant` to one reviewed workload contract. All four
arms are required before `points` emits any member of that comparison. Missing
bindings stay `unqualified`; an invalid supplied binding is an error. The
binding format is:

```text
{
  "suite/case@variant": {
    "source": <exact source object from catalog>,
    "application_version": <matching application version>,
    "profile": "nested" | "outer-malloc",
    "parameters": <exact parameters from plan>,
    "adaptation": <description of unchanged and adapted upstream behavior>,
    "oracle_reference": <preserved native reference stdout file>,
    "oracle_reference_sha256": <its SHA256>,
    "oracle": {"expected_stdout": <exact output>, "expected_phases": [...]},
    "input_sha256": {<input role>: <SHA256>, ...},
    "arms": {
      <each of the four arm IDs>: {
        "binary": <host binary path>,
        "build_manifest": <existing build manifest path>,
        "inputs": {<input role>: <host path>, ...},
        "resources": <pool/grant, tables, observer, stack and metadata accounting>,
        "qualification_evidence": <adapter correctness result identity>,
        "point": <existing application runner point with argv/environment/files>
      }
    }
  }
}
```

Build manifests identify the application, binary SHA256, `allocations=true`,
and `allocations_sha256`. Capstone manifests also carry `heap` and `nested`;
the latter is `none` or the application ID whose nested allocator is integrated.
The existing builder supplies this for CPython and mruby. Other integrations
need an application adapter before receiving that identity. Never manufacture
a manifest that labels an allocator replay as an application build.

Both CheriBSD arm entries share the same binary, manifest, argv, environment,
files, resources and qualification evidence. The runner supplies the policy
switch. Input hashes are checked across all arms; output and phase oracles are
common. The checks verify identities and declared configuration, not that a
manually described adapter is semantically correct. Qualification evidence must
include native reference execution and a review of workload/allocator boundaries.

```sh
python3 capstone/experiments/study/study.py points \
  --plan "$STUDY_ROOT/plan.json" --bindings "$STUDY_ROOT/bindings.json" \
  --platform capstone --repetition 0 --out "$STUDY_ROOT/capstone-r0.json"
python3 capstone/experiments/applications/run.py \
  --state "$CAPSTONE_VM_STATE" --points "$STUDY_ROOT/capstone-r0.json" \
  --repeat 1 --out "$STUDY_ROOT/capstone-r0"
```

Use `--repeat 1` on the existing runner: each emitted point already represents
one planned repetition. For CheriBSD, emit `--platform cheribsd` and use
`applications/cheribsd-run.py` with its existing `--sdk/--rootfs/--disk/--port`
options, or `--key` to attach to an owned persistent guest. Use the existing
interpreter environment with pexpect. Both policy arms run in one guest boot.
Do not run an empty point list. Respect the shared QEMU lock; stop only an owned
idle VM when switching platforms, and restore it in a `finally` block.

Repetition blocks alternate platform order. Within each block, a seeded order
interleaves both arms and workloads. Preserve that order when emitting each
platform/repetition file. VM changes may be batched when boot cost demands it,
but record that departure from the planned platform order. No automatic retries
or hidden restarts are permitted. Independent cells survive an ordinary
application fault; lost VM control or failed cleanup stops the platform batch.

Pass previous `runs.jsonl` files with `--runs` to emit only unattempted cells.
Failures remain attempted; they are not retried by resume. Duplicate attempts,
foreign plans, changed binding identities and changed local artifacts are rejected.
`study.py status --plan ... --bindings ... --runs ...` lists pass/failure/pending/
unqualified counts for every workload and arm. Unrecorded cells remain pending;
they cannot disappear from a success denominator. A lost runner between launch
and writing its record needs manual reconciliation before resuming that cell.

## Current verification and next integration

The [memory metric specification](memory-metrics.md) defines the scientific
endpoints for the four-arm campaign: a disjoint byte ledger, simultaneous
peaks, fixed reuse cohorts, sustained burst recovery and fixed-budget progress.
It distinguishes allocator capacity from residency, and address history from
an access working set. The current pilots do not yet provide every required
counter; successful execution and build-manifest checks are necessary but
insufficient for a paper memory claim.

The [SQLite 3.22.0 nested-memory pilot](results/sqlite-322-memory-20260927/README.md)
qualifies all 32 official `speedtest1 main --size 1` phases across the four
nested arms against an independent native SQL-result oracle. It preserves
per-phase memory records, three figures, raw-log digests, source/binary hashes
and failed capacity attempts. The selected passing capacities yield 0.80 MiB
additional application-visible backing for Sublet within Capstone and 7.58 MiB
for corrected temporal PoisonCap within its spatial control. Smaller corrected
PoisonCap trials panic in the published kernel, so these are exploratory
configuration observations, not minimum-capacity or total-memory claims.
The [build audit](results/sqlite-322-memory-20260927/README.md) additionally
finds unmatched SQLite source forks and compile options. These figures cannot
be promoted to a cross-platform memory claim until a rebuild records the
actual compiler commands and passes the [four-arm build gate](check-build-comparability.py).
The gate takes one JSON manifest with `schema: 1` and an `arms` object keyed by `capstone`,
`capstone-sublet`, `poisoncap-spatial`, and `poisoncap-temporal`. Each arm records
`platform`, `upstream_sha256`, `port_source_sha256`, `base_port_patch_sha256`,
`protection_patch_sha256`, `driver_sha256`, `workload_sha256`, `binary_sha256`,
`compiler_sha256`, `target`, `vfs`, `runtime_sqlite_config`, and actual argv arrays
`compile_argv`, `driver_compile_argv`, and `link_argv`. Invoke
`python3 capstone/experiments/study/check-build-comparability.py MANIFEST.json`.
Unknown values or reconstructed command lines do not qualify. The platform
VFS and compiler may differ across OSes, while SQLite options and optimization
must match across all four arms and the platform baseline must match within
each spatial/protected pair.

The [whole FFmpeg 9.0.1 decoder pool pilot](results/ffmpeg-pool-memory-20260927/README.md)
connects the existing PoisonCap pool adapter to the complete configured
decoder. Across 1, 4 and 16 independent streams, six new PoisonCap runs and
eighteen earlier Capstone repeats match the exact frame oracle. The PoisonCap
temporal adapter retains a 315,072 B snapshot and increases within-platform
jemalloc allocated by 294,912–318,336 B. The two FFmpeg figures keep that
process ledger separate from Capstone's system-allocator peak and pool payload
accounting; they make no cross-platform total-RSS or QEMU-speed claim.

The [upstream mruby lists readiness run](results/20260927-mruby-lists.json)
passes all four original arms at 300 iterations and 10,000 elements. Prepare
the pinned fetched source with `prepare-mruby-lists.py --source PATH --out DIR`,
then run the generated script unchanged on each interpreter. The adapter adds
per-iteration checks, two phase writes and final text output, without forcing
GC. An independent native Ruby run supplies the preserved output oracle.
Both Capstone arms recover from six allocation failures at their 64 MiB system
allocator budget. These are functional passes, not a qualified nested-memory ranking:
backing budgets and internal-slot counters still need matching.

The [PoisonCap source/build audit](results/20260927-poisoncap-source-audit.json)
records source and binary hashes, the published quarantine paths, and the
header prerequisites needed to compile its SQLite fork. It is separate from
application qualification.
The [two SQLite execution smokes](results/20260927-poisoncap-sqlite.json)
complete the artifact's 20 active main phases at size 1; twelve are commented
out and the main result oracle is absent. The
[external readiness archive](results/20260927-poisoncap-archive.json) preserves
the commands, raw output, builds and tested sources, excluding guest credentials.

The planner has thirteen tests for pairing, matrix identity, missing controls,
scope, input/observer/oracle identity, and preservation of failed attempts.
Six CheriBSD runner tests cover false passes and explicit policy selection.
The real policy smoke uses existing FFmpeg/mruby discovery images, not known
benchmark results: both on/off pairs and a subsequent on process pass in one
guest, with the expected state at every phase. See [the checked policy result](results/20260927-policy.json).
The [external evidence archive](results/20260927-archive.json) preserves raw
attempts, commands and the tested sources, with guest credentials excluded.

Next: qualify mruby's upstream workloads, broader SQLite speedtest1 sizes and the selected
pyperformance bodies, including nested allocator counters. The existing
runner contract currently expects textual stdout; binary-output benchmarks
(such as Mandelbrot PBM output) need exact byte-hash oracles before qualification.
PostgreSQL's structured SQL oracle and FFmpeg's broader CLI need corresponding
shared runner/build support. No full standard suite is claimed to pass yet.
