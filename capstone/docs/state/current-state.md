# Current Capstone state

Minimal snapshot. Read first in every session.

## 2026-09-30 — delegated syscall buffer bounds prototype

The `trusted-linux-syscall-bounds` lane adds per-object bounds to the default
level0 heap and checks requested buffer spans before delegated syscalls use
their exchange region. The [recorded one-hart test](../../runtime/tests/application/results/20260930-syscall-buffer-bounds.json)
shows `malloc(16); read(fd, p, 4096)` returning `EFAULT` without consuming
pipe data; a deliberately unbounded heap build misses that check. The direct
exec contract passes its nine modes, and the socket contract passes 11/11 in
the pinned guest. The bridge still lacks recoverable copy faults after
concurrent revocation and is not a native Linux capability ABI.

## 2026-09-30 — trusted Linux memory design

`memory-trusted-linux` starts from `dev` at `4439dd0a55f9`, without the
caplified mapping implementation stack. The
[new design direction](../design/trusted-linux-memory.md) trusts Linux and
relevant firmware, uses ordinary process page tables, and retains Capstone
object bounds and lifetimes. The
[application compatibility plan](../plans/trusted-linux-application-compatibility.md)
sets ordinary Linux OS functionality as the target for all seven application
ports. It defines M0–M7 gates for the inventory, execution/ABI choice, VM,
threads, processes, libraries, application parity and measured benefit.
Dynamic malloc/free and fork with independently cloned private lifetime state
remain design targets; the final execution mode and hardware choice are open.
This is a documentation change, with no new implementation or qualification.

## 2026-09-29 — delegated signals

On `delegation-signals`, synchronous delivery passes the 26-mode
[signal contract](../../runtime/tests/application/results/20260929-signal-contract.json)
and 25 native tests. The same launcher passes the application gate, six binfmt
cases and Perl `t/base` (9 files, 493 assertions). The contract covers
positive-PID waits, realtime queue capacity and payloads, jumps out of
handlers, `sigtimedwait` output and inherited signal state. Asynchronous
delivery into a computing domain remains the later bell; the other named
deviations are in [the signal plan](../plans/delegation-signals.md).

## 2026-09-29 — application ports require delegation

The `delegation-ports` branch migrates all
seven application recipes to the shared ABI-v2 SDK: Perl, mruby, CPython,
PostgreSQL single-user, SQLite in-memory SQL, the configured FFmpeg decoder
and offline tshark. The launcher rejects v1 images with exit 126. Private
application entry points, HostCall launchers, argument side files and the
PostgreSQL fake-root/named-input patches are removed. Standalone allocator
and instruction probes are separate targets, not application compatibility.

[Migration instructions and qualification](../../ports/common/application/README.md)
link the [checked result lines](../../ports/common/application/results/20260929-delegation.json).
New runs pass 108 mixed starts, six shell/exec checks, Perl 9/9 files with
493 assertions, both mruby regular suites, and the selected native-matched
workloads for the other ports. FFmpeg/tshark safety is 41/41. libc-test is
46 PASS, 4 FAIL, 4 FAULT, 3 NOBUILD, 20 EXCLUDED; all 45 earlier passes remain.
Two newly buildable TLS tests reach unsupported thread creation and fault.

The qualified compiler is `compiler/sroa-keep-capability-whole` at
`7d01722aab88`, not the compiler sources on the delegation-only branch.
The mixed matrix uses 1024 MiB CMA and 768 MiB retained process storage.
Protected PostgreSQL passes at 262,144 nodes (84,193 high water), after
exhausting 65,536. The larger protected mruby GC stress exhausts both
capacities; only its smaller smoke is qualified. FFmpeg uses the measured
`-O1 -fno-omit-frame-pointer` workaround for C-70. Dynamic memory grants,
asynchronous signal handlers and general threads remain outstanding.
Historical memory/performance archives retain their original images and ABI.

dev (through `2909060e`) is merged in. Its two HostCall-era arms are ported to
v2 rather than kept as a second transport, and pass as registered from the
merge tree: the FFmpeg pool corpus on the Sublet port (36/36, three rounds of
both arms) and tshark's wmem `chunks` arm with its `sublet` control (safety
13/13 each, all stages and capture verdicts). [Result lines](../../ports/common/application/results/20260929-dev-merge.json).

## 2026-09-29 (evening) — R-43 FIXED ON SILICON: the replay design `8f6a0af98` is flashed and accepted a1..a10

**Resident bitstream is `caplifive_r43_8f6a0af98.bit`** (RTL `8f6a0af98` = R-42 + R-43 second fix + R-45; sha256
`61443441…5345`), flashed on the lead's word, name read back from the console. Every arm that trapped cause 25 on R-42
now completes with its oracle: the R1 warm and cold harnesses, live128/512 (4352 / 17408), and **P1 cell 6 `-O2`**
(112006 38bb59fd, 25,010 lookasides). The ladder and P1 cell 5 are unchanged within noise; the R-35 probe still traps
25 at `+0x4354` and the new refusal record reads the boot's first cause-25 verdict as the probe path (arm 0100: not live — dead
or a stale generation), not deny-on-miss, id 0x5f on two boots (fresh on the second); that it is the stale access's own verdict
is consistent with the trap but not proven (the record is first-since-reset; the aed492ab control is a different fixture).
Synthesis: loops the same as R-42, ORDER 0/500, WNS −9.595 (inside the null of −10.615). Simulation: the 8-variant batch
as predicted, the 92-test sweep 0 differences, lint at baseline. The after-audit refuted four documented properties
(recorded, none a defect); the `noclear` control was vacuous and is replaced by arm 6b (`capstone-ariane 5aa316e0d`).
Open and NOT in this bitstream: R-44 (CPMP adopt), R-46 (refetch metadata, accepted), M-1's RTL half, and a
timeout-DEAD residual of R-43 (fail-closed, wedged-rev-node-only). One N=1 note: the cold revoke ramp reads 2–7 %
lower mid-range with a candidate mechanism (probe reads pre-warm the D-cache). Report:
`tests/fpga-repros/R43-revocation-cache-false-deny/` (results/board-8f6a0af98.result-lines.txt); R-12's reclaimer
now has its own folder (`R12-revnode-exhaustion-reclaimer/`).

## 2026-09-29 (later) — R-43's first fix is REFUTED BY SYNTHESIS; R-45 is not implicated; redesign next

`0f5185a6d` routed at WNS −24.495, against R-42's −10.615. All of the worst 500 paths run from the revocation
lookup into the load/store request (R-42: 0 of 500). The combinational stall gate is the cause, even though
the fix is correct in simulation. Not flashed; the board stays on R-42. The redesign must never gate the
load request: replay the missed access through the existing registered exception path instead. Result
lines: `tests/fpga-repros/R43-revocation-cache-false-deny/results/synth-0f5185a6d.result-lines.txt`. An
early "3 new loops" reading was withdrawn: they are the same loop families, cut differently.

## 2026-09-29 — delegated runtime review

The `delegation-spawn` stack has reviewed syscall marshalling and process
lifetime fixes, with [checked native and guest evidence](../../runtime/tests/application/results/20260929-delegation-review.json).
Twenty-one ASan/UBSan native tests and 19 Python tests pass; the v1 application
gate passes 108 starts, and fresh v2 libc-test has 45/77 PASS. Output tails,
command pointers, vector offsets, private descriptors, spawn inheritance,
wait/kill confinement and exec continuation are covered. Fault symbolization
checks the sealed image SHA-256. See [applications](../../runtime/applications.md#review-verification-2026-09-29).
The follow-up [binfmt qualification](../../runtime/tests/application/results/20260929-delegation-binfmt.json)
passes Perl `t/base` 9/9 (493 assertions), direct shell execution with original
argv[0], 108 application starts, and the same 45 libc-test passes. Buildroot
`227fdfa` enables the QEMU kernel option; the VM setup mounts and registers the
format with literal magic escapes. Complete process-group behavior, memory and
signal delivery remain open. The 88-byte entry is not an io_uring SQE,
and the historical rdtime sample is not an icount/cycle result.

## 2026-09-29 — R-43 and R-45 fixed in RTL (capstone-ariane `r43-query-on-miss`, `0f5185a6d`); synthesis next

- **R-43:** R-35's revocation cache refused LIVE capabilities once ~256 revocation ids were live, which
  killed both R1 harness runs on the resident `caplifive_r42_6cbdaeeb4.bit`.
  - Fix `bbd4d1478`: on a miss, probe the rev-node unit and resolve from the existing read tap.
- **R-45:** a load/store right after REVOKE/DROP could be checked before the revocation took effect. It
  predates R-43, which extended it.
  - Fix `0f5185a6d`: flush younger instructions when a REVOKE/DROP commits (the lead chose to close it in
    this bitstream).
- **R-46** (filed, accepted for this bitstream by the lead): a commit refetch keeps the PC-capability
  metadata. Harmless with one code capability per domain.
- **Evidence:** verified in RTL simulation with seven deliberately broken builds, each failing as
  predicted; R-35's fixture unchanged; 92-test sweep neutral; lint gate PASS. Details in
  `tests/fpga-repros/R43-revocation-cache-false-deny/`.
- **Not yet synthesized or on silicon.** Resident: `caplifive_r42_6cbdaeeb4.bit`, which has R-43.

## Review baseline (2026-09-28)

The runtime and application study are organized into dependency-ordered review
branches: `domain-process-runtime`, `application-runtime-capacity`,
`application-memory-tooling`, `application-nested-ports`,
`perl-cheribsd-interpreter`, and `application-memory-results`. **All six are on
dev as of 2026-09-28** — #108, #110, #111, #113 and #114 merged, and #112's
content through merge commit `9881dc4781c8`, which GitHub records as closed
rather than merged because its branch advanced afterwards.
The runtime capacity branch also requires QEMU's `runtime-node-reuse` follow-up.
The squashed runtime tree and reconstructed study tree match their measured
predecessors exactly; only this review-status documentation changes afterward.
Original development commits remain available for measurement provenance.

The verified measurement baseline is five complete four-arm inner-reuse
comparisons. Perl's CheriBSD interpreter smoke passes, but its inner lifetime
adapters remain the next implementation milestone. Review publication supersedes
the older instructions below to keep the study off a PR; it does not mark the
broader memory study complete or add new guest measurements.

## 2026-09-28 — CPython four-arm reuse and CheriBSD Perl bootstrap

The [CPython archive](../../experiments/study/results/cpython-reuse-four-arm-20260928/README.md)
validates 12/12 complete-interpreter JSON/GC processes, three per arm: six new
CheriBSD runs and six archived Capstone runs. Both new CheriBSD modes enable
outer libc revocation and use one fresh `-O1` binary and repaired kernel.
A VM-object probe avoids recursive user faults under the revoker's VM-map lock;
unsupported probes fail explicitly. The resident/PROT_NONE/zero-fill regression
passes. General swap-pressure and concurrent-VM qualification remain outstanding.

Deferred free-list publication retains pymalloc occupancy until revocation
completes. The published 4,096-entry and 16 MiB/quarter thresholds are transferred
without allocation-triggered sweeps. Every protected process records 18 capacity
drains and one teardown drain, with an empty final queue. Sublet and its control
have median observed reuse gaps [2,3]; PoisonCap has [4096,8191] versus its own
control's [2,3]. Reuse shares are 63.48% in the Capstone pair and 55.77% versus
65.13% in the CheriBSD pair. These are successful-handout, observed-reuse metrics,
including startup/shutdown, not physical working-set or total-memory results.
The [first figure](../../experiments/study/results/reuse-five-applications-20260928/README.md)
now contains five applications and seven workloads.

The [Perl CheriBSD recipe](../../ports/perl/cheribsd/README.md) freshly builds
5.36.3 purecap with the existing seven interpreter patches and an explicit
`d_nanosleep` configure answer. Its 17-section smoke matches native output under
both libc policy switches. This is interpreter qualification only: SV-head/body
lifetime adapters and the inner reuse observer are missing on both comparison
sides. The outer Capstone Sublet malloc switch does not fill that gap. Keep the
study off a PR (superseded 2026-09-28: the review stack is merged into dev); complete Perl's inner boundary before adding a
sixth application.

## 2026-09-28 — PostgreSQL four-arm inner reuse

The [four-arm archive](../../experiments/study/results/postgres-reuse-four-arm-20260928/README.md)
validates 12/12 complete PostgreSQL 17.5 SQL processes: six new CheriBSD
processes and six archived Capstone processes, three per arm. All match the
native SQL oracle and have error-free inner histograms. The protected adapter
now transfers the published SQLite quarantine thresholds, with full-queue
revocation correction and no allocation-triggered sweep. Each protected
process performs five capacity sweeps, zero percentage/managed-reset sweeps.
Both CheriBSD modes use the same fresh `-O1` binary and corrected libc;
application libc revocation is verified enabled. Guest setup services use a
disabled default because the separate kernel lock panic can occur in SCP.

The blocker was fixed metadata exhaustion, followed by recursive allocation
of PostgreSQL error messages. Earlier parser attribution missed the PIE load
bias and was incorrect. Both modes now provision 65,536 chunk and 8,192 block
records, reporting occupancy and failing directly on exhaustion. The protected
processes peak at 20,472–20,502 chunk records. The enlarged metadata must be
charged in any later total-memory comparison.

Sublet observes 80.89% same-start reuses versus its control's 83.29%; both
median observed gaps lie in [4,7]. PoisonCap observes 36.20–36.27% versus
83.28% in its control, with median observed gap in [8192,16383]. The
[first-plot extension](../../experiments/study/results/reuse-four-applications-20260928/README.md)
pools all three raw histograms per arm; slight protected-process variation
is preserved in the archive. Four applications now have complete four-arm
reuse evidence. CPython and Perl remain incomplete. The SQL is qualification
work, not pgbench; no physical working-set or total-memory conclusion follows.

## 2026-09-28 — CPython inner-reuse qualification, three arms

The [archived observer campaign](../../experiments/study/results/cpython-reuse-three-arm-20260928/README.md)
passes 9/9 complete CPython 3.13.7 `objects.py 8 3 0` processes: three
Capstone spatial, three Capstone+Sublet and three CheriBSD PoisonCap-adapter
spatial. The common runners reject missing or inconsistent 32-bin inner
pymalloc histograms, and the raw archive validator reproduces every reported
count. The Capstone pair uses 262,144 nodes; an earlier underprovisioned
65,536-node attempt failed and is excluded. The protected PoisonCap
interpreter still has no full application oracle, so this is not a four-arm
comparison. The observer currently indexes successful new lifetimes, not all
failed allocation attempts required for the fixed-follow-up metric.

## 2026-09-28 — Published-threshold application measurements

The [fresh campaign](../../experiments/study/results/published-policy-20260928/README.md)
validates 48/48 complete mruby/FFmpeg processes with the published SQLite
quarantine thresholds transferred to these allocators, the disclosed queue
correction, and outer libc revocation enabled in both PoisonCap arms. The
[policy audit](../../experiments/study/poisoncap-policy.md) distinguishes this
transfer from an author-provided port. FFmpeg's eager reissue sweeps and
mruby's slot-exhaustion sweeps are removed from the new comparison.
Sublet's carved FFmpeg pool extent remains 1.00× its control; PoisonCap uses
6.66× and 12.84× for Xvid and resize. mruby peak GC ratios are 1.39× for
Sublet versus 1.67×/1.83× for PoisonCap; after AO16, Sublet's retained ratio
is worse (2.08× versus 1.83×). These are selected allocator quantities,
not total-memory or physical-working-set claims.

Outer defaults exposed a libc asm-output constraint defect and incomplete
retirement of stored poison capabilities. Both FFmpeg arms use the same
corrected libc; a focused allocation test and all complete decoder oracles
pass. SQLite additionally needs explicit nested poison-metadata retirement;
all six repaired 17-unit outer-default processes pass. Their inner ledgers
match the historical controls exactly. Paired with six archived Capstone runs,
the new SQLite plot shows 1.00× versus 3.99× allocated-address coverage and a
Sublet selected-allocator-peak countercost. The three new figures therefore
validate 60 processes (54 fresh plus six archived). Historical SQLite figures
still use outer revocation disabled and must retain that label. Review copies are under
`/home/biecho/nested-allocators-paper/review/published-policy-2026-09-28/`.

## 2026-09-28 — Three-application paper figures

The [checked figure set](../../experiments/study/results/application-memory-paper-20260928/README.md)
reanalyses SQLite, mruby and FFmpeg with common CDF axes, differences from
each spatial control, selected-memory companions and all eight workloads in
a supplement. All 96 reuse-process records reproduce exactly through the
original raw-data validators; the 12 separate SQLite memory transcripts pass
their SQL oracle and ledger checks. The five vector figures use paper width
and embedded fonts; a captioned PDF, LaTeX snippets and derived CSVs accompany
them. Five measurement-guard tests pass. No new application runs or total-memory
claims are added; the complete four-arm application count remains three.

## 2026-09-28 — FFmpeg adapted-FATE four-arm qualification

The [checked four-arm campaign](../../experiments/study/results/ffmpeg-fate-four-arm-20260928/README.md)
passes 24/24 complete FFmpeg 9.0.1 decoder processes over two adapted FATE
inputs, with three exact-oracle repetitions per arm and input. The original
published CheriBSD kernel's poison-probe panic was traced to a missing L2
superpage case in RISC-V `pmap_extract_and_hold()`. A narrow patch fixes the
lookup; both CheriBSD arms were rerun on the same patched kernel. All four
arms have identical pool lease-gap bins for each input. The protected
PoisonCap adapter peaks at 224.4 and 167.1 KiB of selected snapshot backing;
these are not total-memory results. The CPython protected interpreter still
hits its separate `share->excl` kernel panic with this patch. An isolated
trap-PC diagnostic identifies RISC-V `fupoison` probing a nonresident user
target while the revoker holds the VM-map read lock. A temporary bypass only
changed the failure into a 120-second pre-workload timeout; it was removed and
supplies no paper measurement. A subsequent fail-closed direct-map diagnostic
reached a backed page with no PTE and stopped rather than treating possible
stored poison as absent. That experimental kernel was also removed; the
protected CPython arm and its memory plot remain unqualified. A later
scratch kernel classified that specific vnode page and a swap page without
capability tags as safe misses and reached `startup` without panic, but the
protected interpreter still timed out before `baseline` at 120 seconds. An
external-free-list/quarantine candidate passed its spatial control; a
diagnostic protected run performed over 100 sweeps in 40 seconds, including
84 on normal block issue, so that candidate was not admitted. Its source
changes were removed from the branch. No CPython four-arm result is claimed.

## 2026-09-28 — Cross-application reuse preview

The [paper-width three-application CDF preview](../../experiments/study/results/cross-application-reuse-preview-20260928/README.md)
derives from the qualified SQLite memsys5, mruby GC-slot and FFmpeg pool-lease
four-arm campaigns. It checks three complete process repetitions per arm and
uses all inner-boundary issues as the denominator. SQLite and mruby separate
the protected PoisonCap reissue curves from their spatial controls; FFmpeg's
four curves coincide. It is not a six-application paper result, physical
working-set comparison or new measurement.

## 2026-09-28 — PostgreSQL and FFmpeg application-memory follow-up

The PostgreSQL 17.5 `-O1` Capstone original and memory-context Sublet images
pass 6/6 complete 2,000-row single-user `work.sql` attempts with the exact
22-row native SQL-output hash, across three fresh copies of the same cluster
per arm in one Linux VM boot. The SDK images take arguments and environment
directly from the shared runner. `dynamic_shared_memory_type=sysv` uses the
existing domain System V segment service; POSIX and file-backed mmap DSM
cannot run in this runtime. This is a functional two-arm qualification, not a
four-arm memory ranking: the Sublet context region and node storage are not
included in the reported outer-heap peak. The initial temporary cluster had
been used before the campaign. A clean native 16-byte-MAXALIGN fixture builder
now creates a pristine C-locale/GMT/System V cluster; its tree hash matches
the shared runner's declared fixture, and the [archived 6/6 campaign](../../experiments/study/results/postgres-pristine-20260928/README.md)
from that untouched source retains the exact 22-row native oracle. This still
needs the full four-arm and inner-storage ledgers before a paper memory plot.

The same persistent Capstone VM also passes 12/12 complete FFmpeg 9.0.1
decoder attempts on two [adapted FATE MPEG-4 inputs](../../experiments/study/fate-mpeg4-inputs.json)
at 20 and 150 frames, each with spatial and Sublet pool modes and three
repetitions. Every run matches the native per-frame oracle. A separate fresh
CheriBSD spatial guest passes 6/6 attempts across the same two inputs and
three repetitions. For each input the three qualified arms have identical
32-bin pool lease-gap histograms (262 issues/219 reuses, and 1,852/1,751),
so these inputs show no Capstone-versus-spatial reuse difference. The
protected PoisonCap arm hit the published kernel's `Poison probe missing page`
panic; that interrupted campaign is excluded. A second fresh guest running the
protected arm first hit the same panic; eagerly touching the 4 MiB pool
before use and omitting its final explicit `munmap` did not resolve it. The
generic CheriBSD runner now records `guest-panic` and aborts immediately when
the serial console reports one; a repeat diagnostic detected this panic four
seconds into the application attempt. The original FATE bitstreams were
losslessly remuxed to Matroska for the configured decoder, so these are
adapted application inputs, not official FATE scores.
The [archived 18/18 three-arm qualification](../../experiments/study/results/ffmpeg-fate-qualification-20260928/README.md)
preserves accepted raw runs and the excluded panic separately. At that point it
could not be rendered as a four-arm FATE memory figure; the later patched-kernel
campaign above supersedes that qualification for the four-arm input.

## 2026-09-28 — CheriBSD CPython complete-interpreter spatial qualification

A fresh CPython 3.13.7 purecap build with ordinary pymalloc links on
CheriBSD and passes the JSON/GC `objects.py 8 3 0` workload. The complete
interpreter now also links the existing PoisonCap pymalloc lifetime backend;
its adapter mode-0 control passes 3/3 in a fresh guest. At matched `-O1`,
Capstone spatial and real per-block Sublet modes pass 3/3 each in one Linux
boot. The [archived nine-process qualification](../../experiments/study/results/cpython-objects-qualification-20260928/README.md)
preserves exact oracle hashes and build evidence. Protected PoisonCap mode 1
reaches Python startup but triggers the published kernel's `share->excl`
VM-map lock panic, including when it runs first in a fresh guest. Both
interrupted attempts are excluded. The reported outer heaps omit the inner
pymalloc regions and Capstone node storage; no four-arm CPython memory plot
is established.

## 2026-09-28 — PostgreSQL complete-backend memory qualification

The PostgreSQL 17.5 single-user backend now builds in Capstone spatial,
Capstone with all four real Sublet memory-context hooks, CheriBSD purecap
spatial, and a CheriBSD binary with the existing PoisonCap context backend.
The pinned 2,000-row `work.sql` passes its 22-row native PostgreSQL oracle in
the first three arms and in the PoisonCap binary's mode-0 control. The
Capstone results were initially built at `-O2`; both Capstone modes have now
also been rebuilt at the CheriBSD pair's `-O1` and passed the shared-runner
oracle above. An earlier CheriBSD index-build SIGPROT came from a stale
`src/port/qsort.o` predating the tag-preserving swap patch; the new builder
forces that object to rebuild before linking.

The earlier PostgreSQL PoisonCap adapter sweeps every free and is not a valid
lower-bound memory comparator. The complete-backend variant instead poisons
at free, retains an external chunk queue, and sweeps before reissue. Its
protected `SELECT 1` qualification passes with 8,692 hands, 6,097 drops,
1,619 sweeps and a 525,312-byte peak queue. A full `work.sql` mode-1
diagnostic was stopped during its first INSERT after more than 53 minutes of
guest CPU and 7,154 sweeps; it produced no completed SQL oracle and is excluded
from the study. The current eager reissue policy needs batching or a different
threshold before full-workload qualification. No four-arm PostgreSQL paper plot or total-memory
ranking is established. The new build path and limits are in the
[single-user port](../../ports/postgres/app/README.md).

## 2026-09-27 — mruby GC-slot four-arm memory behavior

The [full mruby 4.0.0-rc2 AO-render campaign](../../experiments/study/results/mruby-gc-memory-20260927/README.md)
passes 24/24 independent native-PPM-matched processes at widths 8 and 16: three repetitions of
Capstone spatial GC, real per-slot Sublet GC, PoisonCap spatial GC, and an
explicit PoisonCap temporal GC adapter. Every arm issues 217,070 slots at
width 8 and 915,981 at width 16. Sublet and PoisonCap spatial have identical
32-bin release-gap histograms at both sizes. Within 1,023 subsequent issues,
Sublet reissues 75.38%/75.54% of slots at widths 8/16, versus 1.31%/1.95%
for the temporal PoisonCap adapter.
PoisonCap temporal peaks at nine GC page groups versus six in its spatial
control at both sizes; Sublet peaks at six versus six. The peak groups stay
flat across 4.22× more slot issues, within this tested range. Sublet's per-page metadata and its
retention of all-dead groups are countercosts, so this is a logical-reuse and
selected GC-page result, not a total-memory ranking. The process-level
CheriBSD jemalloc ledger excludes the mmap GC pages, and Capstone node
storage is not charged. `study.py` now admits a pinned mruby four-arm binding
for future planned runs; the reported campaign itself used the shared guest
runners directly, before that binding was assembled.

## 2026-09-27 — Normalized SQLite repeated-work memory

The [complete FFmpeg 9.0.1 decoder lease-gap study](../../experiments/study/results/ffmpeg-reuse-gaps-20260927/README.md)
now adds a second measured internal allocator boundary. All 36 four-arm
1/4/16-stream runs pass the exact frame oracle, and all 32 pool lease-gap bins
are identical across arms and three repetitions at each size. At 16 streams,
the selective PoisonCap temporal adapter targets 116.155 MiB of cumulative
payload spans with per-granule poison/clear and copy operations while snapshot
backing peaks at 36,288 B and finishes at zero. These operation span counters
are not time, DRAM traffic or total memory. Both platforms use the same prepared FFmpeg
9.0.1 source and the shared pool observer; Capstone's application SDK reuses
one Linux VM. The new allocator ledgers match the prior selective FFmpeg
campaign. This result shows why SQLite's delayed-reuse behavior cannot be
generalized to every nested allocator policy.

The [complete-application reuse-gap follow-up](../../experiments/study/results/sqlite-reuse-gaps-20260927/README.md)
adds 12/12 complete four-arm runs, all 6,528 native-matched SQL phases, and
same-start release-to-reuse CDFs over 550,137 memsys5 allocations per run.
Capstone original and Sublet are identical in every gap bin. The fraction
reusing a start within 15 allocations is 73.116% in both Capstone arms,
73.113% in CheriBSD original, and 0.052% in corrected PoisonCap. PoisonCap's
overall same-start reuse is 86.358% versus 99.561% in its own original.
Three repetitions per arm coincide. All 6,732 allocator phase rows match the
prior campaign except for the extra 524,560 bytes of static observer storage.
This is a full SQLite application result at its memsys5 boundary, not a
physical-memory, runtime, or other-application claim.

The [normalized SQLite campaign](../../experiments/study/results/sqlite-normalized-memory-20260927/README.md)
passes the four-arm build-comparability gate and all 12 repeated-work attempts
(three per arm, one warmup plus 16 measured full `speedtest1 main --size 1`
units per process). Every complete unit matches all 32 native SQL-result
oracles. The CheriBSD original-layout control now removes PoisonCap's external
allocator adaptation from the denominator. Both platform pairs have equal
allocatable atom counts, matching SQLite feature switches, lookaside off and
application/driver `-O0`; explicit platform patches and compiler differences remain.

Sublet's cumulative allocated-address footprint stays at 1.00× its original
baseline; corrected PoisonCap reaches 3.99× and then plateaus. Sublet returns
all pool spans after each database close, but its selected allocator metadata
is much larger. Selected peak H (rounded live + quarantine + specified tables)
is **4.68× original for Sublet versus 4.27× for PoisonCap**. This is a reuse
advantage with a metadata tradeoff, not a general total-memory win. Neither
address footprint nor H is resident working set or complete platform memory.

The fixed size-4 burst passes 3/3 in Sublet and both original-layout controls.
Corrected PoisonCap fails its first qualification with SQLite OOM during phase
190; two later repeats remain blocked. This is equal usable pool capacity,
not equal total reservation. No four-arm post-burst recovery claim is made.

The legacy SQLite path still needs 4,194,304 provisioned emulator nodes for
these repeated runs, identically configured in both Capstone arms. Its
65,536-node Sublet control faults; supervised-runtime node reclamation does
not operate on this legacy path. These application-pool metrics exclude node
storage and cannot establish hardware memory/scalability. New source/build
and plotting tools reuse the port builders; no new VM manager is introduced.
The earlier pilot below remains historical and build-unmatched.

## 2026-09-27 — SQLite budget and selective FFmpeg memory controls

The four-arm SQLite 3.22 pilot passes the same 32 SQL-result phases but **is
not build-normalized**: Capstone uses the official amalgamation with its
deployed omit/heap/VFS profile, while PoisonCap uses a ported fork with a
different source ID and incompletely recorded compile argv. The pilot memory
plots are exploratory. The [campaign contract](../plans/application-memory-campaign.md)
now includes a build-comparability gate; compile-only probing confirms that
the PoisonCap fork accepts Capstone's SQLite defines except the required
`SQLITE_OS_OTHER` VFS switch, and the CheriBSD prototype links. Its first
guest attempt panicked in the published kernel during `scp`, before SQLite
started. No normalized four-arm benchmark has run yet.

The [memory follow-up](../../experiments/study/results/memory-followup-20260927/README.md)
validates 21 full-SQLite budget attempts against native size-1/size-2 SQL
oracles and 12 FFmpeg decoder attempts against the exact frame oracle. At
SQLite size 1, the smallest successful budgets tried are 1.25 MiB Capstone
spatial, 1.25 MiB Capstone + Sublet (2.05 MiB with tables), and 1.125 MiB
PoisonCap spatial (1.39 MiB with tables). The corrected and pressure-reclaim
PoisonCap temporal policies complete at 8 MiB heap (9.11 MiB with tables);
the pressure policy panics at 7.5 MiB in the published kernel. These are
successful selected capacities, not measured minima or total RSS. At size 2,
both spatial arms pass with 2.5 MiB; Sublet faults in SQLite with 2.5/3 MiB
and PoisonCap temporal panics with 16 MiB. The four-arm scaling cell is open.

The fairer FFmpeg PoisonCap adapter copies only stateful `AVRefStructPool`
entries and frees snapshots with their backing. All revised 1/4/16-stream
runs match decoder output. Snapshot backing peaks at 36,288 B and ends at
zero; final jemalloc allocated matches spatial at 1/4 streams and differs by
13,632 B at 16. The original full-copy FFmpeg advantage was an adapter
artifact, not a PoisonCap lower bound. The older result below remains a
record of that adapter's behavior.


## 2026-09-27 — Full SQLite nested-memory pilot and benchmark readiness

The [FFmpeg whole-decoder pool pilot](../../experiments/study/results/ffmpeg-pool-memory-20260927/README.md)
now connects the existing PoisonCap AVBufferPool/AVRefStructPool adapter to the
actual configured 9.0.1 decoder. For 1, 4 and 16 independent 30-frame streams,
all six new PoisonCap mode-0/2 runs and all eighteen existing Capstone
pool-mode-0/2 repeats match the same frame oracle. The PoisonCap temporal arm
retains a 315,072 B snapshot; within-platform jemalloc allocated rises by
294,912–318,336 B over spatial. Capstone's reported outer-heap peak and pool
payload used are equal between its two modes. The ledgers differ across
platforms, and neither includes all kernel metadata. Two plots and per-phase
records are retained; broader FATE coverage and PoisonCap repetitions remain.

The [SQLite 3.22.0 memory pilot](../../experiments/study/results/sqlite-322-memory-20260927/README.md)
now runs all 32 official `speedtest1 main --size 1` phases on Capstone spatial,
Capstone + memsys5-only Sublet, PoisonCap spatial, and corrected PoisonCap
temporal. All four arms match the independent native SQL-result oracle for
4,301 rows, with lookaside disabled. Selected successful application-visible
reservations are 1.25, 2.05, 1.53 and 9.11 MiB, respectively. These omit
platform metadata and are not measured RSS or minimum viable capacities.
The published PoisonCap full-queue path drained six times without revoking;
the corrected path drained and revoked six times at 8 MiB. Its 4.5 and 7 MiB
attempts panic in the published kernel. Figures, attempt statuses, phase data,
and source/binary/raw-log hashes are preserved with the result; raw VM logs
remain outside the repository.
The owned persistent Capstone VM is restored. This pilot still uses the legacy
SQLite domain host, which boots a guest per attempt; full application PoisonCap
adapters for other ports remain future work.

The `application-poisoncap-study` branch adds a [Sublet/PoisonCap memory
study design](../plans/sublet-poisoncap-memory-study.md), a separately pinned
SQLite 3.22.0 artifact catalog and matched-platform planning. Existing CheriBSD
on/off results remain the secondary reference. Generalized PoisonCap execution
qualification stays closed until the shared application runner observes the inner policy,
allocator boundary and all quarantine/revocation paths; outer malloc policy
cannot substitute for that evidence. Fourteen planner and six runner tests pass.

[Upstream mruby lists](../../experiments/study/results/20260927-mruby-lists.json)
passes 4/4 original arms at the full 300 × 10,000 work count with a native output
oracle. Both Capstone arms recover from six allocation failures at a 64 MiB outer
heap limit; these runs need matched backing budgets and GC-slot counters before
memory comparison. No PoisonCap mruby application is claimed.

The earlier published PoisonCap SQLite fork builds after supplying header prerequisites.
[Both workload modes complete](../../experiments/study/results/20260927-poisoncap-sqlite.json)
20 active main phases at size 1 using the preserved published libc and outer
revocation off. The artifact comments out 12 phases; `--verify` does not check
main results. Its full-quarantine drain bypasses the explicit revoker call, and
revocation errors are unchecked. These findings require policy-path accounting
and an independent result oracle, not a security benchmark. This is not a full
32-phase reproduction or a Sublet/PoisonCap memory comparison. The pilot above
supersedes that readiness limit.

## 2026-09-27 — Four-configuration benchmark study foundation

The `application-benchmark-study` branch adds a [pinned candidate catalog and
matrix planner](../../experiments/study/README.md) for six application ports.
Plans distinguish internal-allocator Sublet from outer-malloc Sublet, enumerate
all four arms, lock work parameters, preserve unavailable/failing cells and
resume without silently retrying failures. CheriBSD on/off uses explicit process
switches and checks effective policy at every phase; guest defaults stay intact.

Twelve planner tests and six runner tests pass. Five real FFmpeg/mruby policy
smokes pass, including re-enabling revocation after an off process, in one guest.
The original Capstone VM is restored and idle. These smokes reuse discovery
workloads; no standard benchmark suite is yet qualified across four arms.
The [rollout plan](../plans/application-benchmark-study.md) records recognized
benchmarks, source pins, internal accounting requirements and application gaps.

## 2026-09-27 — Address reuse and post-release application memory

The `application-reuse-metrics` lane studies Cornucopia and Cornucopia Reloaded
and checks twelve FFmpeg/mruby workloads against default CheriBSD purecap.
All 72 paired attempts pass. The complete allocation and allocator phase
samples match old-QEMU controls for all twelve Capstone workloads (108 equal
control/repetition comparisons). The larger-capacity controls do not collect
nodes during application execution; the temporary sweep does not create the
reported differences in these workloads.

Sixty-four FFmpeg streams use 178 distinct starts versus 2,982 (16.75 times fewer)
with 46,720 allocation calls on both platforms. mruby's 512-record, 16-batch case
uses 14,554 versus 44,552 (3.06 times fewer). Full post-release curves show both
retention advantages and the large-retained-graph case where buddy occupancy is
initially higher. No total-RSS, physical-fragmentation or timing win is inferred.
An invalid 128-batch observer-overflow attempt and its interrupted repeat remain
recorded separately. Four analysis guard tests pass. [Results and figures](../../experiments/applications/results/20260927-reuse/README.md)
and [paper analysis and metric definitions](../../experiments/applications/memory-behavior.md)
identify exact scope and the remaining port work.

## 2026-09-27 — Node reuse within a running application

The `runtime-node-reuse` follow-up fixes the six mruby failures below without
raising the 65,536-node capacity or changing application binaries. Under node
pressure, one-hart QEMU now saves the current application at its allocation
instruction, enters the trusted monitor context, clears stale tags and recycles
retired identities, then resumes the same process. Valid or pinned identities
remain unavailable; genuine exhaustion still faults with cleanup headroom.
Previously, the collector was invoked only when the process owner was released.

All 27 original Capstone application repeats and nine extended runs now pass;
the longest mruby run allocates 1,092,495 identities within the fixed pool.
The regression completes
200,000 allocation/free cycles per process, preserves live data and rejects an
old reference after reuse. The full lifecycle gate again passes 1,008 mixed
starts in one boot with stable retained resources. Four native runtime tests
and twelve host CLI tests pass. [Checked results and extended workloads](../../runtime/tests/application/results/20260927-node-reuse/README.md)
identify the exact platform and keep the old failing control.

This is a QEMU software tag sweep, not a new FPGA result or a hardware cost
measurement. The earlier comparison and larger-node controls remain historical
data; default CheriBSD was unchanged and was not rerun for this fix.

## 2026-09-27 — Default CheriBSD application memory comparison

The same FFmpeg 9.0.1 decoder and mruby 4.0.0-rc2 workloads now run on Capstone
Sublet malloc and default CheriBSD purecap, with shared requested-byte and address
reuse counters. No allocator-policy variants or forced drains are used. The
primary matrix has 54 attempts: 27/27 CheriBSD pass; Capstone at 65,536 nodes has
21 pass and six larger-mruby signals. All 12 mruby repeats pass at 262,144 nodes,
recorded separately. All 39 Capstone launches return zero live domains, regions
and bytes. Fifteen instrument/runner tests pass, including failed realloc,
calloc overflow, observer-table exhaustion and false-pass rejection.

Sixteen independent 30-frame streams perform 11,680 allocation calls on either
platform. They use 178 distinct start addresses on Capstone versus 2,803 on
CheriBSD (15.7 times fewer, identical across three repeats). Observed requested
bytes return to zero on both. Capstone occupied blocks return to zero; CheriBSD's
allocated ledger retains 8,590,776 bytes. Capstone separately reserves an 8 MiB
logical pool from a 16 MiB grant plus 1,343,636 bytes of static allocator tables.
Both observers add 1.5 MiB of static address-history storage. These findings do
not establish lower total RSS, bounded in-process node use, or a general
fragmentation advantage. Hardware timings are not measured.

See the [comparison contract](../../experiments/applications/comparison.md).
The paper's `eval/application-memory` branch contains five figures and all 66
attempts under `experiments/application-exploration/results/2026-09-27-default-cheribsd/`.
Only these two applications have matching default-CheriBSD measurements so far;
the earlier six-application discovery below is a separate campaign.

## 2026-09-27 — Application memory workload discovery

The `application-memory-experiments` lane adds one Python build/link adapter and
one persistent-VM runner with real workloads for Perl, CPython, mruby, SQLite,
PostgreSQL single-user, the configured FFmpeg decode app, and prepared tshark
PCAP inputs. Shared SDK atomic/integer helpers and the existing 128-file table
adaptation let these cached application objects use the common launcher.

The bounded discovery records 183 attempts: 147 pass, 21 signal, 3 PostgreSQL
exits at unsupported FileFallocate, and 12 unavailable tshark attempts. All 171
launched cases return with zero live domains, regions and bytes. A separately
recorded 262,144-node configuration lets previously failing mruby cases complete
but larger protected CPython cases still fault near the limit. Collection between
processes is verified; continuous in-process node reuse is not established.

Four native runtime tests, twelve host CLI tests, eleven runner false-pass tests,
known-allocation calibration, 128-file capacity/reuse, and native output oracles
pass. The [workload documentation](../../experiments/applications/README.md)
gives protection scopes and accounting limits. Compact measurements and six plots
live on nested-allocators-paper's `eval/application-memory` branch. These are QEMU
memory/capacity observations, not CheriBSD comparisons or hardware timings.

## 2026-09-26 — Persistent application processes, reclamation and shared SDK

The `domain-process-runtime` lane implements the complete **one-hart QEMU**
application lifecycle across the emulator, monitor, driver, Buildroot package,
Linux launcher and host CLI. Boot once into Linux, then run application ABI v1
images through `capstone-exec`. Faults produce real SIGSEGV; normal exit 139
remains a normal exit. Protected continuations return control even after a
corrupted stack/trap vector or a no-yield loop. Linux signals terminate the
launcher, and final file/VMA release revokes and scrubs its resources for reuse.

The installed-rootfs acceptance passes node exhaustion/recovery followed by
**1,008 mixed starts in the same boot**. Live domains, regions and bytes return
to zero. Cached storage stays at 138,559,488 bytes, live nodes at 67, retired nodes
at zero and tag pages at 648. Cumulative node allocations rise from 65,923 to
77,011 against a 65,536-node pool, demonstrating reuse after stale tags are
removed. This is bounded retained storage, not physical pages returned to Linux.
The test also covers ownership isolation, rollback, fork/dup/VMA lifetime,
overlapping processes, blocked I/O cancellation and transferred Sublet heaps.
See the [checked result](../../runtime/tests/application/results/20260926-qemu-rebased.json).

Four native ASan/UBSan tests and twelve host Python tests pass. A fresh Buildroot
rootfs installs the driver, launchers and Dropbear. The shared CMake application
SDK provides a compiler driver for upstream Make/configure builds; Perl's private
compiler wrapper, entry adapter and VM runner are removed. Fresh Perl 5.36.3 and
SDK-linked mruby pass the common application gate. The current complete Perl
`t/base` run through ordinary `prove` has eight passing files and one failing
file: `base/term.t` test 2 needs a target subprocess, but clone syscall 220 is
unserved. `prove` reports 9 files, 493 emitted assertions and exit 1. The
[rebased-QEMU result](../../ports/perl/musl/results/2026-09-26/base-tests-rebased-qemu.txt)
uses the C-46-corrected compiler and Perl's capability-preserving regex-save
patch. The [earlier controls](../../ports/perl/musl/results/2026-09-26/base-tests-fixed.txt)
isolate the compiler and regex fixes. The
[earlier 6/9 result](../../ports/perl/musl/results/2026-09-26/base-tests.txt)
used an older compiler and a VM CLI that recorded, but did not pass, QEMU
environment settings. The CLI now passes the recorded settings on every boot;
the common SDK rejects a compiler binary with the old linear direct-call bug.
The full upstream Perl suite has not been run or claimed to pass.

Legacy CoreMark, shared-region and the first three HostCall proofs pass on the
new platform. The available legacy snapshot fails all three `null_blk` arms
in `null_submit_bio` (bad address 0x6f) and the borrowed-region file-open-close
proof at INIT (cause 29). Both signatures also reproduce with the old QEMU and
original firmware/rootfs; these are unresolved baseline failures, not passing
regression gates. The managed path uses its own reclamation protocol.

Existing FPGA hardware does not implement the new supervised CALL extension.
There is no new silicon result, full POSIX implementation or hostile-code audit.
The pool has explicit limits (default 384 MiB); the module retains carved physical
storage until reboot. The old and managed driver APIs are mutually exclusive
within a module lifetime. Commands, architecture and remaining port limitations
are in [applications](../../runtime/applications.md) and the
[implementation plan](../plans/domain-process-runtime.md).

## 2026-09-25 — R-42 FLASHED; ladder ACCEPTED, but R-43 false denies block revocation-heavy workloads (R1, P1 cell 6)

**Resident bitstream is `caplifive_r42_6cbdaeeb4.bit`**, sha256 `0cd45bb0…8c05`.
- **The reflash:** it was flashed at 04:16 and `nv_bitstream_name` was read back on the same session
  after the power cycle.
- **Acceptance Boot A, the ladder, PASSED every pre-registered check**
  (`ladder-revival-2026-09-22/r42-acceptance-bootA.result-lines.txt`):
  - `ctrsanity` K=0 1.167× → 1.000×, and `ctrsanitys` K=1 1.210× → 1.045×;
  - 68/68 rows correct, with retval and instret unchanged;
  - the baseline gate reads CPI 1.2000 at every K.
- **The R1-harness acceptance boots, one image per boot**
  (`ladder-revival-2026-09-22/r42-acceptance-r1boots.result-lines.txt`):
  - PASSED: the workload regression (the q0 speedtest1 + m1 drop run, values identical to 054cea69b)
    and the R-43 live16 control (sum 544);
  - the R-35 stale probe still traps cause 25 at +0x4354 (regression check passed);
  - **BOTH R1 release-cost harnesses TRAP cause 25 on their first invocation**, inside `run_series`,
    on loads into the domain's own memory. B3's load is through a capability to a GLOBAL, read from
    the cap table.
  - These are **R-43 false denies of live capabilities**: deny-on-miss after id churn evicts the entry.
    The RTL lane reproduced it in RTL simulation on 6cbdaeeb4 (`r43-evict-live.S`,
    capstone-ariane 93f509f54).

**The lead decided (2026-09-25) that R-42 is the platform of record for the remaining Sublet paper
numbers. That is now BLOCKED by R-43:** no image that churns more than a few hundred revocation ids
can run on R-42.
- **Blocked:** R1 re-measurement, and P1's Sublet cell 6. P1 is held, with E2's own -O0 images ready
  (`e6ee5255`/`ceeded25`, from `xfer/p1-r42-O0-2026-09-25`).
- **Unblocks when:** the R-43 fix (query the rev-node unit on a miss instead of denying) is on a
  bitstream.
- **The -O2 arms** stay blocked on C-32 regardless.

## 2026-09-25 — R-35 CLOSED (fixed on the M-mode LSU data path); R-43 and R-44 filed; R-42 reflash authorized

**R-35 is closed**, decided by the RTL lane on the lead's delegation after an adversarial audit. It is scoped: the fix covers the M-mode LSU check. The same optimistic adopt **still exists at the CPMP**, which is all of S/U-mode enforcement and live in production. That is now **R-44**, deferred on a boot-kill risk (the hardcoded `cpmp(0..2)` ids have no rev-node traffic). The fix's own cost, false denies from deny-on-miss, is **R-43**, whose first test is a SQLite run on the fix bitstream. R-37/R-38's Stage 0 is in the flashed build (fixed in source, not verified on silicon).

**R-42**: bitstream `caplifive_r42_6cbdaeeb4.bit` (sha256 `0cd45bb0…2b8c05`) is hash-verified on apollo and handed to the board lane. The reflash was authorized by the lead 2026-09-25, and the acceptance is pre-registered in `tests/fpga-repros/R42-icache-killed-miss-refill/`. It is a superset of the R-35 image, and its acceptance boot re-runs the R-35 stale probe as a regression check.

**M-1**: unchanged in RTL. Default-built probes still wedge on a fault (both R-35 probes did, 2026-09-24), and that cost R-35's record one attribution. `tests/fpga-repros/RTL-domain-trap-vector-unset/` now has a one-picture summary.

## 2026-09-24 — R-35's fix is on silicon, and the probe that exposed R-35 no longer reproduces it

**ON SILICON (2026-09-24, caplifive_r35_4ad0df694.bit):** the probe that exposed R-35 no longer reproduces it. The same image that read the current occupant's live data through a revoked alias on 054cea69b (is_live_data=1) now traps with cause 25 on the stale read of leaf[0]. A live alias to the same leaf, at the same address, commits without a trap, as do ~43k earlier live accesses, and 17 of 17 ladder rungs return their pre-flash values. N=1 per arm. NOT shown on silicon: WHY the stale read was denied -- cause 25 cannot separate observed revocation from the cache's deny-on-miss, which is the expected route for an id reissued thousands of times; which probe age trapped (k=0 or k=21648); the stale write; and false-deny rates under SQLite. Result lines and limits: the R-35 folder's `results/board-4ad0df694.result-lines.txt`.

**Next for R-35:** a SQLite workload on this image, which is the false-deny test at scale. If the lead wants the denial attributed, a k=43295-only variant paired with a long-idle live alias separates revocation from residency. **R-42** (I-cache killed-miss, performance): fix at capstone-ariane `6cbdaeeb4`, one commit on `4ad0df694`, validated in simulation and synthesized: no new loop, bitstream written (sha256 `0cd45bb0…2b8c05`), routed WNS −10.615 against the resident −8.341. Whether to flash it is the lead's call; see ISSUES.md R-42.

## 2026-09-24 (earlier) — R-35's fix is synthesized and timing-clean at `4ad0df694`: a reflash candidate, not yet on silicon — **SUPERSEDED by the section above**

**R-35 FIX IS SYNTHESIZED, TIMING-CLEAN, AND A REFLASH CANDIDATE (2026-09-24).** `capstone-ariane` **`4ad0df694`** routes at **WNS -8.341, 0.034 ns from the flashed base's -8.307**, with 51.70 % failing endpoints against base's 51.76 %. All five pre-registered synthesis predictions pass. Bitstream sha256 `8db73f8e20244438a2663fef070202e95dde29fe5b1d957b9804a7babf60382c`. *(Flashed since; see the section above.)* Full readings: the R-35 folder's `results/synth-4ad0df694.result-lines.txt`.

**What fixed the timing, in two parts** — `f83fe9342`'s −27.665 had two causes. The rev-node's *combinational* write-request selector drove the cache's write decode; `a87a24a59` registers the fill taps. And the cache's footprint crowded a congested region; `4ad0df694` moves the tag array into distributed RAM. Paths that went *past* the cache recovered ~18.5 ns untouched, which measures the congestion rather than inferring it.

**The correct revocation check turned out to be essentially free.** An earlier claim that it must cost about what Stage 0 cost — because the base was cheap *by being vacuous* — is refuted: Stage 0's 4.593 ns was its implementation, not the price of correctness.

**Next:** put the `.bit` on the console's server-side BITSTREAM store (GUI Bitstream Manager or the board owner — our driver would file it as a boot image), verify its hash, flash (authorized by the lead), and run the folder's board probe. The board lane takes the window after, for its ladder refresh — which doubles as the false-deny test at scale.

## 2026-09-23 — R-35 is fixed in simulation at `f83fe9342`; it is NOT yet deployable, and neither synthesized hash is — **SUPERSEDED 2026-09-24 by the section above**

**Current fix: `capstone-ariane` commit `f83fe9342`, branch `r35-m1-revnode-cache`, pushed.** `f83fe9342` WAS SYNTHESIZED 2026-09-23 AND IS NOT DEPLOYABLE. Area: fixed -- 177,669 post-synth LUTs, -17,423 against the crossbar build with -111 FFs, i.e. the crossbar rewritten as a register file and nothing else. Slack: REFUTED -- routed WNS -27.665, the worst this design has produced, against a pre-registered prediction of roughly -12.9. The worst path is SHALLOWER than Stage 0's (94 vs 121 logic levels, less logic delay) but carries +15.45 ns of ROUTE delay: congestion, absent from every earlier build. Traced at pin level by the synthesis lane: D-cache read-port grant (fan-out 88) -> the rev-node's memory-channel write-request selector, a deep combinational Anvil mux (fan-out 70) -> the cache's WRITE decode. The fix hung a 256-entry write decode off that selector. The read mux is not on the path. Next: register the fill taps, so the decode sees flops.

What follows was written before that result: It replaces the LSU's single core-wide revnode tracker with a tagged 4-way ×
64-set positive validity cache, filled **only** by two passive taps on the rev-node unit's own
node-memory traffic, so an access is allowed only if its exact 30-bit `(generation, index)` is resident
and was last seen live. Acceptance fixture (`r35-rotate-stale.S`): exactly **7 traps**, the three
revoked accesses trap 25, both live-alias controls still return data. Lint at the committed baseline.

**Neither synthesized hash is a reflash candidate:**

| hash | what it is | routed WNS | failing endpoints | routed LUTs |
|---|---|---|---|---|
| `054cea69b` | flashed base | −8.307 | 51.76 % | 168,757 |
| `247b76896` | Stage 0 (tracker reorder) | **−12.900** | 57.25 % | 169,953 |
| `079dc720a` | first cache | **−14.415** | 60.82 % | **192,642 = 94.53 %** |

**The cheap-looking edit was the expensive one.** Stage 0 — 32 lines, zero new signal declarations —
cost **4.593 ns**; the entire first cache cost 1.515 ns more. A before-audit localized Stage 0's cost to
its **LSU** half — *audited, not measured: no build has separated the two halves Stage 0 changed* (the post-adopt value fed `cap_exception` combinationally); every CPMP consumer reads
the registered value, so the CPMP half is flop-to-flop into 16 endpoints. Loops stayed at 1 on every
build: these are path-depth costs, not new cycles. The first cache's LUTs came from describing a
crossbar — a `_d`/`_q` pair rebuilt by dynamic index every cycle, about 90 LUTs per entry.

**All of these costs were invisible to lint:** nine counters at exact baseline and `UNOPTFLAT` unmoved at
40, on every hash.

**The register-file rebuild (`6ee277cc3`) introduced an authority escape — R-35's own class — closed at
`f83fe9342`.** An adversarial after-audit found it with a microtest built from verbatim extracts plus
mutants as positive controls. **The acceptance fixture returned exactly 7 traps both before and after the
fix**: it cannot see same-cycle coincidences. Do not treat a passing fixture as evidence about paths it
does not reach.

**Next:** synthesize `f83fe9342` against pre-registered area and slack predictions (see the plan). The
reflash, and the board run after it, are the lead's call.

**Do NOT:** flash `079dc720a` — at −14.4 ns and 60.82 % failing endpoints a flake would be
indistinguishable from the fix not working. Do NOT revert the CPMP half of Stage 0 to compare the
registered value as a timing "fix" — it admits a persistent false ALLOW (entry adopts Y while a
broadcast names Y; the compare against the stale X misses; the entry vouches for a dead Y forever).

## 2026-09-22 — R-35 registered, and Stage 0 of its fix is ready for synthesis — **SUPERSEDED 2026-09-23: Stage 0 was synthesized and is a 4.593 ns regression, and the "Stage 1+2 / Stage C refill" framing below was replaced by the positive cache. See above.**

**R-35 — a REVOKED capability still reads and writes the storage its object gave up, and the access
does not trap.** Root-caused at `054cea69b` to `load_store_unit.sv`'s single core-wide revnode
tracker, whose adopt arm takes any unseen id as valid without asking the rev-node unit. Reproduced
**two-sided in RTL simulation** with a single-variable arm pair, a witnessed revoke and the faulting
unit attributed by a control that demonstrably fires — strictly better evidence than the board
capture, which sits on a bitstream whose timing does not close. Folder:
`capstone/tests/fpga-repros/R35-revoked-reference-retains-authority/`.

**Now in the registry**, which stopped at R-34 before today: **R-35** (the defect), **R-37** (the
trackers' invalidate/adopt mis-ordering), **R-38** (the CPMP tracker's pre-adopt compare, a permanent
false deny), **R-39** (unguarded index-0 invalidation broadcasts, believed harmless on a stated
invariant), **R-40** (no authorization check on genesis region ids 0-2 — **materially corrected**
after an auditor refuted its first mechanism), **R-41** (the type guard's early return drops the CPMP
entry it read; unproven). **R-36 is a stub**: the number was drafted 2026-09-21 and withdrawn before
filing, and `history/` references it.

**Stage 0 of the fix is implemented, validated and awaiting synthesis.**
`capstone-ariane` branch **`r37-stage0-tracker-ordering`**, commit **`247b76896`**, pushed. It moves
each tracker's invalidate **after** the adopt and compares the **post-adopt** value — the shape
`commit_stage` has always had.

* **Lint: identical to the committed baseline on all nine counters** — LATCH 52 / MULTIDRIVEN 3 /
  ALWCOMBORDER 0 / COMBDLY 0 / UNOPTFLAT **40** / BLKSEQ 2 / UNDRIVEN 25 / UNUSEDSIGNAL 736 /
  ANVIL_UNOPTFLAT 0, over 6,511 lint lines. Two files, 32 insertions, no new signal declarations.
* **Full directed sweep, matched pair, seed pinned both sides: ZERO status changes and ZERO cycle
  drift** (66 PASS / 26 TIMEOUT / 3 NOBUILD / 0 FAILED on each side; 92 comparable tests with
  bit-identical cycle counts). The baseline arm reverted only the two files, in the same worktree.
* **Both halves validated functionally.** The M-mode reproducer is bit-identical, and the CPMP half
  was validated separately on `r12-recl-cpmp.S` — the only fixture that reaches the CPMP gate, since
  that check is gated on privilege *not* being M and the M-mode fixture is structurally blind to it.
  Pre-registered `0 1 1 0 1 0 1` / total 4, matched exactly.

**IT STILL NEEDS SYNTHESIS, and "signal-neutral" was wrong.** Retargeting a compare from `_q` to `_d`
swaps a flip-flop output for a **combinational** node inside two modules that are both in the standing
`UNOPTFLAT` loop set, which no lint counter can see. Per the project rule about feeding a signal into
a cone that already carries a loop, this goes to synthesis before a board or another lane.
**`054cea69b` is NOT an ancestor of the submodule's working-tree HEAD**, so any hash handed on must
name its line.

**Stage 1+2 (the validation broadcast and registry-gated adopt) is NOT started, and its price went
up.** A trap-path enumeration established that the LSU's plain-dereference working set is **~20 ids
and rising**, so registry-capacity eviction can miss *any* id — including the monitor trap stack's,
whose 30-store prologue would then nest silently. So **Stage C (refill) is mandatory rather than
likely**, and it is either a hardware query with a stall (crosses the combinational ring) or a
retryable cause 25 (an ABI commitment). That is a project decision, not a lane's.

**Two instrument facts from today that outlive this work**, both now in the `rtl-sim` skill:
`cva6.py`'s `--sv_seed` defaults to a **fresh random value**, so "zero cycle drift" is unassertable
unless it is pinned on both sides; and `cva6.py` returns **0 for a TIMEOUT as well as a PASS**, so an
exit-status tally cannot classify a sweep.

## 2026-09-18 — Whisper ggml context component

`capstone/ports/whisper/ggml-context/` ports whisper.cpp 1.9.4's real context
allocator through the shared template. A native tiny.en recording contains
60,179 events and 58,192 object allocations; stock and instrumented transcripts
match. Native allocation layout matches both the extracted reference and the
ordinary full ggml library. The recording completes in spatial and Sublet QEMU.
Borrowed-buffer graph objects survive descriptor destruction; reset, owned free
and exclusive owner rebind are distinct epoch boundaries. The README documents
that rebind contract, capability-header capacity adjustment and fixed backing
budget. This is allocator replay, not protected inference or FPGA measurement.
## 2026-09-18 — Opt-in generic client-fault recovery (QEMU)

The [shared runtime](../../runtime/domain-faults.md) provides a domain build
helper, cooperative fault return/quarantine, and a Linux process-termination
policy. Its standalone tests require no PostgreSQL or Sublet allocator sources.
Enable `CAPSTONE_DOMAIN_FAULT_RECOVERY` only with the matching trap-delivery QEMU.
This is launcher-chosen SIGSEGV termination, not monitor-enforced containment,
complete resource reclamation, or a new FPGA result. Allocator integration is
reviewed separately; existing ports are not enabled automatically.
Port navigation and pending integration: [component catalog](../../ports/README.md)
and [integration plan](../plans/port-stack-integration.md). The shared runtime's
missing `include/sublet/sublet.h` is restored from the identical port-branch
header. This repairs a missing build input; the dated silicon results below
and the pending fault-recovery validation remain separate.
## 2026-09-19 — Experimental PoisonCap pymalloc port

The extracted CPython 3.13.7 allocator now has a trusted PoisonCap backend,
reusing the three existing upstream patches and the FFmpeg platform. The full
33-process QEMU suite passes: ABI/platform controls, linked example, five API
checks, nine paired lifetime cases and two native-recording replays. Both
modes process all 115 events and match the native logical oracle; payload
preservation is checked inside the guest. The small complete recording is not
the existing larger workload or the defect corpus.

An initial failed replay exposed stored poison capabilities remaining after
`cclearpoison`, causing a later sweep to revoke a fresh unwritten allocation.
The adapter now overwrites those remnants and counts the additional writes;
a targeted regression and the complete suite pass. The protected recording
uses 63 sweeps and 80,336 bytes each of poison/clear/zero work. Snapshot copy
traffic is zero on this recording; separate in-place realloc controls exercise
it. Both modes report 5,275,200 bytes of private metadata high-water, including
the same authority-record layout and replay scratch. These are not total
memory overhead or hardware timing measurements.

Automatic libc revocation remains explicitly off while adapter sweeps remain
on, using the documented platform workaround. Native tests and all four
backend builds pass. This is not whole-interpreter protection or isolation of
hostile nested managers. [Pilot and provenance](../../ports/cpython/pymalloc/results/20260919-poisoncap/README.md),
[build/link/run guide](../../ports/cpython/pymalloc/host/cheribsd/poisoncap/README.md).

## 2026-09-19 — Experimental PoisonCap FFmpeg port

The published PoisonCap compiler, QEMU and matching CheriBSD kernel/userspace
are reconstructed with pinned sources. The FFmpeg allocator library, direct
example and per-lease adapter build. Seven platform controls pass; ten protected
pool cases pass in a separate fresh guest, including persistent RefStruct state
and stale access after reuse. A 2,379-event native recording matches its complete
event oracle. The first adapter snapshots payloads before poisoning and sweeps
before reuse, with its storage and copy costs counted separately.

The full suite passes all 29 processes when the automatic guest libc-revocation
default is disabled before SSH starts; explicit adapter revocation remains
active. Three successful replays have identical output and counters. Preserving
the guest default instead produces a captured kernel `share->excl` panic in
longer suites, including a spatial-only arm. The explicit guest configuration
is a workaround, not a kernel fix or a Capstone/PoisonCap performance ranking.
[Pilot, failed attempts and scope](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md).

The [backend regression](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md#backend-regression-verification)
rebuilds all four FFmpeg configurations, passes 4 native and 23 shared Python
tests, and repeats the full 29-process PoisonCap suite with an identical replay
report. The spatial CheriBSD replay and its five controls also pass.

## 2026-09-19 — Four CheriBSD allocator libraries and examples

FFmpeg buffer pools, PostgreSQL memory contexts, CPython pymalloc and Whisper
ggml contexts now share a CheriBSD purecap toolchain, build/run scripts and
CMake library targets. Standalone examples and separately supplied client
sources link and run in QEMU. The suite checks the ABI/runtime policy and an
exact bounds fault before running allocator programs. FFmpeg, CPython and
ggml native-recording replays match their logical native oracles; PostgreSQL
passes its four-manager fixture. Native regression tests pass, and all four
Capstone domain configurations still build. These are capability-compatible
component ports with explicitly documented boundaries, not automatic inner
temporal protection in CheriBSD.
[Build/link/run guide](../../ports/common/host/cheribsd/README.md).

## 2026-09-19 — Scattered aliases and ancestor revocation

The synthetic A1 fixture passes 44 QEMU executions: 20 stale read/write
attempts fault after parent revocation, 20 matched no-revoke attempts complete,
and four valid-authority controls complete. Five alias locations are covered
(global, heap object, linked list, independent sibling pool, register), both
immediately after revocation and after same-address reuse. Disassembly confirms
the register alias stays in a register across revocation without calls or
spills. New authority and the unaffected sibling remain usable. This is
functional Capstone evidence, not protected decoding, hostile-manager domain
isolation, a performance measurement or a measured competitor disadvantage.
[Matrix, protocol and provenance](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-alias-scatter/README.md).

## 2026-09-19 — CHERI spatial arena comparison

Nine CHERI spatial replays match the native recordings, with three identical
repetitions per workload. Five companion controls distinguish bounds faults
from ordinary stale pool accesses. The
[three-arm export and plots](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-cheri/README.md)
contain Capstone spatial, Capstone Sublet and CHERI spatial only. Payload
padding and static storage are separate observations; the arms do not provide
equivalent lifetime guarantees or a complete protection-memory ledger.

## 2026-09-19 — Paired FFmpeg replay measurements

The measurement worktree adds a reproducible native-to-QEMU comparison on
three FFmpeg recordings. Eighteen accepted spatial/Sublet points match the
native event sequences, with three bit-identical repeats per workload/arm.
Six companion lifetime controls and four native CTests pass. Failed runner
attempts are retained. These are allocator observations and resource counters,
not application timing or a complete protection-memory ledger. See the
[result bundle](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-replay/README.md)
and [measurement plan](../plans/replay-memory-measurements.md).

Trace-development branch: the current port/runtime/corpus PR heads are combined
in `integration/2-trace-ports`. The [shared trace tooling](../../ports/common/host/port_trace/README.md)
reads all four existing formats and supplies staged-input validation and
versioned result metadata. This is host tooling; allocator replay semantics
and previous QEMU/silicon result identities are unchanged.

Port navigation and pending integration: [component catalog](../../ports/README.md)
and [integration plan](../plans/port-stack-integration.md). The shared runtime's
missing `include/sublet/sublet.h` is restored from the identical port-branch
header. This repairs a missing build input; the dated silicon results below
and the pending fault-recovery validation remain separate.
## 2026-09-18 — PostgreSQL's four Sublet allocator ports

The canonical `ports/postgres/memory-contexts` CMake component ports AllocSet,
Generation, Slab and Bump at PostgreSQL 17.0. Generation/Slab revoke individual
allocations; all four support context reset/delete. Bump retains bulk-only
lifetime semantics. Upstream block policies remain in the versioned patches,
with out-of-band context/block metadata and allocation authority from Sublet.
Native tests, mixed-manager replay and paired exact-access QEMU fixtures are
documented in the component README. Debug manager layouts are explicitly
refused. The original shell builds remain AllocSet-only, and protected
consumer-defect reproduction is still a separate milestone.

Runnable clients are in `ports/postgres/memory-contexts/examples/`: an AllocSet
buffer, Generation message queue, Slab job table and Bump request scratch.
Each source is shared by native/spatial/Sublet executables. All four pass in
all three builds; the examples README gives commands and the client interface.

## 2026-09-18 — PostgreSQL 17.0 pin consistency (before the additional ports)

The PostgreSQL component, original shell builds and native defect corpus now
read one `memory-contexts/upstream.json` pin: 17.0. Five native and four QEMU
CTests pass; both original domain scripts compile and the native defect
reproduces. Historical 17.5 profiles retain their original provenance. The
shared Sublet runtime header required by the CMake ports is restored, with a
configure-time completeness guard. See the component README for commands and
scope: AllocSet is protected; the consumer reproducer's Capstone arm remains
future work.


## 2026-09-18 — Opt-in generic client-fault recovery (QEMU)

The [shared runtime](../../runtime/domain-faults.md) provides a domain build
helper, cooperative fault return/quarantine, and a Linux process-termination
policy. Its standalone tests require no PostgreSQL or Sublet allocator sources.
Enable `CAPSTONE_DOMAIN_FAULT_RECOVERY` only with the matching trap-delivery QEMU.
This is launcher-chosen SIGSEGV termination, not monitor-enforced containment,
complete resource reclamation, or a new FPGA result. Allocator integration is
reviewed separately; existing ports are not enabled automatically.

## 2026-09-18 — Whisper ggml context component

`capstone/ports/whisper/ggml-context/` ports whisper.cpp 1.9.4's real context
allocator through the shared template. A native tiny.en recording contains
60,179 events and 58,192 object allocations; stock and instrumented transcripts
match. Native allocation layout matches both the extracted reference and the
ordinary full ggml library. The recording completes in spatial and Sublet QEMU.
Borrowed-buffer graph objects survive descriptor destruction; reset, owned free
and exclusive owner rebind are distinct epoch boundaries. The README documents
that rebind contract, capability-header capacity adjustment and fixed backing
budget. This is allocator replay, not protected inference or FPGA measurement.

## 2026-09-17 (evening) — CURRENT

> **THE BOARD WAS REFLASHED AND CAPABILITY EXCEPTION DELIVERY IS NOW LIVE ON SILICON.** The reclaimer
> build `054cea69b` is resident, carrying the revoke-walk splice, the node reclaimer **and R-34 plus
> R-24** — the last of those found by reading the build's contents rather than its name, two hours
> before the flash. **Every board result recorded before this reflash is on the old silicon and is
> stale under the standing rule**; re-check before relying on one.
>
> **The monitor fix was a prerequisite and it held.** With delivery live, the old monitor's integer
> add on the stack capability would have faulted inside its own trap handler at every `rdtime` and the
> boot would have looked like a bad bitstream. First boot after the merge: control `retval=4` twice,
> **zero refused-or-trap lines**, all three invocations returned. The pin is `2dcd3a5` in all eight
> drivers and `capstone-bootstrap` carries the fix.
> **`tests/monitor/scan-integer-bases.py` was run on BOTH sides of the bake** — one integer-derived
> base and exit 1 before, zero and exit 0 after — which is what makes "the fix is in the firmware I
> am about to boot" a different and stronger claim than "the fix is in the source".
>
> **The reclaimer does what it was built to do:** 200,000 allocations against a 65,532-index pool
> **without exhausting it**, and both cost curves flat — the ~12× release growth P1 measured and the
> capacity knee are both gone. Bundle at `experiments/results/M1/2026-09-17-reclaimer-pilot`,
> labelled a **platform pilot and not an M1 arm**, because M1's start gate is still one of three.
>
> **The console reports `caplifive_m1_054cea69b.bit`**, taken from a fresh read of
> `flash_state.nv_bitstream_name` rather than from the filename flashed, and confirmed to be a genuine
> gate match rather than the `None`-tolerating branch (zero `BITSTREAM IDENTITY UNVERIFIED` lines in
> the boot log — worth checking, because the drivers set that tolerance and a gate passing on it would
> have proved nothing about the string). All four places that named the old resident are updated:
> `known-good-controls.md`, `bitstream-usability-is-the-census-not-the-slack.md`, the launch doc, and
> the drivers' `FPGA_BITSTREAM` default.
>
> **A REFLASH INVALIDATES TWO CLASSES, AND ONLY ONE OF THEM IS GREPPABLE.** The first is every "the
> resident bitstream is X" line, which a search for the old name finds. The second is every statement
> about **what a board result MEANS**, and those live in documents that need not contain the word
> bitstream anywhere — the capacity bundle published that morning measures a knee that does not exist
> on the silicon now resident. It was caught only because the pilot happened to contradict it, and it
> was annotated rather than left standing. When a reflash lands, re-read the RESULTS of the last
> bitstream, not just the lines that name it.

## 2026-09-17 — CURRENT

> **A bitstream queued for flashing carries two fixes its name does not mention.**
> `caplifive_m1_054cea69b.bit` is the splice-plus-reclaimer build and **also contains R-34 and R-24**.
> Established by CONTENT: `misaligned_ex_q` in `cva6_mmu.sv` goes 0 → 5 against the flashed
> `1bfff7776`, and `DEBUG_REQUEST` goes 24 → 32. That makes capability exception delivery LIVE, and
> the monitor the drivers bake still computes its trap-handler writeback with an integer add on the
> stack capability — a dropped exception today, a delivered one inside the handler afterwards, on the
> `rdtime` path. **The D3 fix (`capstone-sbi` `d3-monitor-capability-writeback` at `2dcd3a5`) must be
> merged and the `4274268` pin moved in eight drivers before the first boot on the new silicon.** The
> memory map does NOT move — both capability-region constants are byte-identical at the two revisions,
> checked independently by two lanes — so the device tree stays valid.
>
> **The re-flash procedure is written down** for the first time
> (`docs/ref/HOW-TO-LAUNCH-ON-FPGA.md`, §RE-FLASHING); it had existed only inside the S-12 repro
> folder. Two corrections could not be applied in the list they correct, because
> `precommit-scan.sh`'s credential pattern matches a phrase committed in that list and blocks any
> diff that touches those lines, as context or as a removal. Both directions demonstrated. The file
> is frozen around that line and the fix is the lead's, since the pattern guards the real credential.

## 2026-09-16 — CURRENT

> **A RETRACTION leads this block.** The cold-regime revoke slope of **87.1 cycles per node is
> WITHDRAWN**: N = 2,048 sits only 1,239 cycles above the warm fit, so the 2,048 → 3,072 segment
> crosses the warm/cold knee and its slope measures the knee rather than either regime. The cold
> figure to cite is the board lane's **silicon** measurement, and it is a **range: 25–27 cycles per
> node walked** — Q4's capacity boot fits 25.45 (R² = 0.999969, eleven points wholly inside the cold
> regime) and the `M1_LIVE` sweep gives 26.96 over a different allocation range. **Both are silicon,
> both on the same machine, and they are not averaged**: two measurements taken over different ranges
> do not combine into one figure, and quoting either alone drops the other's existence. The 3.000 warm
> slope is unaffected.
>
> **AND AN INSTRUMENT RELABEL EVERY LANE NEEDS BEFORE ITS NEXT SIMULATION.** `S12_MEM_DELAY` is **not a
> cycle count**: `stream_delay.sv` has a 4-bit counter, so the value is truncated to its low nibble and
> the knob is a **period-16 sawtooth, not a dial**. The `40` used in all 39 places it appears — and
> described everywhere as "a 40-cycle memory" — realises as **8**. Measured totals on one tree and
> test: define 0 → 708, 2 → 1,415, 12 → 3,427, **16 → 1,004**. Turning it *up* from 12 to 16 turns
> latency *down* to near bypass, so anything ≡ 0 mod 16 asks for a large delay and gets essentially
> none, which reads as a clean negative rather than an obvious failure. **Usable range is 2..15.**
> This is a magnitude label, **not a retraction** — S-12, R-26 and R-34 all stand, dated to the
> revisions they ran at, on an 8-cycle memory. Read the define back from
> `work-ver/Variane_testharness__verFiles.dat`, never from the log.
> **2026-09-14 and 2026-09-15 are NOT in this file**; that block of board work is in
> `current-next-step.md`, whose 09-15 header stands.

* **R-12 is FIXED-IN-SIM on `m1-reclaimer` (`054cea69b`): the reclaimer is built and its approval test
  passes as a pair.** This is R-12's CAPACITY half — the 65,532-node ceiling — and it sits on top of the
  cost half below. Revoked nodes go on a LIFO free list folded into the write that already happens (a
  two-node REVOKE costs 252 cycles before and after); an allocation pops `(generation+1, index)` first,
  **+6 cycles** over a bump, and the allocator's call boundary costs **+1 per allocation** (confirmed at
  N = 65,532: the exhaustion fixture runs +65,542 cycles). Every use of a revocation reference now
  requires `valid && generation == g`. **The approval test is the pair:** on the gen-blind control
  `1ac15c4ef` a retained stale reference destroys the slot's new owner; on `054cea69b`, same tree and
  same fixture, every stale arm is refused with cause 25 and every fresh owner is intact — 8 traps
  against 4. An index retires after 16,384 reclaims rather than wrapping (measured at production width).
  An audit of the finished A5 diff found a composed id reaching a node's LINK, which let a generation
  SKIP past the retirement compare and wrap; fixed at both ends in `054cea69b` and measured as a pair.
  Sweep 65/27/3 with **zero status changes**; lint 733 with **UNOPTFLAT 40 unmoved**. **SYNTHESISED AND FLASHED 2026-09-17** — `054cea69b`, established by a LABEL-INDEPENDENT argument after an initial dispute: a run minting 200,031 nodes from a 65,532-index pool (same on both candidates) and reaching `stop=target` cannot have avoided exhaustion, which is a wedge pre-Part-B and a cause-30 trap after, so the image reclaims and the deployed one does not. **The console exposes no digest of the resident image**, so the cite-by-hash rule is unsatisfiable there; the workable form is hash-before-upload plus label plus a behavioural discriminator in the run. **Unresolved pending the image HASH**, which is the discriminator this project's own rule names; a label is not one. Any silicon measurement attributed to the reclaimer is provisional until then. S1 on `054cea69b`: **WNS −8.307** (best ever on this
  design), **combinational loops 1**, routed Total LUTs **168,757 — 456 BELOW the flashed base** while
  adding the allocator, the free list and the generation check; `capstone_rev_node` 985 LUT / 740 FF.
  All three pre-registered readings met. **Attribution withheld on purpose:** the branch is 14 `core/`
  files from the splice — reclaimer AND the r34-r24 merge — so no reclaimer-specific timing, loop or
  area claim is supported; `f714d2a72` is the only build that would split them and has never been run. The reclaimer is now the design on the board, and R-34/R-24 exception delivery went resident with it through the merge — capability exception delivery is live on silicon for the first time. Box: `ISSUES.md` R-12; note
  `docs/history/16-09-2026_20-30-00_m1-reclaimer-built.md`.
* **R-12's COST half is built, measured and synthesised; the capacity half above now rests on it.** The
  revoke-walk splice (`r12-splice-revoked-nodes`, submodule `f1331daed`, parent `9a5716d5ab6c`)
  unlinks a whole revoked run in two writes at the walk exit, independent of run length. Unspliced
  costs exactly **3.000 cycles per dead node crossed**, spliced exactly **0.000**: 399 / 516 / 481
  cycles at N = 4,096 / 6,144 / 8,192 against 167,498 / 294,367 / 403,353, a **839× ratio at the
  largest rung**. The attribution needs no model — the revoke that *kills* the nodes differs between
  the two trees by exactly +4 cycles at every rung, the revoke that *walks the corpses* by over
  400,000. Synthesis exit 0, WNS **−9.225** against the flashed −12.425, combinational loops 29 → 13.
  **NOT FLASHED**, and the 65,532-node ceiling is not addressed by any of it. Box: `ISSUES.md` R-12.
* **Two warnings travel with that synthesis result, and S2 has now split them apart (2026-09-17).** The
  timing gain is **not attributable to unlinking** — a runtime property cannot move a static loop count;
  the splice put the response send in two branches, so anvil registered the endpoint. **S2
  (`54ac25f97`, the semantically null duplication of that send on the unspliced tree) measured which
  half does what: WNS −8.684, better than the splice's and the best ever on this design, with
  combinational loops STILL AT 29.** So the endpoint registration buys the timing and none of the
  loops; the 29 → 13 drop belongs to the splice commit's structural changes. The earlier sentence
  "nine one-bit registers removed sixteen loops and bought 3.2 ns" is **retracted in its middle
  clause**. The ~642 LUTs outside the unit are likewise not intrinsic to the registration — S2 routes
  307 LUTs BELOW the splice — and remain unexplained. Quote implemented figures against implemented
  totals, never the +133 declared bits.
* **`reports/ariane.utilization.rpt` is overwritten after routing**, so in any archive of a build that
  routed it holds a POST-ROUTE number, not the post-synth one. Measured offset (S2): 171,620 post-synth
  against 169,637 post-route, −1,983 LUTs. This invalidated a post-synth LUT ceiling this lane had
  proposed for S1 — it would have killed S2, which routed fine — and the premise does not survive the
  correction either: the build that FAILED to route projects 163 LUTs BELOW the highest that routed, so
  **LUT count does not discriminate routability on this design**.
* **The sub-8 anomaly is the write buffer, measured rather than inferred.** Exactly two rungs sit
  exactly +91 cycles above the fit, at N = 6 and 7; halving the write-through dcache write buffer from
  8 to 4 moved the pair to N = 2 and 3 — a shift of exactly four — while every rung from N = 8 up
  stayed byte-identical. **The fit `3N + 249` is a bound in NEITHER direction below its range**: it
  over-estimates by 196 / 151 / 60 at N = 2 / 3 / 4 and under-estimates by 91 at N = 6 and 7. A small
  revoke has to be measured, and the answer tracks the buffer depth.
* **The reclamation v2 specification is committed, audited, and does not ship as written**:
  `docs/plans/2026-09-16-revnode-reclamation-v2.md`. The audit found three of its mechanisms
  individually wrong after the author had reviewed it twice. Item 6 — the three sites that grant
  authority without ever consulting a node — is still the fatal one, and Part B (graceful exhaustion)
  is separable and approvable alone.
* **M1's start gate is 1 of 3.** The lead named the RTL lane as runtime/RTL owner on 2026-09-16.
  Approval of the algorithm and of the stale-reference invariant are the remaining two, and cannot be
  given by the owner or the gate is decorative. **Nothing is scheduled and no reclaiming arm may be
  planned until both are given.**
* **The board is `apollo-board`'s; this session is backup, hands-off.** Their P1 no-reclamation
  baseline ran four boots, all `done`, and refuted its pre-registered primary in the informative
  direction: scored apart, `take` is near-flat at ~70 cycles while `give` grows 221 → 1,558 and
  superlinearly in cumulative allocations, against flat-at-16 with slope exactly 0.00 under the
  emulator. Queue, gate order and the lead's open decisions:
  `docs/plans/2026-09-15-consolidated-board-queue.md`.
* **D3 IS CLOSED: the monitor's one plain access through an integer base is fixed and validated.**
  `capstone-sbi` `d3-monitor-capability-writeback` at `2dcd3a5` moves the stack capability's cursor in
  place rather than computing an integer address from it. Validated where it CAN be — against the
  R-34/R-24 delivery-fix branch, because on the deployed bitstream the old form and the new one both run
  clean. Matched pair at `c77c65324`, 601 cycles: the replacement takes cause 0 and its store reads back,
  the capability is bit-identical before and after, the old shape run last takes **cause 24** with its
  store refused, and there is **exactly one trap in the run — the control's**. The replacement is bounds-checked where the integer base was not, so a lower-edge arm tests the one address the preceding `SAVE_REG`s do not (slot 0, `rd` = `x0`): cause 0, value stored. The residual is runtime state — whether the live stack capability's base reaches `frame_base` — and only a boot on a delivering bitstream settles it. The branch is held OFF
  `capstone-bootstrap`, which stays at the commit the drivers pin, so a boot today bakes the monitor it
  expects, and is **now published** — the push allowlist was retired on the lead's instruction on
  2026-09-16, so a task branch no longer waits on a file edit. The hook still blocks shared history,
  deletion, non-fast-forward and `capstone/paper`, which are the four that were ever dangerous. Note found on the way: `RVTEST_PASS` stores to `tohost` through an `auipc` integer base, which
  is the same defect and why ten of the twelve sweep tests time out in their epilogue; exiting through a
  capability minted over `tohost` works. Instrument, readings and note:
  `tests/monitor/`, `docs/history/16-09-2026_15-00-00_d3-monitor-writeback-validated.md`.
* **The stale-artifact trap now fires mechanically** (`a0d83f6b8a14`). A worktree generates its anvil
  output at creation time, so a source patch applied afterwards leaves every local gate testing the
  pre-edit design: a 95-test sweep read as "the change is inert" when it meant "the change is absent",
  and the lint numbers and two cost measurements taken beside it were void the same way.
* **A gate in `board-c6var.sh` could not fail on the variable it existed to pin** (fixed 2026-09-16).
  It demanded an emulator record by its `HEAP` field, which is `sublet_tables_len`, computed from the
  GRANTED ARENA alone — so it witnessed the arena and never the `--tables` grant the boot also sets.
  A record from the committed flow at the 2 MiB arena carries tables 2,523,136 against the boot's
  1,750,285 and prints the identical `HEAP 1344064`: indistinguishable by construction, a clean pass
  against a denominator from another configuration. Arena and tables are now one variable feeding both
  the gate and the boot, and the gate also demands the measure flow's configuration line, which names
  both. Negative-tested: the old gate passed the wrong-tables record, the new one refuses it. Found by
  the compiler lane, verified on apollo, driver half fixed here.
* **Four result bundles on the paper remote were reviewed rather than accepted** (`7e77374aa2b5`),
  including one that keeps all seven entries while recording the loss of its own raw evidence. The
  packaging hole found there generalises to bundles this lane cannot see.

## 2026-09-13 (evening) — EARLIER

> sw64's stall reproduces on an exact redraw and is in the domain's share entry; the mtvec pair
> is CONFIRMED end to end (sw68 fix + sw69 mcause-27 readback); ten collaborator PRs landed (all but capstone-qemu #3). The morning block below
> stands for the bridge and R-30/R-31/R-33.

* **sw66 reproduced sw64's stall exactly** — `F2/share3 → SHA5`, no `SHA6`, no `G/enter` — on the
  manifest-verified artifacts, second draw. Deterministic for this image; per the monitor's own
  marker definitions the hang is inside the domain's share-entry execution with `mtvec = 0`. Its
  baseline arm returned the **size-20 denominator: 54,230,566,323 cycles, hash `3807866 2738af78`**.
* **sw65 said nothing about sw64** (three variables changed; its image had never run on QEMU —
  the runner's QEMU phase had no `cma=` and died before entry, and the image was staged anyway; it
  runs clean there today at `cma=256M`). **The S-15 double delin is not the discriminator**: same
  code in both images, and sw65's passed share3 with it.
* **The pair is proven, not asserted:** `cc55013c2106` from the worktree reflog; the no-flag rebuild
  hit `23da3b126a304585` / `7c27697818b0abe0`; the mtvec image `214b300efd169f03` has its QEMU
  licence. Banked at `~/capstone-artifacts/sw64-pair/`. **sw67 ran it: share3 RETURNS (`SHA6`)
  where sw66 hangs, then `sqlite3_initialize` fails (`0x5117BAD3`, = sw65). S-15's account
  strengthened** — audited as not yet a measured root cause (share3 also differs in region residency
  and size), but its mechanism is now sourced at the RTL commit (DELIN raises on any non-LINEAR
  operand; the only type-sensitive instruction in the branch). The trap word went into the arena
  and sw69 READ IT BACK: `arena0=0xF6C09D13`, mcause field 27 (§7o). Fix on `dev`,
  **proven on silicon by sw68**: the image with the delin removed and nothing else, no trap vector,
  passes share3 and enters where sw64/sw66 hang, and ran size-20 to completion: domain
  **64,732,455,367 cycles**, hash `3807866 2738af78`, **ratio 1.1940** on the 54,214,856,567 baseline
  — inside the pre-registered 1.17–1.27 band. The fix runs the real benchmark end to end on silicon.
* **Boot sw73 (2026-09-14): `main --size 100` on silicon, the default size, both arms complete at the
  native oracle `23674002 573a4409`: native 337,235,381,252 cycles (3.75 h), domain 398,572,346,349
  (4.43 h), **ratio 1.1819 → 1.18**, exactly the pre-registered value (band 1.14–1.24); instruction
  ratio 1.2383 × CPI ratio 0.954. Size series on silicon: 1.2195 / 1.1940 / 1.1819 at sizes 1/20/100.
  Every invalidator checked before the ratio was written; §7p carries the caveats (timing, lookaside off,
  cycles only).
* **2026-09-14, the plan's four boots (§7q):** the Sublet cell re-based on the #3 module (2,794,183,730
  cycles at HEAP 911104; silicon ⑥/⑤ 1.0951), the lookaside-ON arms on silicon (baseline 2,108,202,651 at
  size 1 with 25,122 lookasides), the rounded-arena reclaim completing through csinit (audited: R-33's
  account, attribution inferred), the entry watchdog firing live on sw64's stall, and **`main --size 20`
  with lookaside ON on both arms at 1.1937 → 1.19** (the OFF pair: 1.1940) — SQLite as it ships costs the
  same on this silicon. Two per-boot rules learned the hard way: one Sublet workload per boot (R-12,
  43k rev-node mints per run) and one REGION_ARENA workload per boot (a second 128 MiB arena creation
  stalls the host; mechanism open).
* **Watchdog fixed** (`f7f2c9030623`): liveness is `[uart]` lines; console `[event]` chatter had
  made the entry-stall abort unfireable live.
* **PRs landed on `dev`: ten of the eleven original, and all seventeen new ones (#19–#35).**
  Original: #15, #16 (+print fix), #11/#12/#13, buildroot #2/#3 (submodule `capstone-bootstrap`=`d04bd83`,
  module rebuilt + 3 QEMU controls: the ioctl-struct grew so every board host must be rebuilt with it),
  #17 (+follow-up), #18 (gate PASS=4), #14 (`d616ea4e` + follow-up: the rebuilt 160-test image differs
  by exactly 17 instructions, all `sd ra`→`stc ra`/`ld ra`→`ldc ra`; claim-auditor SUPPORTED; lit 93/93,
  authority 32/32, sqlite-silicon + MicroPython PASS on the rebuilt toolchain). **FPGA-side #3 LANDED,
  proven by boot sw71** (control `retval=4` with the rebuilt controller — its private ioctl struct
  grown, f308efe2 — and the declaring SQLite image running to the size-1 oracle on the new module;
  sw70 before it was VOID: the native baseline controller had been staged into the `lpc` slot). New,
  2026-09-13/14: standalone #20/#21/#28; the nginx stack #29–#35 in order (pool gate 90/0; the
  use-after-destroy pair reproduced, fault at pc 0x166cc; #35's replay not reproducible here — no trace
  file locally, recorded as the collaborator's claim); the MicroPython stack #24→#25→#26→#19→#22→#23→#27
  as plain merges of the collaborator's rebase (0 replayed commits; gates on the tip: default PASS=4;
  full level 557 rows PASS=551 FAIL=1 FAULT=0 SKIP=5; weakref-on 563 rows PASS=552 FAIL=1 FAULT=5
  SKIP=5 — and the gate's `MPY_GATE_TESTS=all` mode cannot fail, its judge compares to the literal
  `all`; recorded in the merge message and the hand-off note). The eight other freestanding hosts with
  the private ioctl struct are grown (B2, 2026-09-14). **Held:** capstone-qemu #3 (rebase); #14/#18's
  force-pushed rebases are to be CLOSED, not re-merged (landed by content).
* **The toolchain binary is STALE** per `toolchain-fresh` (now that #15 lets it say so): the
  `opt`/`llvm-symbolizer` targets were never built. Rebuild at #14's step, never during a suite.

## 2026-09-13 (morning) — EARLIER

> The bridge is discharged. §7f–§7k now carry forward to the flashed bitstream; the 2026-09-12 block
> below stands for R-30/R-31/R-33.

* **THE BRIDGE HOLDS (boot sw63).** §7k's images re-run **unchanged** on
  `caplifive_r30r31_1bfff7776` — verified byte-identical after the bake, from both `overlay/` and
  `build/target/`, which mattered because the overlay was holding the lookaside matrix builds under
  those very names. Every §7k-comparable pair agrees to within **0.07 pp** against the 0.171 pp
  cross-boot band, and `main`, the pair the protocol names, to **0.05 pp** (1.2195 vs 1.220).
  Controls at both ends, **7/7 pairs agree on their verification hash**, `DROPPED 0` throughout.
  **§7f–§7k are carried forward.** Details in §7m. This is the arm sw59 was wrongly reported as.
* **R-33's fix was inert on that boot, and it was proven rather than assumed** — zero
  `not representable` lines. Every constant on the path is a power of two. It would **not** have
  been inert for a `--pool`-derived arena, where both the arena and tables round.
* **The size-20/100 artifacts were rescued from `/tmp`** into
  `~/capstone-artifacts/speedtest1-size100/` with a `SHA256SUMS` and a README. They were
  single-copy, including the only record of their hashes. One set covers sizes 1, 20 and 100 — the
  size is argv and the arena lives only in `HOST_EXTRA_DEFS`, so no rebuild is needed for depth.
* **The `speedtest1` branch is merged into `dev`** (`5e5d9cb42318`). It was on no remote and not on
  the push allowlist. That lands §7l **and `arena-mismatch-gate.py` with its runner wiring** — until
  now a campaign from the main checkout ran with no arena gate at all, and the mismatch it catches
  silently corrupts a ratio instead of crashing a run.

## 2026-09-12 — superseded above for the bridge; current for R-30/R-31/R-33
## 2026-09-18 — CPython allocator component

`capstone/ports/cpython/pymalloc/` adds CPython 3.13.7's actual pymalloc to the
shared port template. Native reference comparison covers normalized allocation
decisions and payloads. A 123,622-event native JSON/regex/bytearray recording
completed in both spatial and Sublet QEMU replays; the 18 paired lifetime cases
passed, with fault verdicts checked at the intended access instruction.
This is allocator replay, not interpreter execution or FPGA measurement.
The component README describes the extraction boundary, fixed backing budgets,
raw fallback, recorder limits and validation commands.

## 2026-09-12 — CURRENT

> Three boots on the R-30/R-31 bitstream. The 2026-09-10 block below is still accurate for the
> firmware half; everything it says about R-30/R-31 being unverified on silicon is now superseded
> by this block. Measurements: `ref/fpga-silicon-measurements-for-paper.md` §4g.1–§4g.5.

* **The bitstream is flashed and verified BY CONTENT.** `caplifive_r30r31_1bfff7776.bit`,
  `nv_bitstream_sha256 = 406e12bf…3b30` read back from a fresh `/api/state` after the mandatory
  power-cycle. Control `k800` = 4 in **all three** boots (sw59/sw60/sw61), `instret` 1089 in every
  one, cycles 4521/4517/4517 — so the flash did not move timing. `known-good-controls.md` is
  refreshed against it; only the `k800` row, the others still carry older bitstreams.
* **R-31 is FIXED ON SILICON** (boot sw60), through the monitor's real share/revoke path rather
  than a fabricated capability: `SHA2:00000003` = cap_type UNINIT where the previous bitstream
  returned LINEAR, and `RCPR` did not fire, so the cursor is at base too. Both halves of the
  contract hold.
* **R-30's headline is SUPERSEDED, and this was over-stated once before being narrowed.** The
  one-byte precondition is fixed — boot sw61 performs **5,334 successful INITs**, every counter
  bit-identical to QEMU. sw60's `RCSH:000006C0` is a *separate* large-region effect one step
  earlier, where INIT refusing is correct. Two of the four candidate accounts are dead: the
  monitor's own arithmetic admits a shortfall of at most 15 bytes (which also kills
  allocator-rounding), and the RTL kills "end moved during the fill". **One account survives: 108
  stores did not advance the cursor.** The discriminator is a pair of boots at two different large
  region sizes — constant 1,728 = a fixed tail effect, scaling = a proportional store-failure rate.
  Not yet run.
* **The silicon allocator matrix is complete.** ABI cost **~1.21**, with the two allocators
  **indistinguishable at this precision** (④/① 1.2124, ⑤/② 1.2107 — 0.17 pp against a 0.171 pp
  cross-boot band, so not resolvable either way; the deterministic QEMU pair *is* resolvable and
  shows lookaside costing marginally less, so "barely depends", not "independent"). The Sublet
  **CONFIGURATION** costs **1.0964 on silicon against 1.0176 on QEMU** — 5.5×, consistent with an
  O(bytes) reclaim. **Not "the discipline":** both pairs carry the heap-geometry mismatch for which
  the QEMU figure was already retracted as a discipline cost, so the 5.5× is suggestive of the
  mechanism rather than a measurement of it. (Both corrected 2026-09-12 after a bench-lane audit.) Two caveats travel with these rows: the ⑥/⑤ comparison
  carries a heap-geometry term (910,008 vs 2,097,152, not equalisable), and the lookaside-ON rows
  must not be blended with the lookaside-OFF §7 corpus.
* **R-33 is a SOUNDNESS issue, ISA-LEVEL, and its cause is the ALLOCATOR.** Demonstrated 2026-09-12 to reach ordinary LINEAR capabilities through `CINCOFFSET` — plain pointer arithmetic — by a matched RTL-sim pair where the representable control does not move and the non-representable arm widens by exactly the predicted amount. The
  rounded `end` is the authority bound — `STC` checks `rs1_up > metadata.end - 16` against the
  decompressed value — so a non-representable region grants writes past itself, and `CINCOFFSET`
  reaches ordinary linear capabilities by the same route. A lossy compressed-bounds format is
  standard and is exact for representable objects; nothing here enforces representability, and
  `create_region(N)` passes `N` through unchanged. **Contained by the kernel's `PAGE_ALIGN` below
  4 MiB and not contained at or above it** — no region used so far escapes (sw60's sits exactly on
  the boundary), but a 130 MiB-class region has a 262,144-byte granule. The over-permissive store is
  **DEMONSTRATED in RTL simulation 2026-09-12** — a representable control's store at its true end is refused OUT_OF_BOUNDS while a non-representable arm's identical store retires without fault (`r33-store-past-end.S`, trap_mask 0x1 as pre-registered). Not yet shown **on silicon**, and the bottom-truncation half is still unexercised.

* **R-30's residual is SOLVED and re-filed as R-33 (boot sw62).** The 1,728 bytes were never a
  failed fill. A capability's bounds are re-encoded once its cursor leaves `base`, and `end` then
  reads high by up to one granule — `compress_bounds` uses an exact form only while the cursor sits
  at the low bound (`ariane_pkg.sv:787`) and otherwise rounds the top up to `2^(E+3)`
  (`:827-828`); `STC` is a DYN op (`decoder.sv:1309`) whose `rs1` is re-compressed on writeback
  (`ex_stage.sv:1188`). sw62's granule-ALIGNED arenas reclaimed clean, while the unaligned arm
  halted with `RCSH = 448` — the compression figure, pre-registered before the boot against 432 for
  a store-failure rate — with `RCCU` showing the cursor reached the true end, so **no store failed**.
  Two accounts were retracted on the way there, both this lane's and the RTL lane's, and both had
  read the fat struct without the function that compresses it.

* **The resident firmware is THREE monitor commits behind and none of them has booted** —
  `75d96d2` (define `CAP_TYPE_UNINIT`), `921f598` (`RCEN`/`RCCU` reclaim instrument), `d1bd7e4`
  (early-clobber on both `C_RECLAIM` outputs). All boots ran `2c49c41`. `board-b59/b60/b61.sh`
  gate on that hash and will now FAIL — correctly; update it deliberately, do not delete the gate.
  The submodule pointer chain is deliberately unbumped, which matters only for a fresh clone.
* **`precommit-scan.sh` had a silent-pass path** — a `--range` git could not resolve contributed
  nothing and printed CLEAN. Fixed (unresolvable *or* empty range now blocks), negative-tested four
  ways. Nothing had escaped it.

## 2026-09-10 — superseded above for R-30/R-31; **2026-09-11 is NOT in this file**

> **Read `state/current-next-step.md` first for 2026-09-11.** Four boots landed that day and none of
> them is described here: sw55 (a **130 MiB** capability region on silicon), sw56/sw57/sw58
> (speedtest1 across seven testsets, the domain's instruction count measured on silicon, and the
> position question settled). The measurements are `ref/fpga-silicon-measurements-for-paper.md`
> §7f–§7k. Also that day: the CMA board half, M-6 fixed and M-7 filed, the R-30/R-31 **firmware
> half committed at last** (all four monitor pointer paths had been committing its pre-reclaim
> parent), and the discovery that **six repositories refuse this credential** — including both
> copies of the monitor and the academic spec, which returns 403 on read as well as write.


* **The reclaim (R-30/R-31 firmware half) is implemented, gated and measured.** The lead ruled
  *fill, then initialise*. Monitor commit `0a5c3d9` adds a `C_RECLAIM` asm loop at **five** sites
  (the annotation branches share a hoisted one, so REV_DEFAULT/BORROWED/SHARED/TRANSFERRED are all
  covered), with the loop bound taken from `cap_end - cap_base` and **not** the cursor, so a
  cursor-at-end arrival faults instead of running zero iterations. Emulator gates green: host-call
  12/12, linear/uninit corpus 7/7, smoke, nullblk 3/3. ~~**`0a5c3d9` is LOCAL ONLY**~~ **CORRECTED
  2026-09-11: it is PUSHED.** `git ls-remote` — the remote itself rather than a cached
  remote-tracking ref — shows `capstone-sbi refs/heads/capstone-bootstrap` at exactly
  `0a5c3d9a3413`. The 403 recorded on 2026-09-10 was real then and was carried forward for a day
  without being re-tried; nobody needs to push this.
* **Boot sw52 — speedtest1 on capability silicon vs native, plus the fill-cost pair.** Control
  `k800` = 4, zero fault tags, 10 of 11 arms. Capability/native **cycle** ratios 1.335 / 1.181 /
  1.214 for parsenumber / orm / main, with **identical verification hashes** on every pair; the cycle
  ratio is BELOW the instruction ratio in all three because capability CPI is *lower* than native's.
  Full numbers: `docs/ref/fpga-silicon-measurements-for-paper.md` §7f.
* **Boot sw53 -- the fill-cost pair with an error bar, and both of §7e's open questions closed.**
  9/9 arms, `k800` = 4, zero fault tags. **24.1 cycles per 16-byte capability store at n=3**
  (within-boot spread 0.2-0.5 %; sw52's single draw of the earlier rung gave 23.6). A 4 KiB reclaim
  is **+3.6 % instructions and 7.8-9.3 % of cycles** at speedtest1's measured CPI, 4.6-26.2 % across
  the full 1.13-6.44 spread -- CPI-sensitive, not "in the lower half of the bracket". An adversarial
  audit had already moved that figure from 5.5-6.6 %: the first version counted the fill loop's
  INSTRUCTIONS but only the stores' CYCLES, and the monitor pays for the loop too.
  - **A warm region is NOT materially cheaper** (`fillwarm`: second pass >= 87 % of the first), so
    the cold-buffer "this is a ceiling" caveat is deleted and the figure applies to the monitor's
    real case.
  - **The cost is per STORE and is ~96 % not capability-specific** (`fillsd`: a plain 8-byte store on
    the same 16-byte walk costs 23.1 against `stc`'s 24.1, writing half the bytes). It is the
    write-through drain, not the tag.
  - **R-3 does not bite the ladder path**: every rung repeated at its own entry VA returned.
  Full numbers: `docs/ref/fpga-silicon-measurements-for-paper.md` §7e, §7f.
* **`RCLM:00000000` on every share (9 of 9).** Zero reclaims, as it must be where the guard cannot
  fire — and the counter's reporting path is now proven readable on silicon, which is the one
  property of it that could not be tested after the flash.
* ~~**Still open and the lead's:** the `end`-convention re-ruling.~~ **RULED 2026-09-10 (evening) —
  adopt the resolution**; `plans/DECISIONS-WAITING-2026-09-10.md:50`. This line said "still open" for
  a day after the ruling landed, which is the same shape as the stale 403 withdrawn on 2026-09-11: a
  blocker asserts a fact about today, so re-verify before repeating it. The spec amendment is
  committed in `capstone-academic-spec` (`21a01f0`, `eadaf87`) and **cannot be pushed** — that remote
  returns 403 for this credential on read as well as write, so its branches cannot even be listed.
* Three tools repaired and negative-tested the same day: `preflight-board-run.sh` (parsed only one of
  the three stage forms, and could not require a host binary at all), the sw52 result parser (needed
  the testset, cycles and hash on ONE line when they are on three), and the driver's staged-marker
  guard (did not know the `BASELINE-PROBE` form, and cost sw52 its last arm).

## 2026-09-09 — SUPERSEDED BY THE ABOVE, still accurate for what it covers

* **The R-25/26/27 bitstream is on the board** (`caplifive_r25r26r27_66c4e7517.bit`, flashed persistently
  2026-09-09; `caplifive_s12fix_5097eb166.bit` stays in the console store as the restore path). Boots sw44
  (closing set 9/9), sw45 (acceptance 7/7, R-25 PROBE wedges as predicted), sw46/sw47 (firmware variants D/E
  clean — the R-26 board arms, no-regression only), sw48 (the s06agg isolation). R-25/R-26/R-27 archived.
  The monitor drops its four R-26 `fence.i` (monitor `1a39e37`, wrapper `f17110a`, buildroot `fe31893`,
  caplifive-system `884b716`; nested pushes need the lead's credential).
* **RETRACTION: S-06's struct-assignment half was never fixed — now `R-29`.** `s06agg` reads 66 on the
  new bitstream in two firmwares (sw46, sw48); the RTL lane's directed test fails on `5097eb166`,
  `ef5a8eaf2` and `66c4e7517` alike when the plain `sd` of the high half is adjacent to the 128-bit `ldc`
  (write-buffer forwarding, S-07/S-10 family). The S-06 folder's cited acceptance (boot B1 "15") was a
  different program. Memcpy half stands. `W-12` stays. Not a regression of the new bitstream.
* Driver classifier: `created`/`entered` now also read the monitor's `DBAS`/`ENT1` tags, so a wedged
  `rtpc`/`lpc` probe is no longer reported as "domain never created" (sw45/sw47 r25dup).


* **The monitor stack is unified onto ONE branch, `capstone-bootstrap`, in every nested repo**
  (caplifive-buildroot `b7fc740`, opensbi `3de3342`, monitor/sbi.dom `3da7ebe`, caplifive-system
  `fec33fa`; parent `dev` `1a1a3dff5778`, all pushed). One source builds both targets:
  `make TARGET=fpga|qemu`, output in `build-<target>/`, `build` a per-checkout symlink,
  per-target code under `CAPSTONE_TARGET_FPGA`/`CAPSTONE_TARGET_QEMU`. `CAPSTONE_CC_PATH` is
  required. The old two-branch split (board vs QEMU, drifted for six weeks and forced Q-03/Q-05 to
  be ported twice) is gone; the `-board`/`-qemu`/`-unified`/`-dts-65536` names are frozen pre-merge
  tips (ancestors of `capstone-bootstrap`, also `pre-unify/2026-09-08/*` tags). Design and record:
  `docs/plans/monitor-unification.md`; layout: `docs/ref/REPO-MAP.md`.
* **Validated on both targets.** FPGA `.c.S` pair, `fw_jump` and `fw_payload .text` byte-identical
  to boot sw31; **board boot sw32 8/8** (control first, six BEEBS rungs, SLT `select1` identical to
  native, zero fault tags), with the SQLite host program rebuilt against the merged loader library
  so that library has now run on silicon; **QEMU nightly tier 18/18**. sbi.dom now builds from the
  one monitor source (Phase B item 7 done).
* **One nightly regression, surfaced and fixed the same day.** `linear-uninit-corpus` /
  `linear_drop_sibling_ok` failed; bisected to the Q-05 monitor commit (a pre-existing stale
  host-observer read the unification's first nightly exposed, NOT a merge defect), fixed in the
  corpus controller to read back through the domain's alias. Auditor-confirmed; a non-blocking
  monitor-robustness note recorded (ISSUES.md Q-05, 2026-09-08).
* **Phase B collapsed the per-target behaviour (2026-09-08, evening and night).** Geometry on QEMU
  (item 3), M-2 bounded at 96 (item 9, refusal seen on silicon), the pre-carve refusal on FPGA
  (item 1), the rounding and the diagnostic store (6a/6b), eleven of fifteen `fence.i` gone (item
  8, three variants booted), the null-blk package and the relocatable S-mode loader (item 4), ONE
  `create_domain` (item 5, nine differences gone) and the transferred slot a hole on the board too
  (item 2: boots sw36/sw37 ran the first transfer-annotated share on silicon, one HOLE line).
  Board boots sw33–sw37 all at the oracles, zero fault tags; QEMU tier 18/18 through item 3,
  17/18 on item 5A (one BEEBS case silent before the loader's first line amid five boot-login
  infra flakes; 3/3 rerun alone, first in a fresh boot). Left: kernel
  unification (item 10, deferred by the lead), the dead `mem_l`/`mem_r` locals, and the
  `gpoff == 0` loader branch that no board image reaches. Monitor 5b27d01 / buildroot d3c2402.
  **Closed on the shipping firmware 2026-09-09: boot sw38, 9/9** (control, six rungs, SLT select1,
  the transfer probe; one HOLE line; zero fault tags). Next: `docs/plans/after-phase-b.md`.

## 2026-09-05

* **SQLite passes its logic tests on silicon at `-O1`** — the first validation above `-O0`.
  `select1` 1031 records / 1000 queries / 0 failures and `q_two` (the S-12 trigger) both completed
  in a capability domain on `caplifive_s12fix_5097eb166.bit` with the cycle-2 compiler, valid
  control first. Full sweep: **8 boots, 8 valid controls, 19 rung readings, every rung at its
  oracle** (`tests/board-results/2026-09-05.tsv`, compiler lane's branch). RV8 `-O2`, CoreMark
  `-O2` with sibling calls, BEEBS `-O2`, two csmith rungs — all at oracle. C-28's tail-call fix runs
  on silicon, so `-fno-optimize-sibling-calls` can be retired.
* **S-12: 6 of 6 post-fix draws clean, p = 0.033** — see ISSUES.md; the two new draws are `-O1`
  and therefore weaker, so this is strong evidence and still not "proven".
* **The gp-captable miscompute (OPEN since 2026-07-23) does not reproduce** — `rc_p1` = 2080 at its
  oracle. Probable cause **R-20, fixed in hardware by `f623c48a1`**, whose signature is exactly
  that bug's. The blocks it carried (silicon-compatibility claim, branch merge, app-level silicon
  perf) are no longer supported by a live failure.
* **R-20 is FIXED in the resident bitstream** — an alert claiming otherwise was filed and
  **retracted** the same day: the fix is a cherry-pick under a different SHA, and both lanes had
  tested ancestry by hash. Presence-by-content is the check; see ISSUES.md.
* **S-13 does not reproduce at `-O1`**, but bitstream and compiler both changed, so it attributes to
  neither yet.
* **2026-09-07:** Q-03 ported to the BOARD firmware (`fw_payload 44c88d9ebeb1`, audited; boot sw30 7/7 in one
  boot, no exact fit occurred so the hole path is unexercised on silicon and self-reporting); Q-05 fixed in the
  stand-in (the probe observes through the domain; both copies make the transferred slot a hole).
* Q-02 (QEMU build break) closed end to end; Q-03 (position-dependent wedge, reproducible off-board),
  R-25 (INIT linearity break), C-41 (compiler `return` encoding), I-01..I-03 filed and verified.

## 2026-09-04 — superseded by the section above

* **Bitstream: `caplifive_s12fix_5097eb166.bit`** (sha256 `7a97ccd0…62999b0`) — the S-12 fix,
  synthesised and flashed 2026-09-04. It IMPROVED timing over its predecessor: WNS −16.400 →
  −15.311, 987 fewer failing endpoints. Every silicon number taken before it should name the
  bitstream it was taken on.
* **S-12: ROOT-CAUSED, FIXED IN RTL, FLASHED — "consistent with fixed", NOT proven.** A capability
  store's scoreboard rd is aliased to its own store-data register; when it stalls on a full store
  buffer the commit stage holds `we_gpr` while withholding `commit_ack`, the WAW guard clears on
  that write, and forwarding hands the consumer `create_cnull()`. The write happens; the
  RETIREMENT does not. Fix = require `commit_ack_i` in both WAW-clearing clauses, four lines.
  Post-fix the SQLite domain completes 4 draws of 4 against a pre-fix arm that trapped 3 of 4 —
  Fisher p = 0.071. **Two more draws would settle it; until then do not write "fixed" unqualified.**
  Full mechanism: `capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/S12-explanation.md`.
* **QEMU is REPAIRED and rebuilt (2026-09-04).** The c128 merge had left `capstone-qemu` unable to
  compile, and because nothing rebuilt it, every QEMU verdict for a day came from a binary dated
  2026-08-27. Three defects fixed in `f5972c364f`; smoke passes and the SLT negative control
  passes, so the comparator is proven able to fire. See `ref/ISSUES.md` Q-02.
  **Two things still do NOT follow from that fix.** The "SLT corpus matches native 15/15" figure
  has no committed harness — it was run ad hoc, so a rebuild does not re-establish it; treat it as
  withdrawn until a re-runnable harness exists. And the nightly still cannot catch a
  non-compiling QEMU. SILICON results were never affected: they came from the board.
* **SQLite CORRECTNESS on silicon: the SQLLogicTest corpus (2026-09-05 → 09-07).** Seven files,
  one boot each, control first, fresh toolchain, every result compared with the native baseline
  from the run's own transcript: negative control and `aggfunc` reproduce their known failures
  exactly (the comparator fires on silicon); `select1/2/3/4/5` identical to native with zero
  divergences — **the whole corpus, 10,807 records, 8,746 checked queries, on silicon.** Rows sw23–sw29 and B8 in
  `tests/board-results/2026-09-05.tsv`; write-up in `ref/fpga-silicon-measurements-for-paper.md` §7c.
* **(Superseded by the line above, kept for the caveat it carries.) SQLite RUNS ON SILICON — that is a LIVENESS result, not a correctness one.** The `slt/`
  corpus executes end-to-end in a capability domain; `s12stress` completes 120/120 prepares and
  15/15 of the corpus matches native under QEMU on the current compiler.
  **Read what that measures.** These files are S-12 *wedge probes*, and they say so in their own
  first lines — `p8_trivial.test`: *"WEDGE PROBE, not a correctness test: expected values are
  dummy, the signal is RETURNED vs WEDGED."* Every table in `s12stress` is deliberately EMPTY,
  because S-12 fires at PREPARE time with no rows processed. So "matches native" is a strong claim
  about **completing without wedging** and a nearly vacuous one about **computing the right
  answer** — the queries mostly return nothing on both sides. Establishing SQLite *correctness* on
  silicon would need a different corpus with populated tables and real expected values, and that
  has not been run. Do not let this line become the citation for a correctness claim.
* **C-19: RESOLVED.** Reading a capability's address now uses a plain move, never `lcc rd, rs, 2`,
  which is not total and traps on an untagged (NULL) operand.
* **The c128 capability value type is MERGED** (external collaborator's branch, 2026-09-04).
  `MVT::c128` replaces i128 as the carrier. Merging it silently reverted C-19 and three header
  declarations; all repaired — see the merge commit. One known coverage gap remains in
  `ptr-diff-signed.ll`.
* **S-06, S-07, S-08: fixed and verified on silicon** (see the 2026-08-16 section below).
* **The debug instrumentation is STALE and expensive.** Every mux reading across the S-12 campaign
  was weak, void or faulted — its own decoder says "UNKNOWN SEMANTICS for this bitstream" — while
  costing 1.820 ns, more than the S-12 fix gained. Every verdict came from software instead.
  `plans/instrumentation-cleanup.md` is now unblocked.

**Next steps are in `state/current-next-step.md` §0.** Sections below this one are retained as the
historical trail; the newest of them is dated 2026-08-16 and predates all of the above.

---

---

## Everything older

The historical trail — the append-only layers from 2026-08-16 back to June, including the S-06 /
S-07 / S-08 bring-up, the R-18/R-19 handovers, the UART retirement and the original overhead
tables — is preserved verbatim in
**`history/04-09-2026_17-00-00_current-state-historical-trail.md`**.

It was split out on 2026-09-04 because this file is the first thing every session reads, and 97%
of it described states that two RTL fixes and a reflash had already invalidated. Nothing was
deleted. When this file and `ref/ISSUES.md` disagree about a defect, **ISSUES.md wins** — it is
the registry; this is a snapshot.
