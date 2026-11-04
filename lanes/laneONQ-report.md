# ONQ: single-hart unpinned qualification

Status: **UNRESOLVED for default enablement**. Keep ON experimental and
default OFF. The lbm/xz performance target is met, the new controls pass,
and all 176 gate rows execute; memcached fixture 24 still differs from OFF.
SMP is outside this task.

Runtime branch `runtime/unpinned-qualify`, based on fetched `origin/dev`
`a476237c079ba48931eb1def25a1e8ffd474fff8`, in `~/llvm-capstone-laneONQ`.
The pinned model is `0542c5556c2daddf71795f3d48eec1fb8ae137b3`.
No model change was made. All remote writes are under
`~/lanes/laneONQ`. Builds run on p12; gate guests use the shared queue.

## Measurement before source changes

Read the common brief, ONQ task, M5P report, both review rounds, census-7
validation, and landing-gate procedure. The current gate source contains
additional adaptive fault-ahead cases beyond the earlier qualification.

The first completed baseline measurement is the unchanged OFF pin-ahead gate,
run on p12 through `~/retire-gate/gate-queue.sh`, with the production source
and the gate-only 32-slot module default. The private rootfs wrapper verifies
`unpinned=N supervisor_slots=32` after every load. The command was:

```sh
GATES='pin-ahead step-hooks' bash ~/lanes/laneONQ/run-arm.sh off baseline-subset-off
```

Quoted gate summary and failing checks:

```text
pin-ahead  rc=1     41s
FAIL w64_vma_region
FAIL w64_vma
```

Evidence: p12 `~/lanes/laneONQ/baseline-subset-off/gates/pin-ahead.log`
and `pin-ahead/guest/serial.log`. These failures remain failures; the OFF
oracle is unchanged. The remaining OFF subset and the sequential ON subset
are still pending at this initial report entry. No production source or gate
oracle has been changed before recording this measurement.

Preparation reuses the existing source-matched census compiler/SDK and exact
patched kernel read-only. The compiler audit log read back on p12 says
`checked 109961 files; mismatches 0`; this is provenance evidence from that
build, not a compiler rebuild or new source audit by ONQ. The partial export's
normal freshness helper reports `cannot check`; that warning is retained.

Initial disk checks showed 43G free on p12 and 67G on p13. Preparation checks
enforce the 12 GiB reserve. The first platform-copy attempt assumed a
System.map inside images; it lives under kernel. The next attempt assumed
the old landing platform carried platform.json; it did not. Both preparation
errors were corrected in private scripts. An incomplete source export also
omitted capstone/sublet, causing contract, Perl and memcached build failures.
The missing directory was exported from the same base, and those own builds
are being repaired. No failed preparation is a gate result.

## Initial open work

Separate ON oracles with positive fault/admission witnesses; point-1 injection
and failed-resolution recovery; per-page fault attribution; namespace scrub
negative controls; explicit protected-fork refusal; tagged-swap and DMA
tests; full queued OFF/ON gates; recommendation, commits and cleanup.

## Resumed inspection and first measured control

The worktree remains at the requested base, with no published ONQ branch.
The inherited uncommitted module changes are retained: they add diagnostics
for successful unpinned resolutions, touches and counted fault events, plus
the missing point-1 hook and an in-step resolution-deferral fixture. The
existing four candidate module logs are empty; their presence is not build
evidence. The subsequent executed diagnostic gate establishes that its
hooks module loaded and ran. Fresh module rebuilds have explicit output.
No ONQ process was running on either host at resumption. p12 reported 32G
available; p13 reported 67G. No p13 ONQ directory existed.

Readback commands: `git status --short --branch`, `git log --oneline -8`,
`git diff`, `git ls-remote --heads origin runtime/unpinned-qualify`, and
`ssh p12/p13 'df -h ~; ps -eo pid,ppid,etime,args'`.

Read back both completed baseline subset summaries and their FAIL lines:

```text
OFF pin-ahead  rc=1  41s
OFF step-hooks rc=1 106s
ON  pin-ahead  rc=1  38s
ON  step-hooks rc=1 100s
OFF FAIL w64_vma_region
OFF FAIL w64_vma
OFF FAIL adaptres_walk_accounting
```

These are baseline failures, not passes. The ON transcript additionally
fails pin-specific predicates, adaptive-window checks and point-1 injection.
The full raw evidence remains in p12 `baseline-subset-{off,on}/gates/`.
The queue log records completion of both arms. Application build failures
in the initial full-arm preparation remain excluded from qualification.

Fresh lbm/xz OFF/ON quick measurements use the source-matched prepared
compiler, SDK, benchmark executables and production platforms read-only.
The lane owns its platform copies, harness run directories and outputs;
installed benchmark sources and executables are shared without modification.
The copied profiling tools retain their watchdog, stop-boundary and SLICE
checks. Results will be recorded before evaluating the fault-ahead candidate.

## Fresh baseline quick census

Command: `python3 ~/lanes/laneONQ/profile.py baseline off on` on p12.
The command printed eight rows with `rc=0 verdict=SLICE`, then `PROFILE_DONE`.
All runs use one hart. These are work-bounded prefixes, not full SPEC PASS
results or wall-clock speedups. lbm stops at `LBM_performStreamCollideTRT:3`;
xz stops at `sha_compress:13392`.

The following values are the output of the copied `prof-northstar.py` in
`runs/baseline/{off,on}-<benchmark>-northstar.txt`. A is same-batch plain
Linux, B is Capstone; percentages divide by A's program plus boundary work.

| Benchmark | Mode | P_A | K_A | P_B | K_B | Boundary overhead |
|---|---|---:|---:|---:|---:|---:|
| lbm | OFF | 795022912 | 453611585 | 811601092 | 480550928 | 2.157504% |
| lbm | ON | 795022912 | 455221620 | 811704145 | 496183164 | 3.276283% |
| xz | OFF | 441607047 | 545633935 | 454670384 | 612018912 | 6.724293% |
| xz | ON | 441607047 | 545666743 | 454802675 | 628485887 | 8.388670% |

The baseline reproduces the requested regression. Full raw profiles,
platform/input hashes, module parameter readbacks, stop records and top-15
function deltas are retained under p12 `~/lanes/laneONQ/runs/baseline/`.

## First committed diagnostic step

`dd3686f9d32a`: retain the inherited per-event/per-resolution diagnostics,
point-1 injection and demand-deferral fixture. Staged precommit scan was
CLEAN; the absent external exact-name list and skipped URL-file check were
reported by the scanner. The mandatory path-only commit form was used.

The queued `diagnostic-on` step-hooks run printed `rc=1`, as expected with
the original pin-only oracle; no whole-gate PASS is claimed. Direct readback
of `result.json` establishes these point-1 injection/stop counts:

```text
resched1 injections=54 stops=54 violations=0 fault_pending=0
signal1  injections=56 stops=56 violations=0 fault_pending=0
deadline1 injections=55 stops=55 violations=0 fault_pending=0
kill1 injections=1 stops=1 violations=0 fault_pending=0
short64 fault_requests=161 fault_deferred=80 fault_pending=0
```

The first adaptive candidate's ON oracle found a genuine fixture-path bug:
an unsuccessful deferred demand attempt incremented the repeat count, so
its first successful CV_RESOLVE skipped stream training. The candidate now
uses the existing successful-resolution `sfence_pending` state instead.
It also found real budget cuts: w64 stop covered 21 pages; signal4 walked
32 pages using three demand events and two nonempty windows. These are
explicitly observed addresses/windows, not unattributed counter allowances.
Deterministic ON policy checks use the hooks module's full-window fixture;
the direct and injected step cases retain real stop checks. OFF predicates
are unchanged.


## Adaptive fault-ahead candidate

`0a336f5fe8cc` shares AA's stream detection/window policy with the unpinned
resolver. It keeps demand permission checks, arena/VMA bounds, repeated-fault
limits and stop checks, and faults Linux pages without retaining payload
pins. Prefault failures truncate speculation; they do not turn demand
success into failure. `sfence_pending` distinguishes a successfully handled
escape from an unsuccessful deferred demand attempt. Production defaults
remain OFF. All six modules (production, 32-slot plain, 32-slot hooks, for
both kernels) rebuilt successfully; `logs/ahead3-build.log` ends
`MODULE_BUILD_DONE`. The measured ON production module SHA-256 is
`628853ec985d6367db7e3721278ebec1e18788a268af4d5f955353fb4f9baed8`.

Command: `python3 ~/lanes/laneONQ/profile.py ahead3-module on off` on p12.
Again, eight rows reported `rc=0 verdict=SLICE`, followed by `PROFILE_DONE`.
The same stop boundaries and executable inputs apply. `logs/measurements.txt`
and `logs/measurements.json` preserve the extracted numerator/denominator
counts, including harness counts that are excluded from the boundary metric.

| Benchmark | Mode | P_A | K_A | P_B | K_B | Boundary overhead |
|---|---|---:|---:|---:|---:|---:|
| lbm | OFF | 795022912 | 453752564 | 811601096 | 480726293 | 2.160014% |
| lbm | ON | 795022912 | 453376266 | 811601130 | 431043505 | -1.788912% |
| xz | OFF | 441607047 | 545755069 | 454670388 | 611972547 | 6.706504% |
| xz | ON | 441607047 | 545864203 | 454670446 | 546537006 | 0.068134% |

Quoted analysis output:

```text
519.lbm_r candidate_ON_minus_OFF_pp -3.948926175065628 candidate_ON_minus_old_ON_pp -5.06519445882535
557.xz_r candidate_ON_minus_OFF_pp -6.6383698842752095 candidate_ON_minus_old_ON_pp -8.320536447874721
```

The required quick-census target is met for both benchmarks. This is not a
full SPEC result, elapsed speedup, or proof of a hardware latency benefit.
Against the old ON Capstone arm, the candidate's total attempt deltas are
`-66,438,429 (-5.025%)` for lbm and `-82,053,203 (-7.476%)` for xz.
Without the idle loop they are `-65,246,277 (-4.943%)` and
`-82,068,275 (-7.479%)`. Top function B-A deltas from the two retained
`logs/<benchmark>-on-candidate-functions.txt` files:

| Function | lbm attempts B-A | xz attempts B-A |
|---|---:|---:|
| mt_find | -26782204 | -34388216 |
| resolve | -14114184 | -17886038 |
| next_resident | +12562054 | +16146349 |
| vm_ioctl | -12406173 | -15640312 |
| ktime_get | -12178257 | -15643438 |
| __down_read_common | -5666265 | -7279800 |
| cv_step_stop | -5586002 | -7175112 |
| do_raw_spin_unlock | +5115696 | +6588103 |
| up_read | -4739058 | -6088560 |
| do_raw_spin_lock | +4279703 | +5507940 |
| find_vma | -2369184 | -3042026 |
| unpinned_resolve | +2336082 | +2998555 |
| __might_resched | -2267518 | -2915132 |
| __might_sleep | -2061340 | -2650220 |
| down_read | -2060460 | -2647200 |

The main tradeoff is additional resident-PTE probing and spinlock work in
exchange for fewer repeated demand escapes and VMA lookups. OFF's small
boundary variation is retained in the tables; no exact instruction-count
identity is claimed.

## Mode-specific oracles and process refusal

`ab0fdfb76bba` adds separate ON resolution accounting to pin-ahead and
step-hooks, plus transcript mutation controls. `436a9c9899d8` adds ON
lifecycle/CV_TOUCH witnesses and tests the existing clean ENOSYS refusal of
protected fork and process-style clone. Both staged scans were CLEAN, with
the same scanner limitations recorded above. No protected process cloning
or independent child namespace is implemented or claimed.

The old 87 ON failure families are handled as follows:

- Runtime payload-pin release checks become zero persistent payload pages,
  actual demand events, and successful heap/thread behavior. Functional,
  reclamation, denied-access and recycling predicates remain active.
- Pin-ahead compares distinct resolved virtual pages and demand events,
  reconciled with public fault counters and per-arena retirement diagnostics.
  It never labels a resolution as a pin.
- Pthread requires the actual unpinned CV_TOUCH record for the exit word,
  stack retirement and zero payload pins. Supervisor-frame pin balance is
  still checked, and every normal thread test runs after fork refusal.
- Step-hooks retains positive injection/stop/entry predicates. ON additionally
  requires no GUP requests, positive demand-resolution requests, batches of
  at most 16, no fault operation while stop is pending, demand deferral and
  partial-window recovery. Every counted data fault has an address, PC,
  cause, route and per-context sequence number. Injected walk cases must
  cover exactly the 32 region pages through demand plus recorded windows.
  No unexplained extra-fault allowance is used.

The p12 queued `controls-on` arm completed with:

```text
pin-ahead   rc=0 37s
pthread     rc=0 31s
step-hooks  rc=0 67s
unpinned-mm rc=1 15s
```

The raw `gates/*/guest/serial.log` files and JSON checks remain authoritative.
`logs/oracle-mutations.log` reports PASS for all 14 deliberate corruptions:
wrong mode, zero public fault counts, removed event/window records in both
runners; zero injections, entry violations, pending fault activity, oversized
batches, missing deferral and missing partial results in step-hooks. These
are rejection tests against real passing transcripts.

Application preparation is now repaired: `logs/repair-apps.log` ends
`APPS_REPAIRED`. The memcached native dependency initially failed one timing
test, then passed all six individual reruns accepted by its existing build
script. No preparation failure is counted as a passed gate.

## Swap and DMA evidence

The isolated mm fixture calls the current guest's actual nonexported
`arch_prepare_to_swap` entry, obtained from its kallsyms by the trusted root
fixture. Quoted guest output:

```text
RAW_MM acknowledge_refuses_live_tag=1
RAW_MM copied_tag=1 expected=1
RAW_MM tagged_swap=-16 untagged_swap=0
RAW_MM reused_same_frame=1 tag=0
RAW_MM PASS
```

This proves hook refusal of tagged backing and acceptance after scrub/ack;
it does not prove reclaim under swap pressure, an OOM bound, or migration.
The driver and fixtures run only in an isolated single-hart guest without
protected applications.

Model fixtures were exported from exact pin `0542c5556c2daddf71795f3d48eec1fb8ae137b3`.
The first compile attempt lacked a bootstrap include and failed; the include
was exported from the same pin before retry. `logs/model-inputs.sha256`
records the inputs. The private runner only enables raw moves on the
existing tag-writer CPU options. Rule-1 arms 0, 1, 19, 20, 23, 24, 25, 26,
28, 32 and 33 report **11 passed, 0 untestable** in `logs/dma-rule1.log`.
They include real fw_cfg/virtio/aplic DMA paths and tag-retaining controls.
`logs/raw-dma.log` reports
`PASS raw_write_mapped_refusal_preserves_bytes`. SMP arms were excluded.
No model source change was made.


## Attribution of the five accounting cases

The old report named `resched2_syscalls`, `signal1_walk`, `deadline1_walk`,
`resched4_walk` and `deadline4_walk`. Its archived run has no per-page ON
census, so the five historical one-fault deltas cannot be retrospectively
assigned a PC/address. That historical attribution remains UNRESOLVED; it
is not explained away as scheduling noise.

The current candidate emits an exact record for every increment of the
public counted-data-fault total, independent of successful resolutions.
`logs/fault-attribution.json` retains complete records for these five cases
and their base1 references from the p12 `controls-on` arm. The command's
summary output is:

```text
base1:syscalls public=24 events=24 region_events=0 region_pages=0 outside=24 causes={15: 17, 13: 7}
resched2:syscalls public=24 events=24 region_events=0 region_pages=0 outside=24 causes={15: 17, 13: 7}
base1:walk public=56 events=56 region_events=32 region_pages=32 outside=24 causes={15: 49, 13: 7}
signal1:walk public=54 events=54 region_events=30 region_pages=32 outside=24 causes={15: 47, 13: 7}
deadline1:walk public=50 events=50 region_events=28 region_pages=32 outside=22 causes={15: 44, 13: 6}
resched4:walk public=18 events=18 region_events=2 region_pages=32 outside=16 causes={15: 12, 13: 6}
deadline4:walk public=18 events=18 region_events=2 region_pages=32 outside=16 causes={15: 12, 13: 6}
```

All five current cases reconcile public totals, numbered events and arena
retirement counts exactly. The walk's remaining pages have explicit
prefault-window witnesses; syscall mode has no test walk arena. Events
outside the region name their own image, stack or exchange arena and PC;
they are subtracted explicitly, never inferred from a pin count or allowed
as a fixed margin. No current counted fault is unattributed. Instruction
fetch faults in the RX image prefix remain outside this pre-existing
public data-fault statistic.


## Full queued gate arms

On p13, `full-gates.sh` runs control OFF, candidate OFF and candidate ON
sequentially, each through the shared queue. The first two arms completed
at 03:50:13Z and 03:56:16Z. The same repaired application binaries, launcher,
QEMU pin and gate sources are used throughout. OFF uses the stock kernel;
ON uses the source-matched patched kernel. OFF test CSRs are enabled; ON
runs with them disabled and raw moves enabled. The private rootfs wrapper
prints and verifies the loaded unpinned bit and 32 supervisor slots.

The unchanged comparator's OFF output includes:

```text
hosted     IDENTICAL  rows ran 3/3
runtime    IDENTICAL  rows ran 25/25
nodes      IDENTICAL  rows ran 4/4
malloc     IDENTICAL  rows ran 39/39
pin-ahead  IDENTICAL  rows ran 21/21
fp         IDENTICAL  rows ran 1/1
pthread    IDENTICAL  rows ran 5/5
step-hooks IDENTICAL  rows ran 49/49
memcached  IDENTICAL  rows ran 29/29
VERDICT: IDENTICAL
```

Thus all 176 rows executed in each OFF arm, including repaired
SQLite/mruby/Perl and memcached. An identical failure set is not a whole-suite
PASS. The initial p12 OFF subset's timing-sensitive window/accounting failures
are retained; on p13 both full OFF arms pass pin-ahead and step-hooks.
Evidence: `full-{control-off,candidate-off}/gates/`,
`logs/off-comparison-early.{txt,json}` and final `logs/off-comparison.{txt,json}`.


The full ON arm completed at 04:01:29Z; the comparator reports all 176 rows
executed, eight gates IDENTICAL and memcached DIFFERENT. Full summaries:

| Gate | Control OFF rc / seconds | Candidate OFF rc / seconds | Candidate ON rc / seconds |
|---|---|---|---|
| hosted | 0 / 15 | 0 / 15 | 0 / 16 |
| runtime | 0 / 116 | 0 / 120 | 0 / 113 |
| nodes | 1 / 19 | 1 / 21 | 1 / 19 |
| malloc | 1 / 27 | 1 / 28 | 1 / 27 |
| pin-ahead | 0 / 16 | 0 / 16 | 0 / 16 |
| fp | 0 / 8 | 0 / 8 | 0 / 8 |
| pthread | 0 / 17 | 0 / 17 | 0 / 16 |
| step-hooks | 0 / 36 | 0 / 37 | 0 / 36 |
| memcached | 1 / 45 | 1 / 47 | 1 / 45 |

The unchanged baseline failures are `node_count_reported`,
`real_linux_mremap_growth_and_shrink`, and memcached's four stderr checks
(`normal1`, `normal2`, `usr1`, `marker`). Fixture 24 is different:

```text
OFF: RETURN detail=18e0018; fx24_prediction=true, fx24_trigger=false
ON:  FAULT-ELSEWHERE detail="untagged access 0, target 3f8c5fd73e";
     fx24_prediction=false, fx24_trigger=true
```

ON's raw output establishes that the header page moved, the intentionally
reversed header-only path ran once, and the orphan target was
`3f8c5fd73e`. The actual trap is cause 24 at PC `3f9622b134`, address zero.
This does not satisfy the fixture's predicted RETURN result or establish a
fault at the intended target. OFF's `18e0018` mark means its trigger was not
created, also a failure. Neither outcome is converted into a pass. Whether
the ON difference is an allocator/timing-sensitive fixture problem or an
underlying defect remains UNRESOLVED. This blocks the requested full parity
qualification. No memcached source or oracle was changed.

Evidence: p13 `full-candidate-on/gates/memcached/fx24/`, all three
`gates/memcached/result.json` files, `logs/on-comparison.{txt,json}` and
`logs/full-gates.log` (ends `FULL_GATES_DONE`). Pin-ahead reports 65 ON checks
passing; step-hooks reports 429 ON checks passing. All normal application
rows run, with no rejected-image or missing-executable substitute for a test.
The full nine-row arm predates only an added argument-validation guard for
the optional `unpinned-mm` gate; that guard is verified by the host checks
and the later final mm arm. It does not affect any of the nine default gates.

## Fresh-namespace negative control: fixture repaired and passing

The first fixture used ioctl numbers 1 and 2. Number 2 collides with the
VFS's FIGETBSZ request: Linux returned filesystem block size into the sample
instead of invoking the probe, leaving the old tag field unchanged. The
initial positive failure was therefore a fixture defect, not evidence of a
scrub failure. A temporary diagnostic module established `summary=0` after
scrubbing the same physical page, while the faulty client still printed 1.
Its subsequent direct-clear request failed because FIGETBSZ had corrupted
the address field. Both failed raw arms and the diagnostic remain retained.
No diagnostic printk was added to the production source.

`e70be1d3f269` uses a shared private header with encoded ioctl requests and
clears raw register cargo before restoring interrupts. It also adds the
swap-hook fixture and optional queued mm gate. All checks pass against the
unchanged final candidate module in p12 `mm-final`, queue DONE 04:06:00Z:

```text
MM_MODE:Y:N
NAMESPACE tag_before=1 tag_after=0 bytes_equal=1
NAMESPACE frames=8232d000:8232d000
NAMESPACE direct_scalar_clear=1 bytes_equal=1
NAMESPACE PASS
NAMESPACE_EXIT:0:0
NAMESPACE_SWEEP_PAGES:1
MM_MODE:Y:Y
NAMESPACE tag_before=1 tag_after=1 bytes_equal=1
NAMESPACE frames=8230e000:8230e000
NAMESPACE direct_scalar_clear=1 bytes_equal=1
NAMESPACE FAIL
NAMESPACE_EXIT:1:1
NAMESPACE_SWEEP_PAGES:0
unpinned-mm rc=0 5s
```

The deliberately printed FAIL is the required bypass-negative outcome.
`mm-final/gates/unpinned-mm/result.json` records all seven checks true and
hashes of every probe and the candidate module. Normal close/open removes
the tag while retaining both representation words; bypass preserves it and
fails that same oracle. Direct scalar clearing independently succeeds.
The negative bypass parameter exists only in CV_TEST_HOOKS builds.

The p13 diagnostic ticket was withdrawn while WAIT, behind another lane's
full arm, before it ever ran. The diagnostic and final mm arms then ran
sequentially through p12's idle queue. No other lane's process was touched.
The own-ticket queue records are retained in each host's `logs/queue-own.txt`.

`logs/host-checks.log` ends `HOST_CHECKS_PASS`: Python compilation, shell
syntax, invalid-memory-mode rejection and missing-probe rejection passed.
It retains the source-export freshness warning rather than calling that
warning a successful toolchain audit.

## Recommendation and remaining work

**Do not enable ON by default.** Task 0 meets the measured performance target;
ON mode-specific resolution/lifecycle oracles, positive point-1 injections,
failed-demand recovery, current per-fault attribution, clean protected-fork
refusal, fresh-namespace positive/negative controls, actual tagged-swap-hook
refusal and tested DMA rule-1 paths now have executable evidence. OFF has
identical failure sets to its control. All repaired application rows execute.
The full ON suite nevertheless differs on memcached fixture 24.

Remaining qualification work:

- Explain/fix the ON fixture-24 null-access outcome and make its intended
  trigger reliable; retain the current failing oracle until then.
- Exercise reclaim/swap and migration pressure. Non-present/migration PTEs
  still cause safe collection/open refusal; node growth can continue without
  recycling unswept IDs. This is an availability limit, not default readiness.
- The five historical fault addresses are irrecoverable from their old logs;
  current replacements account for every counted event. No retrospective
  attribution is fabricated.
- Shared tagged backing, protected process cloning (beyond the tested
  refusal), a whole-system leak/OOM bound, SMP, KSM and huge pages are not
  qualified here. Single-hart model counts imply no hardware timing result.

No kernel patch, compiler, benchmark source or model source changed. No
model branch was needed. The implementation uses existing Linux fault and
COW handling, bounds every speculative window, preserves stop/repeat checks,
and retains the exclusive-namespace, retirement and collection rules. Passing
performance slices are not used to override any unresolved safety evidence.

## Commits and evidence handling

The five implementation/test commits are `dd3686f9d32a`, `0a336f5fe8cc`,
`ab0fdfb76bba`, `436a9c9899d8` and `e70be1d3f269`; the final documentation
commit records this report and the updated unpinned README. Before publishing,
the four initial private commits were recreated with a neutral project author
identity, each with a fresh staged scan and the required path-only commit
command. The final tracked file tree matched exactly. Every commit's scan
was CLEAN; the scanner's unavailable exact-name list and deliberately skipped
URL-file check remain explicit limitations.


Final cleanup ran only under this lane's directories. The process audits
reported `{"active_lane_jobs": []}` on both hosts. All executed ONQ queue
tickets are DONE; the single p13 diagnostic WAIT was explicitly withdrawn
before execution, then replaced by the completed p12 diagnostic/final arms.
No ONQ gate remains running or queued.

Removed own module/probe/application builds, platform and profiling-platform
copies, source exports, adapter copies, benchmark run scratch, temporary
model binaries/disks, VM state, test staging executables/disks and build
symlinks. Shared compiler/SDK, kernel, benchmark sources and every other
lane's files were untouched. `logs/cleanup-record.json` records removals;
`logs/cleanup-process-audit.json` records the empty process audit. The cleanup
script is retained under `logs/reproduction/`.

Kept result/serial logs, JSON verdicts and hashes, quick-census runs and
profiles, failed-attempt evidence, queue records, model/build logs and
reproduction scripts. Raw model scratch logs were copied to
`logs/model-evidence/`; VM console logs to `logs/vm-console-evidence/`;
application build logs to `logs/application-build-evidence/`. Scripts and
platform manifests moved into `logs/reproduction/`. Scans are retained in
`logs/precommit/`. Any recorded command pointing into removed build/source
scratch must recreate that scratch before rerunning; it is provenance, not
a claim those binaries remain installed.

Quoted cleanup output:

```text
p12: CLEANUP_REMOVED 100; available 45G -> 51G; retained lane directory 121M
p13: CLEANUP_REMOVED 76;  available 69G -> 70G; retained lane directory 169M
```

The subsequently removed empty p13 withdrawn-job directory/build symlink
and the root cleanup-script copies are excluded from those removal counts.
Publication is limited to branch `runtime/unpinned-qualify`; neither dev nor
the model's shared branch is changed.
