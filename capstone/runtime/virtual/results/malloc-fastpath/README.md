# Initial mallocng prefix selection

This experiment removes repeated cursor/base selection of an object's prefix
before `enframe` has moved the slot pointer. It does not solve the allocator's
overall instruction overhead. Results are instruction **attempts** in matched
SPEC test-input slices, not elapsed-time speedups or complete SPEC passes.

Control: `a476237c079ba48931eb1def25a1e8ffd474fff8`.
Candidate: `7c5163dab67c8fd4f68535c32b97ccfaa02fb0a2`.
The compiler is unchanged. The candidate musl archive and SDK were rebuilt;
the compiler's source hashes match the control. QEMU source is
`0542c5556c2daddf71795f3d48eec1fb8ae137b3`.
The earlier `runtime/heap-perlbench` optimizations are absent from this base.

## Change and capability argument

`bind_slots` constructs each parent with its cursor at the slot's base.
`__capstone_malloc_slot_take` returns that parent, after any required reuse
scrub. Consequently the initial prefix is always `cap_s->prefix`.
`cap_start_prefix` expresses that fact for the initial offset read, assertion,
and offset-marker writes. Once `enframe` applies the offset, the general
`cap_prefix` selection remains. The non-Capstone macro retains upstream's
`p - 4` representation.

The allocation policy, offset cycling, prefix/footer assertions, slot index,
record publication, and allocator locking remain unchanged. This avoids
needing a cache of prepared slots: such a cache would need to preserve
mallocng's slot selection and group-release decisions and invalidate every
cached capability when its group ends. That broader change is unproven here.

The following checks and operations remain necessary:

- `checked` first loads a byte through the program-supplied capability. This
  validates its lifetime before private metadata is used. Revoking the
  record's issued child would not validate the caller's stale capability.
- Exact-key lookup and the public-record kind check reject interior pointers
  and invalid frees. No caller pointer becomes allocator authority.
- Object CDERIVE/CREVOKE, group CREVOKE, root confinement, scrub before reuse,
  and the bounds/lifetime measurement switches retain their existing rules.
- Allocator state and SDK-internal entry points have the trust boundary in
  [mallocng.md](../../mallocng.md); this change does not isolate libc statics
  from deliberate access through the image-wide gp capability.

## Measurement method

Raw evidence is retained on p12 under `~/lanes/laneMF`. The quick pair is
`census/runs/{MF-control-q2,MF-step1-quick}`. The deeper perlbench pair is
`census/slices/{control-svclear,step1-svclear}`. Both perlbench arms use two
supervisor slots in QEMU and the module; the gate platform uses 32.

The quick perlbench stop is `Perl_pp_nextstate:10000`. The deeper stop is
`Perl_sv_clear:1000`, across all profile windows: the first invocation
finishes and the next reaches the target. Every accepted run must report
`SLICE` with a clean watchdog. This is not the full census workload that
motivated the allocator investigation. Allocation counts differ between
Capstone and plain musl, so their helper means are not a paired per-request
latency comparison.

`budget.py` sums all instruction windows and uses ELF **FUNC** symbols. The
NOTYPE label `cap_malloc_validate` is inside `checked`; treating it as a
function boundary would incorrectly report only four attempts per call to
`checked`. The program denominator comes from the profile's class ledger.
Function entries are attempts at the first instruction, and all function
budgets below are exclusive of callees. Fault/retry attempts can exceed
successful entries. Mixed-caller `memset` and `memcpy` are excluded from the
heap aggregate; their totals are recorded separately.

`stages.py` partitions reviewed half-open PC ranges in annotated disassembly,
and verifies that their sums cover each complete function. Its addresses
are specific to the retained perlbench images. Unchanged functions may be
relocated; the two `enframe` layouts have separate range files. Do not apply
these ranges to a newly compiled image without reviewing its disassembly.

Reproduce a budget after build cleanup by passing the retained executable:

```sh
python3 budget.py RUN --kind cap --nm LLVM_BIN/llvm-nm \
  --objdump LLVM_BIN/llvm-objdump --exe RETAINED_EXECUTABLE \
  --json budget.json --annotate annotations-cap
python3 stages.py annotations-cap enframe-control.json stages.json
```

`artifacts/inputs.json` on p12 maps original inputs to retained binaries and
SHA-256 hashes. Raw profile and gate records preserve original paths; build
trees are disposable, and those paths may no longer exist.

## Budget before the edit

The initial quick control was measured and recorded before changing source.
It has only 28 public free entries and is strongly weighted toward startup:

```text
cap program_attempts 9498399 heap_attempts 586397 per_public_call 684.24
cap malloc=380 calloc=389 realloc=60 free=28
plain program_attempts 6165833 heap_attempts 207985 per_public_call 171.60
plain malloc=740 calloc=384 realloc=60 free=28
```

The deeper control has 1,634 Capstone public free entries. Output from
`logs/slice-budget-control-{cap,rv64}.log`:

```text
cap windows 2 program_attempts 4539092756
cap malloc=3304 calloc=1384 realloc=991 free=1634
cap heap_attempts 3961602 per_public_call 541.72
plain windows 2 program_attempts 2996561975
plain malloc=4586 calloc=1374 realloc=988 free=1622
plain heap_attempts 1606290 per_public_call 187.43
```

Different helpers have different entry counts. The following means must not
be added to claim an exact inclusive public `malloc` or `free` cost.

| Operation / function | Cap attempts / entry | Plain attempts / entry |
|---|---:|---:|
| mallocng entry | 107.68 | 67.00 |
| `alloc_slot` | 230.72 | 143.25 |
| `enframe` | 137.37 | 63.55 |
| runtime `allocate` | 68.14 | — |
| runtime `publish` | 56.54 | — |
| slot take | 21.20 | — |
| slot key / index insert | 37.07 | — |
| public `free` wrapper | 73.00 | 1.00 |
| `checked` | 38.09 | — |
| mallocng free-slot / plain `__libc_free` | 119.80 | 70.06 |
| slot release / index removal | 33.14 | — |
| plain `get_meta` | — | 48.37 |
| `nontrivial_free` | 73.39 | 57.21 |

The reviewed stages give the following exclusive costs:

| Stage | Cap attempts / corresponding function entry | Plain |
|---|---:|---:|
| mallocng entry, request checks, large path | 21.00 | 17.00 |
| size class | 17.62 | 9.49 |
| backend lock check | 9.00 | 3.00 |
| active group selection | 16.78 | 8.89 |
| available slot and `alloc_slot` handoff | 16.28 | 13.62 |
| unlock, `enframe` handoff and return | 27.00 | 15.00 |
| free-slot entry and stride | 24.26 | 12.16 for plain stride/prefix |
| free-slot index and slot-start handoff | 6.00 | 9.00 for plain entry/get-meta |
| prefix/footer checks | 24.31 | 18.51 |
| prefix invalidation / slot-release handoff | 19.82 | 9.00 for invalidation/masks |
| free-mask/group-release path | 35.41 | 14.39 |
| return | 10.00 | 7.00 |

The group allocator's `alloc_slot` and group creation costs are separate
callees, not included in the selection rows. `group_map` averages 426.96
attempts over 189 entries and `group_nested` 264.39 over 369; both include
inlined slot-record binding work. Their combined totals do not isolate the
binding loop from mapping/recycle policy.

Within `publish`, result-record binding/clean flags cost 20.54, old-child
check 2.00, fields/node budget 21.00, and derive checks/CDERIVE/return 13.00
attempts per entry. There are 4,690 CDERIVE attempts over 4,690 entries.
Slot-key processing costs 12.00 for result/key fields, 10.59 for hash setup,
13.48 for index insertion and 1.00 for return. No old-key unlink path is
executed in this slice.

Within `checked`, entry/return cost eight attempts, caller validation one,
and address lookup plus public-kind checks 29.09. Within `free`, entry and
errno save cost 18, lock 15, checked handoff 5, kind/revoke/clear 9, recovered
slot cursor/free-slot handoff 8, unlock/errno restore 7, return 11. All 1,634
public free entries execute CREVOKE. Index removal costs 22.00 for hash
setup, 5.14 for chain search, 4.00 for unlink and 2.00 for kind/return.

Slot take costs 14.00 for entry/clean flags, 0.20 amortized for scrub handoff
and flag clear, and 7.00 for return. The scrub call site executes 114 times
over 5,822 slot takes. The payload `memset` work cannot be separated from
other callers using these flat PC counters. `CAPSTONE_HEAP_SCRUB_FREE` is
OFF; reuse scrubbing remains active. Extra revokes include 763 in
`prepare_free`, 77 in nested release and 42 in group unmap.

## Candidate measurements and qualification

The deeper slice's API entry counts and all measured heap primitive counts
are identical before and after. Output from `logs/compare-slice.log`:

```text
P_A 4539092756 P_B 4539043843 dP -48913 percent -0.001078 dTotal 235428
RESULTS ['SLICE', 'SLICE', 'SLICE', 'SLICE']
WATCHDOGS [(0, [], []), (0, [], []), (0, [], []), (0, [], [])]
TOP enframe=-48810 strlen=-32 memmove=-28 memchr=-20 perl_parse=-16 memcpy=-5 Perl_sv_grow=-2
```

`enframe` falls from 799,788/5,822 = 137.37 to 750,978/5,822 = 128.99
attempts per entry: 8.38 saved, or 6.10%. The measured heap aggregate falls
1.232%, from 3,961,602 to 3,912,792 (541.72 to 535.05 per public API entry).
Every other heap function has exactly the same attempts and entry counts.
The plain program ledger is identical across replays.

| `enframe` stage | Before attempts/entry | After |
|---|---:|---:|
| Entry and slot record | 21.00 | 22.00 |
| Stride plus initial prefix/slack | 31.37 | 23.25 |
| Slot-take handoff | 5.00 | 5.00 |
| Offset cycle | 13.58 | 13.58 |
| Representable offset | 7.62 | 7.62 |
| Offset prefix | 3.62 | 3.27 |
| Final prefix/footer | 37.19 | 35.27 |
| Slot-key handoff | 7.00 | 7.00 |
| Return | 11.00 | 12.00 |

The compiler moves two slack instructions and saves an extra register; the
combined stride/prefix row avoids misattributing that movement. The change
also permits later instruction reductions without changing their source
checks. [measurements.json](measurements.json) retains exact partitions,
function counts, primitive counts and complete leading program deltas.

Seven quick census results (B minus A), from `logs/compare-quick.log`:

| Benchmark | A program | B program | Delta | Percent | Delta all classes | `enframe` delta |
|---|---:|---:|---:|---:|---:|---:|
| 500.perlbench_r | 9498399 | 9491595 | -6804 | -0.071633% | -233905 | -6842 |
| 502.gcc_r | 25974945 | 25937694 | -37251 | -0.143411% | 153305 | -37263 |
| 505.mcf_r | 969605332 | 969605236 | -96 | -0.000010% | 426695 | -108 |
| 519.lbm_r | 811601323 | 811601296 | -27 | -0.000003% | 1264638 | -39 |
| 538.imagick_r | 2727958 | 2721877 | -6081 | -0.222914% | -777724 | -6087 |
| 544.nab_r | 32536296 | 32523326 | -12970 | -0.039863% | 133002 | -12982 |
| 557.xz_r | 454670514 | 454670339 | -175 | -0.000038% | -115840 | -189 |

All 28 quick arm results are `SLICE`, with watchdog returncode 0, no signals
and no remaining processes. Plain program deltas are zero for all seven.
The remaining leading Capstone changes are small: typically `memcpy` +7
and `memchr` +5; perlbench has `memcpy` +14, `strlen` +8, `memmove` +7,
`memchr` +5, `perl_parse` +4; imagick has `IsGlob` -6; xz has
`spec_mem_init` +2. Total-class deltas include varying kernel/service work
and are not presented as this source change's deterministic saving.

The quick perlbench heap budget is 586,397 -> 579,555 attempts, or
684.24 -> 676.26 per public API entry; `enframe` is 136.01 -> 128.75.
The much smaller deeper-slice program effect shows why neither percentage
can be extrapolated to the full allocator workload.

The repaired nine-gate comparison reports `VERDICT: IDENTICAL`, with all
176 required rows executed. Per-gate row counts are hosted 3, runtime 25,
nodes 4, malloc 39, pin-ahead 21, FP 1, pthread 5, step-hooks 49,
memcached 29. SQLite write/read, mruby and perl actually ran and passed.
All application images were rebuilt from each arm's sources. Platform
hashes are identical, and neither arm logged a one-slot refusal.

This is equality to the existing control, not an all-green gate run.
Both arms fail `node_count_reported`, malloc's
`real_linux_mremap_growth_and_shrink`, and memcached's `normal1_stderr`,
`normal2_stderr`, `usr1_stderr`, `marker_stderr`, `fx24_trigger`.
No gate expectation was relaxed and no replay was needed to obtain equality.

Temporal corpora agree case by case: ffmpeg 13/13, wireshark 10/10 and
memcached 3/3 are `CAUGHT`, with the same fault cause and probe attribution.
Each corpus's bounds and use-after-free controls fault as expected. Both
arms use `-fno-builtin`. The first candidate corpus attempt failed before
launch because its source export lacked the corpus harness; it is retained
separately from the corrected queued run.

The SDK build reports 1,349 compiled, zero failed, `SDK_DONE`, with all
mallocng policy checks passing. The candidate's main, perl and memcached
libc archives all hash to
`44616ef2552353700a943ca730ccc51bede5b7cec4a4012051abb2b76ed9671c`;
control is `4400c1ded2cd358b86e8581f6d877fcb2c50a5f914ab6ac1a327dfc0d4453cd0`.
Capstone lit reports 149 passed, one unsupported, and the known
`shared-patches-present.test` failure. Source exports cannot run the normal
checkout drift check; the matching compiler source hashes are the provenance
check used here. No compiler change is included.

The 14-arm malloc matrix reports `MATRIX_PROOF PASS`. Control/candidate
test maps are identical in default (49 checks), spatial (48), and bounds-off
(24). Bounds-off passes every check; default and spatial retain only the
control's mremap failure. All eight negative heaps fail the required checks:

| Removed protection | Configurations | Required detection |
|---|---|---|
| Caller validation | default | `stale-free`, `reuse-free` |
| Object revoke | default | `stale`, `reuse-stale` |
| Group revoke | default, spatial, bounds-off | `coarse-stale` |
| Reuse scrub | default, spatial, bounds-off | `coarse-reuse` |

Default and spatial positive pairs ran on p13; the bounds-off pair and
negative heaps ran on p12. The copied platform, module, native references,
SDK metadata and images were hash-checked. All arms used their host's global
gate queue. Unstarted waiters were canceled when moving to the shorter
queue; no running arm was interrupted. Matrix logs contain no guest refusal,
Exec-format error, kernel panic or Oops. Full failure sets and input hashes
are in [qualification.json](qualification.json); raw logs remain on p12.
The p12 matrix's initial default control is retained separately as
`matrix-p12-first/control-default`.

## Limits and retained evidence

The deeper slice still spends only about 0.09% of its program attempts in
this measured heap set. It does not reproduce the full perlbench allocator
hotspot. Inclusive per-API costs, caller-specific scrub costs and the broad
full-workload improvement remain **UNRESOLVED**. Flat PC counters do not
supply a caller split of shared helpers. Further work needs a representative
allocation/free-heavy window or caller attribution before changing policy.

The existing profiling kit stubs four pack tests in private perlbench run
copies. Both arms use the same adaptation; `perl-pack-stub.jsonl` records
before/after hashes. This experiment adds no adaptation and does not alter
the original benchmark source. It reports slices, not full SPEC validation.
The initial deeper-slice attempts failed because the output directory was
outside the share and then because `Perl_safesysfree` begins with a branch,
which the stop validator rejects as an ambiguous arrival. Those attempts
are retained; neither contributes a measurement.

The corrected analyzer excludes an interior NOTYPE label and uses the
program class denominator. Earlier quick annotations and the inherited
slice annotation wrapper are superseded by `quick-v2-*` and
`slice-{control,step1}-{cap,rv64}`. Only the latter data feeds these tables.
All 28 quick executable hashes match the retained inputs. Recomputing both
Capstone deeper budgets from retained executables reproduces the JSON
budgets exactly, apart from the executable pathname.

Raw profiles, annotations, logs, result JSON and 184 hashed input artifacts
are retained under p12 `~/lanes/laneMF`; build trees were removed after
qualification. The four p13 matrix results
are retained there and consolidated under p12's `matrix`, with transfer
logs under `evidence-p13`. The final cleanup inventory is `logs/cleanup.json`.
Only results, logs, analysis tools and hashed input artifacts remain.
