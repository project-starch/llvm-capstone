# Heap publication and free lookup qualification

Object publication derives directly from a trusted slot parent's absolute
object start. Free retains the index link found during pointer validation and
unlinks through it before entering mallocng. These changes avoid general
derive-wrapper checks and a second hash-chain traversal. Allocation policy,
record size and the public nested derive API are unchanged.

The validating byte load remains essential: the subsequent revoke uses the
record's current issued capability, which cannot validate a stale caller's
capability. Record lookup still starts only from the program pointer's cursor
and checks that the record is public. The retained link is used under the heap
lock, after revocation and before any backend index mutation. Publication's
parent is allocator-owned NONLIN authority; CDERIVE still checks validity,
rights, bounds and representability. See [the contract](../../mallocng.md).

## Inputs and scope

The control is `3702f61342eef89b52ad978b8f8ae38412181930`. Publication changed in
`f6e1e6a8a4cad9439ae0bd2ed5390ac4e67079f3`; lookup reuse changed in
`10304cbd9adcbad012e7a02b33b92c13aa992dcd`, the measured candidate. The compiler
was reused after matching its source identities to
`976df58996f3b7f870061d56ff676aaf6c365856`. Both Capstone SDKs and their benchmark
images were built from their corresponding sources. Unchanged plain RV64
executables were reused with their installed configuration and hashes checked;
both plain arms were run fresh. No compiler or SPEC source was edited.

The census tools are based on `7413d847e172e4bb43bb600c70b87c26e74d769f`.
The private driver shares immutable SPEC sources and data, supports selecting
perlbench for the initial quick profile, and builds only selected benchmarks.
Imagick reference data uses hard links so the guest validator can read it.
The oracles, watchdogs, protection defaults and input hash checks are unchanged.
Perlbench retains the kit's four existing `pack.t` stubs in private run copies,
identically in plain and Capstone arms. Those four checks do not run; the
original source/data tree remains unchanged. Each run records the before/after
hashes in `perl-pack-stub.jsonl`.

QEMU is pinned to `d064e4b060821ae55d641d52a2a64bbada98b695`, with SHA-256
`81799a55c3e2fb0232a3d6b6e020dd5338fde2e721430a85214928da0ea7a9b9` and profiling
plugin SHA-256
`5604b7f0ddcd1e4d7c548aed34c36a2ae27cefd8a24c89f2e2c573bb666093a8`.
This is one-hart QEMU instruction-attempt evidence. Perlbench uses two
supervisor slots in both QEMU and the loaded module; gate arms use 32.
The separately built census modules differ only in their embedded source
filename and GNU build ID: normalizing just those fields makes the complete
ELFs identical. Every other recorded census platform file is byte-identical
between the corresponding arms. The gate platform files are byte-identical.

## Measurements

The unchanged quick profile was recorded before either edit. The canonical
prefix stops at entry 10,000 of `Perl_pp_nextstate`; it has only 28 public frees
and no `S_regmatch` attempts. All three quick pairs return SLICE, not full PASS.

| Quick Capstone prefix | Program attempts | Publication attempts / entry | Slot-release attempts / entry |
| --- | ---: | ---: | ---: |
| Control | 9,318,555 | 59.062 | 33.296 |
| Publication change | 9,310,600 | 48.062 | 33.198 |
| Both changes | 9,308,682 | 48.062 | 4.000 |

Publication saves 11 attempts per entry. Free itself adds 5 attempts as unlink
work moves into it; `checked` adds about 1. The net program reduction is 9,873
attempts. These are exclusive function counts divided by entry-instruction
attempts, not inclusive API costs. The validation label lies inside `checked`
and must not be added to it. Flat memset counts mix several callers and do
not establish the cost of reuse scrubbing alone.

Full perlbench program attempts change from 109,225,276,900 to 108,718,640,188: -506,636,712 (-0.4638% of the control's program attempts).

All seven completed pairs have PASS oracles for plain and Capstone in both
batches, with clean watchdog exits. The corrected paired Imagick control
replaces the two initial harness failures; those failures remain recorded.
A is the control Capstone program and B is the candidate Capstone program.

| Benchmark | Program B - A | Boundary B - A | Total B - A |
| --- | ---: | ---: | ---: |
| 502.gcc_r | -111,736 | -224,672 | -708,807 |
| 538.imagick_r | -37,931 | +263,293 | +836,492 |
| 519.lbm_r | -194 | +1,412,043 | +4,054,425 |
| 544.nab_r | -188,677 | -330,142 | -1,175,525 |
| 500.perlbench_r | -506,636,712 | +308,737,999 | +382,539,768 |
| 505.mcf_r | -2,074,232 | -294,956 | -2,937,065 |
| 557.xz_r | -17,380 | +855,035 | +2,032,115 |

These are instruction attempts. Boundary and idle variation can outweigh
the program changes in total counts; this does not establish a wall-time
speedup. The [qualification data](qualification.json) retain the raw counts,
top program-function deltas, all-window annotations and gate false sets.

In the full run, publication falls from 56.869 to 45.869 exclusive attempts
per entry, and slot release from 35.142 to 4.000. Free rises from 72.999 to
77.999 and `checked` from 40.008 to 41.549 as link tracking and unlink work
move into those paths. Public entry counts match, but internal allocation
counts differ; increased `alloc_slot`, `nontrivial_free` and `memset` work
also offsets some of the saving. The reported result is the measured net
change from one complete pair, not an extrapolation from the quick prefix.

## Protection and compatibility

Both arms ran through the shared gate queue with identical actual platform
hashes. All application images were built from source with the corresponding
SDK. Hosted, runtime, nodes, malloc, pin-ahead, FP, pthread and step-hooks
compare IDENTICAL. Existing failures remain visible: nodes lacks its count
report and malloc's Linux mremap growth/shrink check fails in both arms.

The original memcached comparison is DIFFERENT: the control fails
`fx24_prediction` and the candidate fails `fx24_trigger`. Two complete
alternating replay pairs were declared before replay began. Both compare
IDENTICAL, with all four runs failing `fx24_trigger` plus the same four
pre-existing stderr checks (five failures and 100 passes per arm). The
unchanged control exhibits both fixture-24 failure forms. This is matching
replay evidence, not a claim that fixture 24 passes.

The malloc contract matches in all three configurations: default (49 checks),
lifetimes off (48), and lifetimes/object bounds off (24). Default and spatial
retain the same mremap failure; bounds-off passes all checks. Private negative
SDKs demonstrate that the retained enforcement matters:

| Mutation | Configuration | Positive-true checks made false |
| --- | --- | --- |
| Remove validating byte load | Default | stale-free, reuse-free |
| Remove object revoke | Default | stale, reuse-stale |
| Remove group revokes | All three | coarse-stale |
| Remove reuse scrub | All three | coarse-reuse |

The bounds-off scrub negative also fails header/slack and whole-extent clean
checks. All false sets, including secondary failures, are retained in the
qualification data. No guest thread refusal, kernel panic, Oops or executable
format error was found. The shared compiler's Capstone lit suite reports 147
passes, one unsupported test and the known `shared-patches-present.test`
failure in the exported source tree. The mallocng source-policy guards pass.

## Compiler follow-up

The historical census-6 `S_regmatch` comparison was re-annotated: Capstone has
11,388,349,909 attempts versus plain RV64's 9,359,177,274. The hot plain state
dispatch uses a jump table, while Capstone uses a compare tree. Investigate
switch lowering for `S_regmatch` in `regexec.c` before assigning its whole
2,029,172,635-attempt gap to capability arithmetic. Capability moves, spills
and repeated SHRINK operations are also substantial; flat mnemonic counts
alone do not prove which are avoidable. This qualification makes no compiler
change or wall-time speedup claim.

The immediate cause is visible in
`capstone/runtime/host/capstone_vm/compiler.py`: the SDK driver unconditionally
passes `-fno-jump-tables`. The measured virtual SDK selects
`-capstone-gp-free -capstone-image-gp`. The first experiment should qualify
jump tables under that ABI and measure the state dispatch with the unchanged
oracle. The separate gp-captable ABI's `areJTsAllowed` refusal protects
anonymous tables that no capability reaches; its experimental override is
not a qualification for enabling tables.

control-full: plain 9,359,177,274, Capstone 11,388,349,929, Capstone minus plain +2,029,172,655 attempts.

lane-full: plain 9,359,177,274, Capstone 11,388,349,923, Capstone minus plain +2,029,172,649 attempts.

## Records and reproduction

Records are under p12 `~/lanes/lanePH`: `comparison/`, `annotations/`,
`census/runs/PH-control-full`, `census/runs/PH-control-imagick-fixed`,
`census-B/runs/PH-lane-full`, `gates/`, `matrix/` and `logs/`. The original
Imagick control attempts failed because a host-absolute data symlink escaped
the guest share; both corrected control arms pass with unchanged data bytes.
The original failures and the nonzero original driver exit remain recorded.
Build trees, private SPEC caches, platform copies and staging directories have
been removed. `artifacts/inputs.json` identifies the preserved inputs by hash;
`logs/stage-input-aliases.json` maps staged copies to those artifacts. Profile
metadata retains the original build paths, while the archived executables and
annotations remain available for review.

Source `capstone/tests/capstone-test-env.sh`, rebuild the virtual SDKs from the
two commits, and run `milestone-census.sh` from the stated tools revision with
the stated QEMU pin. Use `prof-compare.py` on each matching pair and
`prof-annotate.py` on every perlbench window. Repeated nested image records
must have identical code placement within each invocation; each raw window
must be counted once. For gates, build the 32-slot module and use
`gate-queue.sh` around `gates/run-arm.sh`; run malloc
with all three configurations and the four mutation classes above.
