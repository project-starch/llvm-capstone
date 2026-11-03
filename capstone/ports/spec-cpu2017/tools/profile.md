# Profiling SPEC on the virtual profile: instruction attempts, not seconds

## What it measures

`tools/profile.sh` runs one benchmark at one size in one arm exactly as `tools/run.sh` does
(runcpu validates it in a freshly booted guest), with two differences: the guest's QEMU carries
capstone-qemu's `contrib/plugins/capprof.c`, and every guest invocation runs under `profrun`
(`tools/capprof-validation/profrun.c`), which opens a capprof window, forks and execs the
invocation, waits, and closes the window. The plugin counts, per instruction address, how many
times the guest ATTEMPTED that instruction inside the window: an instruction that faults and is
resumed counts each time, a translated-but-not-executed block counts nothing. `prof-report.py`
attributes every counted address to a layer (program, launcher, profrun, kernel, module,
firmware, other) and to the containing function, and checks the instruction bytes against the
image they are attributed to. `prof-compare.py` gives the signed per-function difference of two
runs: the same arm twice (noise), or two arms of one benchmark (where the instructions go).

An attempt is neither a retired instruction nor a cycle. A 16-byte capability load and an
`addi` each count one, and software that exists only because the processor is emulated (the
supervisor quantum's round trips, the module's page pinning) counts as much as anything else.
The counts therefore rank the SOFTWARE STACK's instruction expansion -- compiler output, libc,
SDK runtime, launcher, module, kernel work done for the program -- which is what a software
optimization changes. They do not rank hardware cost; a hardware claim needs the cache model
(the 2026-10-20 cachesim lane) or silicon.

## Why counts, and what they buy

A SPEC test input takes minutes to hours in QEMU (perlbench 3.4 h, x264 3.6 h, the kit's
qualification of 2026-10-17), and a timing must be repeated on pinned, quiet cores. Attempt counts
are a property of the instruction stream, so a run's wall-clock and the host's load do not
change them. Every benchmark and arm runs at once (`tools/profile-batch.sh`), unpinned.

Reproducibility: the guest runs with `-icount shift=0` (guest time follows the instruction
count, so timers and the supervisor quantum, a QEMU_CLOCK_VIRTUAL timer, fall at the same
instructions every run), address-space randomization off, syslogd/klogd stopped. The validation
programs then count IDENTICALLY run to run in their user-level windows; what still varies is
guest-kernel work driven by host-timed events (virtio completions, the SSH session), which
lands inside a window or outside it. On the validation set that was 0 in five of seven windows
and tens of thousands of kernel attempts in a window that pins pages (`vt3`/`vt4`, 2026-11-02:
355,638 vs 403,588 in the fault test, all of it kernel). Report that floor beside any difference
near it; `prof-compare.py` on two runs of the same arm measures it for a benchmark.

## Arms and fairness

An arm (`config/profile-arms.example.json`) names what it measures: for the Capstone arm the SDK
(so the compiler, musl-capstone, the heap and runtime policy), the module variant and the CPU
properties; for the plain arm the compiler driver built from the same llvm-capstone source with a
pristine musl 1.2.5 (`config/riscv-plain.cfg`). Both arms compile the same sources with the same
src.alt packages and the same -O2 and portability flags, and both use the F and D instructions
with the lp64 convention. They differ in the libc (musl-capstone's patches), the SDK runtime, the
launcher and module, and the capability code the compiler emits: the comparison is the whole
software stack of the virtual profile against a plain static Linux program, not the capability
ISA alone. A function present in one arm only (an SDK routine, a libc replacement) appears in
the comparison as unmatched, not hidden.

The window is system-wide: everything the guest executes between the markers counts, including
the kernel's work for other processes. The guest is quiesced (one vCPU, no daemons but dropbear),
and the report's `kernel` and `other` classes are read with that in mind.

## What a run records

`OUT/record.txt` (the arm's JSON, platform, icount, memory, share model), `inputs.sha256` (QEMU,
images, adapter, module, plugin, profrun, System.map), `qemu-command.json` (the exact command
line, CPU properties included), `module-sections.txt` (the module's section addresses, the
kernel's command line, the module parameters, ASLR state), `invocations/NNNN.json` (route, argv,
cwd, executable hash, VM identity, guest exit record, window index), `image-records.txt` (the
launcher's entry and code range per image launch), `capprof/window-N.tsv` (the raw counts with the
plugin's footer), `spec/` (runcpu's log and reports, the run directory's command and output
files), `rundir.sha256`, `result.json` (`tools/verdict.py`: PASS only when runcpu validated the
outputs), `windows.json` (every guest invocation has a complete, valid window), `profile.json`
and `profile/window-N.json` (the attribution). The build's `build-record.txt` holds the compiler
and SDK hashes and the executables'. A run whose verdict is not PASS or whose windows are
incomplete exits 1.

Nested re-exec can emit several image records within one window. The reporter
associates them with the invocation's unique staged path, allowing the extra
`../` prefixes used by perlbench's test driver. Every record in that group must
have the same entry and code range; unknown images, missing groups, reordered
groups and differing placements are refused. Instruction bytes and executable
hashes are still checked, and raw attempts are counted once. Each window's
`image_records` retains the entire group. A quick slice's synthetic single
record keeps its existing stop-site byte checks.

## Validation of the instrument

`tools/capprof-validation/vt-run.sh TAG` boots the platform and runs the fixed programs
(`vt.c`, `alias.c`): loops with known trip counts in a domain and as a plain program, a 50M-loop
window that crosses quanta, a prefault and a fault test whose retried stores must be counted
twice, and an alias test (the same bytes at two addresses must be two records). `vtcheck.py`
fails unless every check passes. Run it on every new platform before a measurement.

## Long benchmarks: work-bounded slices

perlbench, x264, mcf and xz test inputs run for 45-220 minutes in the Capstone arm. A slice of
"the first N attempts" or "the first N minutes" compares unequal work between arms (the arm with more
instructions per iteration gets fewer iterations), so it is not used. A WORK-bounded slice is: both
arms build the same source, so the K-th entry of the same function is the same point of the
program's work in both.

    PROFILE_STOP_AT=primal_bea_mpp:3551 tools/profile-batch.sh ARMS test OUTROOT cap-dev rv64 -- 505.mcf_r
    tools/prof-compare.py OUTROOT/rv64/505.mcf_r OUTROOT/cap-dev/505.mcf_r

`PROFILE_STOP_AT=SYMBOL:K[,START_SYMBOL:K]` (or `"stop_at": {"symbol": S, "count": K}` in the arms
file, top level or per arm, optionally with `start_symbol`/`start_count`) makes `profile.sh`
resolve the symbol in the arm's binary of the benchmark (`tools/stop-at.py resolve`, with the arms
file's `nm`) and give the capprof plugin its address: the window then closes at the function's
K-th entry exactly as the stop marker closes it, and the plugin ends the VM. Entries count inside
open windows only and accumulate over the run's windows (an invocation's window continues the
previous one's count). An entry is counted when the instruction after the site runs and the last
instruction executed in that image's text before it was the site (each image instruction carries
one more counter, from its first translation, armed or not), so a first instruction that faults and is retried counts once, a handler that
reaches the instruction after the site by another path counts nothing, and the window ends after
the site's K-th execution. A site that transfers control has no such successor, and a site at the
image's entry (or next to it) would be reset by it: both are refused. Identity: a site counts only
at exactly an armed image base plus its offset, and a base is armed when that image's entry
instruction (the plain program's e_entry, the domain's domain_main) runs; bases stay armed, one per
base, so processes at different bases that interleave each keep theirs (a fork's child shares its
parent's base and counts with it; an exec arms the new image when its entry runs). The plain arm
runs at its link addresses; a domain image is mapped by the launcher at a base it reserves at run
time, a multiple of the image's grant size. The plugin keeps one candidate per (address, physical
page), so another program with the site's bytes at the same address is a second candidate, not
more entries of the first, and it matches the site's successor by its bytes too. The kit derives
the domain's image record, which the launcher never wrote, from the armed base.

A stopped run's verdict is SLICE, never PASS: runcpu cannot validate an output cut short.
`tools/stop-at.py finish` requires the plugin's `capprof/stopped` record to match the request
(entry K, the symbol's bytes, the plain link address or the anchored base, aligned to the grant
size), ONE (address, page) candidate to have entered the site in the stopped window, every
instruction the stopped window EXECUTED (attempts > 0; a record of a translation alone is not
execution) in the domain's text at that base (in the plain arm: inside
the stop function, since other static programs share the link addresses) to be the image's own
bytes, every guest invocation before the stopped one to have passed and the stopped one to be the
resolved binary's. `prof-report.py` refuses a slice (and the verdict becomes FAIL) whose stop
address does not hold the site's bytes in the image as placed, or whose stopped window has attempts
inside the stop function (and, for a domain, anywhere in its program range) that are not the
image's. Every later command runcpu issues is refused at
once as `after-stop` (`capstone-run.sh`). `prof-compare.py` compares two runs only when both are full runs
or both are slices with the same symbol, K and start. A K the run never reaches gives a full run,
whose window footers carry `# stop_entries_so_far` (the site's entries), the count to slice from;
with a start site that run's windows are trimmed, so it is FAIL.

What a slice is not: it measures a PREFIX of the program's work, and the prefix is not the whole
run in miniature. Measured on the census platform (2026-11-03, slices at 1/5/20 % of the entries
of mcf's `primal_bea_mpp` and xz's `sha_compress`, against the full test runs): the TOTAL
Capstone/plain ratio of the 5 % slice was 1.258 against mcf's 1.319 and 1.210 against xz's 1.133.
The runtime's cost is not spread over the work: page population lands at the start (xz's slices
carry kernel work worth +15-20 % of the plain attempts, the full run +6.9 %), and the retire sweep
and teardown at the end, which no slice reaches (mcf's module class is 0.08-0.77 % of the plain
attempts in a slice, 2.04 % in the full run). The PROGRAM class came closer: mcf 1.238 (5 %) and
1.255 (20 %) against 1.227, xz 1.026 and 1.031 against 1.027. So a slice compares compiler output
over equal work; a claim about the kernel and module classes, or about the total, needs the full
run. Where the run has phases of different cost (xz's twelve invocations at six compression
levels, per-invocation ratios 1.06-1.25) a prefix slice measures the first phases only. Pick the function from a full
run's counts: entered often, at a steady rate, the SAME number of times in both arms (a function
whose first instruction is also a loop header in one build counts the loop's iterations there).
The correctness test is `tools/capprof-validation/stopat-run.sh` (`stopatcheck.py`): a loop calling
a function 1000 times over two windows, plain and as a domain, with stop and start sites, checked
instruction by instruction against the full run; K = 1; a site that faults and is retried across a
block boundary; a SIGSEGV handler that reaches the site's successor by another path; two domain
images at different bases interleaved in one window; the refusals; and `stopat-kit-test.py`, the
kit's verdicts on run directories built from those boots, tampered copies included (OLD_PLUGIN= and
OLD2_PLUGIN= run the earlier plugins as negative controls).

## One-command milestone census

On the measurement host, `tools/milestone-census.sh DEV_COMMIT TAG [--quick]`
runs gcc, imagick, lbm, nab, perlbench, mcf and xz at SPEC test size and
creates `runs/TAG/northstar.txt` and `northstar.json`, with a same-time plain
arm, the P/K/H terms, both overheads, and changes in percentage points from
the previous completed milestone. Newly admitted benchmarks have `n/a`
previous-milestone deltas until a compatible baseline exists. `tools/prof-northstar.py A B` also prints
these terms and one JSON line directly from two run directories (or their
`profile.json` files). Missing bucket data, non-positive denominators,
inconsistent totals, failed verdicts and different work boundaries are errors.
The bucket classifier is shared with `prof-buckets.py`; H is an accounting
exclusion, not a causal claim about the harness.

The driver needs a local Git object repository containing the target commit,
the census4 arms/platform and toolchain records, a prepared kernel matching the
platform Image, and a RISC-V Linux cross compiler. It does not build a new QEMU
or kernel: an unrecognized QEMU pin is an explicit UNRESOLVED error requiring
a matching platform. Default inputs are under `~/prof-dev` and
`~/llvm-capstone`; a private lane can configure them once:

```sh
export CENSUS_ROOT=~/lanes/laneQC
export CENSUS_BASE=~/prof-dev
export CENSUS_REPO=~/lanes/laneQC/repo.git
export KERNEL_BUILD=~/lanes/laneQC/kernel
export CROSS_COMPILE=~/lanes/laneQC/host/bin/riscv64-buildroot-linux-gnu-
tools/milestone-census.sh DEV_COMMIT census5
tools/milestone-census.sh DEV_COMMIT quick1 --quick
tools/milestone-census.sh DEV_COMMIT quick2 --quick
```

`CENSUS_ROOT` must exist and is also the guest share. Resolved SPEC cache
paths must stay beneath it, including when using symlinks; on a separate
scratch filesystem, keep the cache and run directories under one common root.
`CENSUS_ARMS` can select a different template.
`CENSUS_PREVIOUS` and `CENSUS_PREVIOUS_ARM` select a previous run explicitly
(the previous plain arm is `rv64`). Otherwise the latest completed compatible
run in this output root is used; the first full run uses census4/cap-dev4,
and the first quick run establishes its own baseline. A quick run never uses
a full run as its baseline. Every run has a new tag: existing run directories
are refused. Failures retain their logs and do not create a completed
`northstar.json`. An incomplete build cache is also refused until inspected.
A root lock prevents concurrent invocations from sharing mutable build state.

Compiler/SDK reuse compares Git object hashes for `llvm`, `clang`, `lld`,
`cmake`, `third-party`, `compiler-rt`, `capstone/ports/common`,
`capstone/ports/musl-capstone`, `capstone/sublet`, the runtime's root
`CMakeLists.txt`, `application`, `cmake`, `common`, `domain`, `include`, `host`,
`capstone/tests/capstone-test-env.sh`, and these files under `runtime/virtual`:
`start.S`, `thread.c`, `pthread.c`, `fenv.c`, `mapping.c`, `heap.c`, `link.ld`,
`build-sdk.sh` and every top-level `.h` header. `Application.cmake`, also
hashed, controls the SDK source list. Launcher, adapter and standalone gate
changes do not invalidate compiler/SDK reuse; the adapter is always rebuilt.
The actual hashes are in `source-hashes.json`; the build recipe version is
also part of the key. Compiler and SDK are reused together only when every
input matches a completed capN record; otherwise both are built with the
census recipe (Release, assertions enabled, default virtual heap protection).
The SPEC kit tree is additionally part of the executable cache key. An existing private cache is extended when newly admitted benchmarks lack
recorded executables; only those benchmarks are built and the hash record is
updated atomically. New caches
build the Capstone executables with a private label in a private SPEC copy;
plain executables retain the template's recorded compiler, label and installed
SPEC verification hashes. Their config must match the requested kit before
its generated `__HASH__` section; a changed plain config requires a newly
built template. This is
a same-time plain execution, not a newly built plain compiler. Sources in the
installed SPEC tree are never edited. Cached compiler, SDK dependencies and
SPEC executable hashes are checked before reuse.

Each invocation exports the target runtime and kit, builds its launcher and
module, copies the template platform with `cp -rL --sparse=always`, replaces the adapter and
regenerates file hashes. It runs both arms using that platform, writes their
usual records, and saves the top 15 Capstone function deltas against the
previous milestone. No existing kit, platform or run is updated. The script
preserves zero-filled image extents as holes to save disk space; the copies
are independent regular files with identical bytes, never links. The script
checks both the home and output filesystems for at least **15 GiB free**
before copying, building and profiling, or **20 GiB** before a compiler build;
this is a reserve check, not a reservation against other host users. LLVM
builds default to 20 jobs (`CAP_JOBS`), runtime builds to 4, profiling to 4
(`JOBS` in either mode). The musl archive can be selected with `MUSL_ARCHIVE`.

perlbench uses `x-capstone-supervisor-slots=2` and module parameter
`supervisor_slots=2` in both arms; the other benchmarks retain their template
settings. Both perlbench arms use `mapped-xattr` and stage images on tmpfs:
its file-owner tests need guest ownership, and its test driver re-executes
`../$^X` after entering `t/`. The staged program word therefore stays relative
via a root-reaching path; an absolute `/tmp/stage-...` would become the invalid
`..//tmp/stage-...`. The census overlays its `capstone-run.sh` as part of the
host profiling tools, with the original fix from `aa25b8013ac3`.
Each benchmark/arm has its own saved arms file. Full xz includes
all twelve SPEC test invocations in the harness's normal order, sums their
windows, and requires the full SPEC oracle to pass. Quick xz stops inside
the first invocation; the remaining eleven do not execute. Any commands
subsequently requested by the harness are refused as `after-stop`.
Its quick number does not represent all six compression levels.

Every profile runs under `profile-watchdog.py`, a Linux child subreaper.
`CENSUS_RUN_TIMEOUT` sets a positive wall-time limit **per benchmark/arm**,
including profile preparation and VM boot: default 600 seconds in quick mode,
14,400 seconds in full mode. The census queue wait is outside this limit;
profiling uses a private, nonblocking VM lock. The watchdog owns even the
VM that detaches from `vm up`, so it does not become an orphan under PID 1.
On timeout or TERM/INT/HUP it signals only its descendants by PID descriptor,
escalates from TERM to KILL, reaps children and writes `result.json` as FAIL.
It also rejects a profile that returns with live descendants. `watchdog.json`
records the deadline, elapsed time, signalled PIDs and any surviving children.
The isolation depends on Linux subreaper and pidfd support; unavailable support
refuses the run. An uncatchable kill of the watchdog or host failure is outside
this cleanup guarantee. Failed rows are retained in `batch.json`, and any
failure prevents a completed `northstar.json`.

Quick mode uses these fixed `PROFILE_STOP_AT` prefixes:

| Benchmark | Stop site | Meaning |
|---|---|---|
| gcc | `final_start_function:1` | first function reaches final code emission |
| imagick | `GetVirtualPixelsFromNexus:32` | first 32 virtual pixel fetches |
| lbm | `LBM_performStreamCollideTRT:3` | two stream/collide steps, then entry of the third |
| nab | `mme34:5` | four force evaluations, then entry of the fifth |
| perlbench | `Perl_pp_nextstate:10000` | first 10,000 statement-boundary opcode entries in makerand.pl |
| mcf | `primal_bea_mpp:3551` | entry of the 3,551st pricing call |
| xz | `sha_compress:13392` | first 13,392 hash-compression calls, in the first invocation |

The census4 raw entry counts for the original four sites agree between Capstone and plain
(1, 1583, 20, 2007 respectively). Every new quick run must reach the specified
boundary in both arms and pass the harness's slice validation. Its verdict is
**SLICE**, never SPEC PASS. Startup/page population is included, teardown is
excluded, and imagick's separate validation process is excluded. These prefixes
are regression checks against the *same prefixes*, not estimates of the full
north-star metric. Initial compiler/SPEC builds are outside the minutes-long
warm-check expectation. `elapsed_seconds` includes preparation and reports
that distinction explicitly. Use repeated quick runs to report the observed
spread; use full PASS runs for milestone claims.

Tool regressions on the measurement host:

```sh
python3 tools/test-prof-northstar.py
python3 tools/test-milestone-census.py
python3 tools/test-profile-watchdog.py
python3 tools/test-prof-images.py
```

Qualification on 2026-11-03: dev `114b90b7be2e`, p12,
`~/lanes/laneQC/quick-results/runs/{quick-s1,quick-s2}`. Both checks validated
all eight rows as SLICE, with the same executable cache and prefixes above.
Elapsed time including adapter builds and platform preparation was 160.913 s
and 165.993 s. The output and cache were on p12's scratch filesystem; each
platform was an independent sparse copy with its template hashes verified.

| Benchmark | Program spread (pp) | Boundary spread (pp) |
|---|---:|---:|
| 502.gcc_r | 0.057804 | 0.278401 |
| 538.imagick_r | 0.146936 | 3.259637 |
| 519.lbm_r | 0.006077 | 0.549037 |
| 544.nab_r | 0.039390 | 0.751334 |

Spread is the maximum minus minimum of two observations, not a confidence
interval. P_A and P_B were identical across repeats for every benchmark;
the program percentage spread comes from variation in the K_A denominator.
The boundary variation, especially imagick's, limits what a small
change in one quick check can establish. Full P/K/H terms and the compatible
quick-to-quick comparison are in each run's `northstar.{txt,json}`; the full
[census 5](../../../docs/plans/2026-11-03-boundary-north-star.md) is separate.

### Seven-benchmark quick qualification (QC2)

On dev `114b90b7be2e`, p12 runs
`~/lanes/laneQC/quick-results/runs/{qc2-fixed-quick1,qc2-fixed-quick2}`
validate all 28 rows as SLICE. Elapsed times including adapter builds and
preparation were **270.111 s** and **297.417 s**. Both use the same compiler,
SDK, executables, corrected staging wrapper, watchdog and driver.

| Benchmark | Program 1 % | Program 2 % | Spread pp | Boundary 1 % | Boundary 2 % | Spread pp |
|---|---:|---:|---:|---:|---:|---:|
| 502.gcc_r | 29.881559 | 29.954796 | 0.073236 | 36.145779 | 36.658368 | 0.512589 |
| 538.imagick_r | 22.564139 | 22.309460 | 0.254679 | 66.990340 | 68.375300 | 1.384960 |
| 519.lbm_r | 1.860512 | 1.872154 | 0.011642 | 13.053159 | 13.481661 | 0.428502 |
| 544.nab_r | 10.915134 | 11.037602 | 0.122468 | 6.687585 | 7.536146 | 0.848561 |
| 500.perlbench_r | 39.640020 | 39.459475 | 0.180545 | 47.018071 | 46.545641 | 0.472430 |
| 505.mcf_r | 2.623631 | 2.631289 | 0.007658 | 0.049863 | 0.294608 | 0.244745 |
| 557.xz_r | 1.448389 | 1.448414 | 0.000025 | 25.199259 | 25.206328 | 0.007069 |

P_A and P_B are identical across repeats for all seven programs; the program
percentage spread comes from the K_A denominator. Spread is max minus min of
two observations, not a confidence interval. All watchdogs reported zero
cleanup signals and zero remaining descendants. Perl's recorded CPU/module
slots are both two, both shares use mapped-xattr, and staged program words
remain relative. Both xz slices stop in window zero with one invocation.

Command: `python3 ~/lanes/laneQC/qc2-summarize.py qc2-fixed-quick1 qc2-fixed-quick2`.
The output `~/lanes/laneQC/qc2-fixed-quick-summary.txt` checks the metrics,
verdicts, windows, stop sites, effective perl settings and watchdogs.

The earlier `qc2-quick{1,2}` prefixes used the original wrapper and plain
share setting. The first full attempt, `runs/census6`, exposed its invalid
absolute staged perl path and was excluded. The corrected prefixes above
supersede those initial quick numbers. Perl's prefix covers `makerand.pl`;
it does not validate the nested test driver. Full xz and perlbench acceptance
require the complete SPEC oracle, with the kit's existing run-directory
`pack.t` adaptations explicitly retained.

### Seven-benchmark full qualification (QC2)

All fourteen SPEC test oracles pass on dev `114b90b7be2e`. The
[census6 table](../../../docs/plans/2026-11-03-boundary-north-star.md) uses p12
`~/lanes/laneQC/quick-results/runs/census6-reported`. Its raw capture,
`census6-verified`, retains a nonzero perlbench reporting exit: the old reporter
rejected 319 image launches for two top-level invocations after SPEC had passed.
The corrected reporter handles the two staged groups (1 and 318 records);
unknown images or differing placements remain errors. No benchmark was rerun
or raw counter edited to recover reporting.

`qc2-reprocess.py` on p12 rechecks archived windows and the perlbench oracle,
then writes separate reported directories. Its replay of the full batch and
both quick batches leaves all 41 previously successful profiles' total,
function, class and instruction-mix counts identical. The derived perlbench
`result.json` is the unchanged oracle checker's output; `captured-result.json`
and `watchdog.json` retain the original failure. `postprocess.json` records the
successful reporting command and recovery. `qc2-final-summary.py` verifies
all seven pairs and both quick repeats. The four existing run-copy `pack.t`
adaptations remain in the full perlbench oracle's scope.
