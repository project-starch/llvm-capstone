# Application benchmark study

Status: candidate suite catalog, four-arm planner and explicit CheriBSD process
policy controls implemented. Standard-suite adapters are not yet qualified.
This is memory-behavior work, with no security ranking, allocation-event replay,
or emulator-time performance claims. This document describes the original
default-CheriBSD reference comparison. The primary nested-allocator comparison
now follows the [Sublet/PoisonCap design](sublet-poisoncap-memory-study.md), with
its own matched spatial control and explicit qualification gates.

## Questions and comparisons

The main paper question is the incremental effect of Sublet inside an
application's existing internal allocator. Use Capstone plus its original
internal allocator as the baseline, keeping the same outer heap. Compare
against one CheriBSD purecap binary with malloc revocation enabled and disabled.
Run the alternative outer-malloc comparison as a separate named profile. Do
not silently substitute it where an internal-allocator application integration
is missing. The [study directory](../../experiments/study/README.md) defines
the exact arm contracts and the executable workflow.

Within-platform contrasts isolate Sublet integration and CheriBSD revocation
more clearly than a single cross-platform ratio. Cross-platform observations
also include differences in allocators, libc, OS services, compiler and layout.
Record those differences. Never describe the four arms as a hardware-only 2×2
factorial experiment.

## Recognizable workloads

Choose upstream or established workloads before inspecting comparative results.
Known workloads provide external relevance; controlled extensions explain
memory behavior. Keep their labels and conclusions separate. Versions are
chosen to match the current ports, not asserted to be the newest releases.

| Application | Primary workload choice | Qualification required |
|---|---|---|
| SQLite | Upstream `speedtest1`: main, orm, cte, json, star, parsenumber, app | Reuse existing work; match engine version, memsys5/lookaside configuration and SQL output; migrate to the persistent application runner. |
| CPython | `pyperformance`: JSON loads/dumps, pickle, Richards, DeltaBlue, regex_v8, nbody | Pin suite 1.14.0; qualify dependencies and fixed-work bodies. Host orchestration replaces unsupported subprocess control, with the adaptation disclosed. |
| mruby | Its pinned `benchmark/` workloads: lists, Mandelbrot, ambient-occlusion rendering | These are upstream workloads, not a claimed universal Ruby application suite. Preserve normal GC; validate outputs and GC-slot instrumentation. |
| Perl | Core `t/perf/benchmarks` for immediate coverage; SPEC CPU2017 perlbench as a separate expansion | Core snippets are microbenchmarks. The Cachegrind controller is not portable here. SPEC needs licensed sources and its own modified interpreter port. |
| PostgreSQL | `pgbench`: select-only, simple-update, tpcb-like | Full pgbench needs a server. A single-user execution of its SQL bodies is a derived workload, not a pgbench result. |
| FFmpeg | Established Phoronix FFmpeg profile as the application target; FATE samples as a nearer decode corpus | Encoding profiles exceed the current configured decoder. FATE provides correctness inputs, not a standard performance score. |

Source details: [SQLite methodology](https://www.sqlite.org/cpu.html),
[pyperformance workloads](https://pyperformance.readthedocs.io/benchmarks.html),
[mruby benchmarks at the port pin](https://github.com/mruby/mruby/tree/9d523e2f74f2e63ca02840937523de61398a617d/benchmark),
[Perl core controller](https://github.com/Perl/perl5/blob/d456446ba5b8fba87f3f26fa700eb749c622e434/Porting/bench.pl),
[SPEC perlbench description](https://www.spec.org/cpu2017/Docs/benchmarks/500.perlbench_r.html),
[pgbench documentation](https://www.postgresql.org/docs/17/pgbench.html),
[Phoronix FFmpeg profile](https://openbenchmarking.org/test/pts/ffmpeg),
[FATE documentation](https://www.ffmpeg.org/fate.html).

The catalog records immutable source commits or archive digests where verified.
Unresolved source pins remain explicit blockers, as do broad candidate families
that still need expansion into exact cases. Do not download licensed suites
without an available license; use a local path plus a recorded content digest.
Fetched source, corpora and generated datasets stay outside Git under
`$CAPSTONE_TMP_ROOT`; add no benchmark submodules.

tshark remains an application-build candidate. nginx, APR/httpd, memcached and
Whisper currently contribute allocator components rather than qualified whole
applications to this study. They do not increase the application count.
MicroPython is excluded in favor of the requested CPython focus.

## Measurement layers

1. **Common requests:** requested live/peak bytes, successful allocation count,
   peak live objects, distinct allocation starts, and reuse-distance CDF. Count
   allocation calls rather than elapsed time. State exactly which APIs are observed.
2. **Internal allocator:** live object/slot requests, slab/context/arena capacity,
   reusable holes, and backing grants at application release boundaries. Instrument
   the same logical boundary on all four arms. Outer malloc misses most pymalloc,
   GC-slot and PostgreSQL-context behavior.
3. **Allocator storage:** occupied blocks, internal rounding, retained backing,
   committed/mapped storage, quarantine where directly observed, and metadata.
   Never subtract unrelated ledgers and label the difference fragmentation.
4. **Platform reservation:** logical pools, physical grants, static tables,
   stack/code/data, observer storage, node/tag storage and driver caches. State
   what can return to the allocator versus to the OS. Keep incompatible RSS
   definitions separate.

Primary figures: four-arm reuse CDFs; distinct starts versus completed work;
requested/backing storage curves through release and bursts; internal occupancy
by allocator layer. Show incremental Sublet/base and revocation-on/off contrasts
alongside absolute bytes. A capacity experiment requires matched available-byte
budgets, complete metadata accounting and a justified failure oracle first.

The current metadata sweep changes execution costs. Before including a new
workload in the memory comparison, check its complete phase samples against a
no-in-process-sweep control with sufficient metadata capacity. Existing equality
for twelve FFmpeg/mruby configurations is not a blanket exemption. Wall time,
pause latency, cache traffic and hardware metadata scaling remain outside this
QEMU memory campaign. Any later hardware study needs a separate platform plan.

## Qualification and campaign lifecycle

Use Python on the host and small C measurement hooks in the guest. Reuse the
shared SDK, build recipes, application runner, CheriBSD Guest and allocation
observer. Add application descriptors/adapters, not per-benchmark VM scripts.
The first implementation uses JSON plans and append-only JSONL attempts; that
is sufficient for the current serial machine. A database or distributed job
queue is not a prerequisite and can later consume the same identities.

For each adapter:

1. Fetch pinned source/data; keep original files unchanged and hash any patch.
2. Establish native useful-output oracles and fixed work counts. Preserve a
   control with little allocation alongside allocation-heavy cases.
3. Build all four configurations from matched application/dependency versions.
   Check internal integration, libc, observers and each resource reservation.
4. Check the effective revocation policy inside each CheriBSD process. Match
   stdout, completion, phase sequence, observer accounting and cleanup.
5. Size the observer for the largest predeclared work count. The fixed address
   table must not overflow; qualify a larger observer on all arms if needed.
6. Run small/medium/reference inputs on all arms and check sweep invariance.
   Record unsupported cases as port, runtime/OS, benchmark-dependency, source/
   license, observer, or resource-budget limitations. Do not treat them as passes.
7. Freeze a confirmatory plan, source/build identities, input sizes and work
   counts. Start with three repetitions for deterministic memory observations;
   increase only under an explicit new plan if nondeterminism needs investigation.

Plans enumerate case × workload variant × four arms × repetition. The host
builds once per configuration, boots once per platform batch, and runs many
fresh application processes. Continuous epochs/bursts execute inside one
process; restarts are not additional epochs. Keep warmup operations explicit
and collect their memory state because they can change later retention.

For long-run characterization, use a predefined size/epoch grid and plot every
sample, including collection drops and cases where Sublet loses. Confirmatory
benchmark work should be fixed by transactions/frames/iterations, not by an
emulator wall-clock duration. Use a seed for ordering and generated datasets.
Do not reduce only one arm's workload to fit a limit or tune away its default
policy. Instrumented adaptations do not produce official SPEC/PTS/pyperformance
performance scores.

Store all failures and unavailable cells with the planned denominator. Resume
only never-attempted cells after reconciling interrupted launches. Retrying a
failed cell is an explicit diagnostic campaign linked to the original. Archive
raw commands, logs, platform identities, inputs and binaries by content digest;
exclude guest credentials. Commit compact result records and standalone plots.
Keep the paper manuscript unchanged until the results and framing are approved.

## Rollout

1. Qualify mruby upstream workloads, SQLite speedtest1 and selected pyperformance
   bodies first. They exercise distinct allocator designs and reuse existing
   application infrastructure. Add the missing binary-output oracle support.
2. Qualify Perl core workload adapters and a matched CheriBSD Perl build; keep
   the licensed SPEC expansion separate. Do not label outer-malloc Sublet as a
   Perl internal-allocator integration.
3. Expand FFmpeg to the required real CLI/codecs; qualify corpus-backed decode
   characterization meanwhile. Bridge nested pools into the common application ABI.
4. Complete PostgreSQL server prerequisites for pgbench. SQL-body experiments
   can inform context behavior earlier under a separate derived-workload name.
5. Expand to tshark and other full applications only after their builds and
   output oracles pass; integrate new adapters into the same study contract.
