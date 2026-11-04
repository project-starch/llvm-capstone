# Unpinned page-fault experiment

These are lane N's `perf/linux-question-micro` tools at
`390a4043851a9995be74b726fc72ceec1cd6a7b8`, extended with `fault MIB WARM`.
Mapping, verification and unmapping occur outside the fault window. A cold
window stores one word per page; a warm window first populates the same pages
outside the window. Every fault-mode run verifies every stored word.
The original `touch` mode includes mapping and unmapping and is retained as
a separate whole-path measurement. No THP is used.

Build `lqmicro.c` with the shared virtual SDK's `capstone-cc -O2` and the plain
RISC-V musl compiler's `-O2`, naming the outputs `lqmicro-cap` and
`lqmicro-plain`. The runner also expects lane N's unchanged `lq-hog` binary,
even for these modes where it does not run it. The runner header documents
its environment; set all writable directories to the experiment's own lane.
It enables raw moves, disables test CSRs and forwards module parameters at
initial load. Use the matching patched kernel and model, with `unpinned=1`.

Run the following list once as `capstone` and once as `plain`:

```
-- fault 4 0
-- fault 64 0
-- fault 4 1
-- fault 64 1
-- touch 4
-- touch 64
```

Let C and W be cold and warm window attempts. The fault estimate is
`((C64-C4)-(W64-W4))/15360`. This removes fixed setup and the successful
program loop, retaining the faulting instruction's extra attempt. Function
counts use the same linear combination. This is a marginal cost, not a
single-fault latency trace. Use capprof's per-window `all_functions`, whose
counts are exclusive of callees; do not sum inclusive call costs.

Baseline on runtime `1ad3511096f0` (module unchanged from `c866e1d2fe76`),
QEMU `0542c5556c2d`, patched Linux 6.1, cap4 SDK and plain musl, p12:

```
A fault pages 15360 total/page 4700.1548828125 busy/page 4601.07109375
plain fault pages 15360 total/page 4093.0784505208335 busy/page 3994.7899088541667
A touch pages 15360 total/page 6521.627083333334 busy/page 6299.8626953125
plain touch pages 15360 total/page 5926.385807291666 busy/page 5703.329036458334
```

`busy` applies the north-star function-bucket exclusions, which are an
accounting proxy; generic mm list helpers can fall into the excluded fs
bucket. Retain total counts alongside them. Every invocation returned zero
with one complete window; fault modes reported `bad=0`.

Raw evidence: p12 `~/lanes/laneM7a/runs/{A,plain}` and
`~/lanes/laneM7a/logs/{baseline-cost.txt,micro-A-plain.json}`. No timing,
cycle, full SPEC or complete runtime qualification follows from these runs.
QEMU counts CSRUNV as one instruction, without pricing its internal state
save/restore work. These measurements cannot justify direct trap delivery
on the grounds of hardware switch latency.

## Software prototype

`unpinned_fast_fault=1` reuses the VMA that resolve already validated under
mmap_lock. Only private anonymous VMAs take the direct handle_mm_fault
path. File-backed, special and text mappings retain fixup_user_fault.
The lifetime lock, grant/access checks, write-backing policy, repeat bound,
stop/deadline checks and tag-aware kernel remain unchanged. The parameter
is read-only while loaded and defaults off. No ISA or Linux change is made.

Three runs per mode, same candidate binary with the parameter toggled,
produce the following command-derived marginal costs:

| Mode | Total attempts/fault | Busy attempts/fault | Total range |
|---|---:|---:|---:|
| Original unpinned path | 4700.155 | 4601.071 | one run |
| Prototype OFF | 4709.625 | 4611.718 | 4709.265–4709.906 |
| Prototype ON | 4355.343 | 4257.468 | 4353.249–4357.864 |
| Plain Linux | 4093.078 | 3994.790 | one run |
| Direct S-mode delivery | UNRESOLVED | UNRESOLVED | not implemented |

Command: `python3 ~/lanes/laneM7a/summarize-micro.py` on p12:

```
saved_total_per_fault 354.28185763888905 saved_busy_per_fault 354.24963107638905
today_minus_fast 344.81217447916606
fast_minus_plain 262.2642578125001
```

All cold/warm and retained touch invocations return zero; all fault-mode
value checks report bad=0. The original binary and prototype OFF differ:
the experiment adds dispatch overhead. Do not describe the OFF switch as
free or pool it with the original baseline. No timing claim follows.

The OFF/ON module self costs are 326.530/340.499 attempts per fault, while
mt_find drops from 520.485 to 260.226 and fixup_user_fault from 61.057 to
zero. The locks bucket is 455.161/455.031 versus plain 453.992. ktime_get
still costs about 122.286 attempts per fast-path fault. Removing lifetime
serialization or stop checks is not justified by these data.

Raw opcode counts give exactly one CSRUNV per cold domain page and none
in the warm windows. Direct delivery remains conditional on a separately
specified cycle model establishing a meaningful hardware escape cost.
The residual instruction gap alone does not establish that condition.
A future arch hook must save capability and FP state losslessly, bind it
to the correct task/mm/domain, preserve grant lifetime across sleeping
faults, handle errors/signals/preemption/nested traps, and retry only after
fresh capability checks. Assigning stvec without that contract is unsafe.

## Workload and safety evidence

The four SPEC test inputs pass in OFF, ON and matched plain arms. Existing
census4/rv64 executables and shared compilers are reused; no benchmark
source is changed. Commands and exact records are retained on p12 under
`~/lanes/laneM7a`: `compare-spec.sh`, `spec-table.py`, `logs/spec-table.json`,
`logs/*-compare.txt`, `logs/*-buckets.txt`, `runs/spec-valid` and
`runs/spec-imagick-real`. Only PASS rows contribute below.

| Benchmark | Total B-A | Boundary K B-A | Boundary points B-A |
|---|---:|---:|---:|
| 502.gcc_r | -508120 | -484470 | -1.523485 |
| 538.imagick_r | +865457 | -65877 | -0.034203 |
| 519.lbm_r | -37227784 | -37132216 | -0.559007 |
| 544.nab_r | +50864 | -72980 | -0.001061 |

For GCC, the matched plain denominator P+K is 31800110. Trace entry counts
show 1422 fewer fixup_user_fault entries, with one auxiliary fault of
variation. Multiplying that approximate eligible count by the micro busy
saving projects 1.584092 boundary points; the workload measures 1.523485.
Boundary overhead is 36.174252 points OFF and 34.650767 ON in this batch.
The software change leaves CSRUNV attempts unchanged at 2079. Top self
changes are mt_find -347641, fixup_user_fault -86742, find_vma -32729,
find_extend_vma -32706 and the new helper +20281.

H (excluded attempts) is reported separately:

| Benchmark | Plain H | OFF H | ON H |
|---|---:|---:|---:|
| 502.gcc_r | 636507 | 1375935 | 1352297 |
| 538.imagick_r | 9883002 | 5005894 | 5937252 |
| 519.lbm_r | 34484856 | 27728291 | 27632735 |
| 544.nab_r | 2228363 | 2165351 | 2289207 |

Imagick and NAB show timer/idle variation larger than their small measured
savings; these single batches establish execution, not a robust small
performance improvement. The bucket exclusions also remove application
I/O and leave some harness-driven scheduler work in K.

A two-process copy-on-write/migration torture run with the fast path returns
zero in both processes and `TORTURE PASS total_bad=0`. The sampled process
has 1564 observed BSS and 18104 heap page moves. Both subsequent stale
capability dereferences terminate with cause 25, exit 139. The test checks
live/stale/scalar/zero granules and pointer graph walks; it does not prove
all writers or the whole architecture. Evidence: `tt/fast-cow`,
`tt/fast-stale`, `check-torture.sh`, `logs/torture.log`.

Runtime gate qualification is recorded separately after both queued arms
finish. No landing is authorized by these micro and workload results.
