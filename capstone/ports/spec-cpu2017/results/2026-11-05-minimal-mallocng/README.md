# Current minimal mallocng: SPEC and PHP instruction measurements

One freshly built SDK from `ae47030ceeb6371835e90f474bb18cbed1165c23`
(PRs #301 and #302 included) was used to relink all eight SPEC C benchmarks
and PHP 5.0.0. Six SPEC **test** workloads validate completely. Perl and x264
reach their time limits; neither is a complete passing result. These are
research measurements, not reportable SPEC scores.

[Machine-readable evidence](measurements.json) contains counts, grouping,
validation records, executable/library/compiler hashes, source-package hashes,
ISA counters and artifact locations. No licensed sources, inputs, output files
or binaries are included here.

## Configuration and limits

- Both arms statically link musl 1.2.5; no glibc/musl comparison.
- Application compilation: `-O2`, matching portability flags, F/D instructions
  and the lp64 calling convention. The Capstone ABI uses 16-byte capabilities;
  plain RISC-V pointers are 8 bytes.
- Capstone SDK defaults: object bounds and lifetimes enabled, freed-payload
  scrubbing disabled. Compressed representable bounds; 32 MiB application data
  reservation and 1 MiB stack. Heap mappings are additional to that reservation.
- **Compiler limitation:** the reused Capstone `hoist-fast` compiler and plain
  RISC-V compiler are different revisions of the LLVM fork. Their binary hashes
  are recorded. This compares the measured software stacks, not the ISA alone.
- **Library optimization limitation:** the default Capstone musl is `-O1`;
  plain musl is `-O2`. The separate libc-only experiment below measures this
  difference without changing the baseline SDK.
- Identical pinned QEMU/kernel/adapter/plugin inputs, `icount=0`, one vCPU,
  8 GiB guest memory, ASLR disabled, executable staging to guest tmpfs.
  Independent guests overlap on the host. Host elapsed times are time limits,
  **not performance results**.
- Metric: byte-verified **instruction attempts inside the program**, including
  libc, allocator, compiler helpers and SDK runtime. Kernel, launcher and other
  outside-program work are excluded. Attempts include retries; they are not
  hardware retired-instruction or cycle measurements.
- SPEC sources and inputs remain unchanged except the existing frozen
  `src.alt` packages for Perl, GCC and mcf, used in both arms. No new edits to
  those packages. Both before/after audits check 3,399 original selected files
  against SPEC's MANIFEST, with zero mismatches. Fresh build source/package
  checks also pass. The relocated replay tree independently checks 1,817 files.

## Complete SPEC test workloads

| Benchmark | Plain instructions | Capstone instructions | Extra | Validation |
|---|---:|---:|---:|---|
| 500.perlbench_r | — | — | — | both timed out at 600 s |
| 502.gcc_r | 18,073,601 | 23,996,732 | +32.77% | PASS / PASS |
| 505.mcf_r | 27,299,535,363 | 28,249,804,291 | +3.48% | PASS / PASS |
| 519.lbm_r | 6,003,861,568 | 6,100,835,364 | +1.62% | PASS / PASS |
| 525.x264_r | — | — | — | both timed out at 600 s |
| 538.imagick_r | 94,490,259 | 96,747,997 | +2.39% | PASS / PASS |
| 544.nab_r | 6,863,873,816 | 7,228,868,307 | +5.32% | PASS / PASS |
| 557.xz_r | 48,993,774,479 | 49,985,865,291 | +2.02% | PASS / PASS |

The separate ImageMagick `imagevalidate` executable is excluded from benchmark
instruction totals but still executes and must validate successfully.
Perl/x264 receive a 600-second profiling limit, other workloads 2,400 seconds.
The deadline includes guest startup and profiling/report work.

### Work-bounded diagnostics

The x264 slice starts at `x264_slicetype_mb_cost:1` and stops at entry 101:
1,136,192 plain versus 1,124,335
Capstone attempts (**-1.04%**). Same binary/input/command
as the full workload, with the profiling plugin stopping the guest. This
counts 100 calls, **not a validated complete x264 workload**. The automatic
repeat also reaches SLICE in both arms; its original records remain on p12.

A 1,000-call `primal_bea_mpp` mcf slice gives
+3.80%; the full mcf result above is the authoritative
workload comparison. The attempted xz slice times out in the Capstone arm and
has no paired result. Full xz validation succeeds independently.

## Where the remaining instructions go

GCC has **5,923,131** additional program attempts:

| Location | Additional attempts |
|---|---:|
| Compiled benchmark functions | 3,556,726 |
| mallocng and mapping authority | 1,542,523 |
| Other libc / SDK runtime | 534,696 |
| Compiler capability-initialization helpers | 289,206 |
| Unresolved startup difference | -20 |

These groups reconcile exactly. Functions are assigned using application
objects and library archives. Inlined work is charged to its containing
function. Symbol-name ambiguities are retained explicitly in the JSON;
function deltas alone are not a causal decomposition, especially where one
compiler inlines a helper and the other does not.

In GCC's **application functions**, capability cursor arithmetic accounts for
1,867,440 attempts, bounds operations for 384,974, and other capability
operations (including moves) for 671,643. Together these are **82.2%** of the
application's extra 3,556,726 attempts. Remaining differences include stack
traffic (+193,813), other memory traffic (+276,651), ordinary arithmetic
(+134,437) and control flow (+27,768). Replacing pointer loads/stores with
capability loads/stores is counted as replacement traffic, not automatically
as an additional instruction.

The allocator runs 2,802,318 Capstone attempts versus 1,259,795 plain attempts.
Across the GCC program there are only **4,927 `cderive` and 2,880 `crevoke`
attempts**: 7,807 total, about 0.5% of the allocator delta. Thus the direct
lifetime instructions do not explain the million-instruction difference.
Ordinary mallocng work, representation/metadata operations, mapping-index
publication, extra allocation demand and their generated code dominate.
`__libc_malloc_impl` is entered 4,231 times versus 3,827 on plain RISC-V.
`enframe` is ordinary musl allocation framing, not a separate lifetime registry.

Other workload findings:

- **mcf:** `spec_qsort` adds 974.95 million attempts, the main positive contributor
  to the net 950.27 million difference. Optimize compiler output for this
  existing frozen source; do not rewrite the benchmark.
- **lbm:** `LBM_performStreamCollideTRT` contributes 81.91 million of the net
  96.97 million extra attempts.
- **nab:** application functions add 222.47 million, and other libc/runtime adds
  141.36 million. `exp` (+84.30 million) and `log` (+51.24 million) are concrete
  library targets; allocator overhead is only 1.13 million.
- **xz:** application functions add 740.67 million, other libc/runtime 184.68
  million, and allocation/mapping 66.71 million. Of the latter, mapping and
  unmapping account for 66.51 million. This is per-page mapping-index work,
  not a per-object tree search. `memcpy` adds 169.21 million; LZMA code generation
  is the larger application-side target.
- **ImageMagick:** `HorizontalFilter` and the runtime's `dl_vector_once` are the
  largest positive function deltas. The allocator delta is 0.234 million.

## PHP: outer allocator versus nested protection

Real PHP 5.0.0 CLI, all 145 translation units from its configured Makefile,
with the existing repository portability patches. Plain and Capstone stock
builds use the same source overlay/configuration; nested adds the existing
Zend Sublet hooks and support TU. `-O2` in all arms; shrink-stack disabled.
The nested arm explicitly receives `PHP_CAP_MODE=7`. Existing required
configure corrections, duplicate dead scanner-symbol localization and source
identities are recorded in the JSON. No interpreter/API stubs are used.

Two fixed workloads run three times each: the existing sort/serialization
sanity case and the existing 2,000-iteration array/string allocation churn
case. Every output matches its expected bytes; each arm has identical program
instruction counts across all three repeats. These are PHP workloads, not a
standardized PHP benchmark score.

Churn (`churn 197900`):

| Variant | Program attempts | Versus plain |
|---|---:|---:|
| plain RISC-V | 227,784,852 | +0.00% |
| Capstone, protected musl / stock Zend | 316,474,036 | +38.94% |
| Capstone, protected musl + Zend mode 7 | 396,095,779 | +73.89% |

Nested protection adds **79,621,743 attempts (+25.16%)** over the Capstone
stock-Zend arm. Its helper self counts are 25.54 million in `zend_cap_issue`,
20.64 million in `cap_block_of`, and 18.72 million in `zend_cap_reclaim`.
These include metadata recovery, checks, counters, mode handling and generated
call/save/restore code. The total increase in direct derive/revoke attempts is
only 692,254. Removing the checks indiscriminately would invalidate the
protection; the evidence points toward specializing/inlining the production
helper path and improving its code generation while retaining the existing
stale/interior/parent and compressed-bounds guarantees.

A separate ABI effect explains much of the **outer allocator** increase:
PHP's original cache accepts rounded requests up to 80 bytes
(`MAX_CACHED_MEMORY=11`, index=`size >> 3`). Compiler-measured layouts are:

| Structure | Plain bytes | Capstone bytes |
|---|---:|---:|
| `Bucket` | 72 | 128 |
| `HashTable` | 72 | 128 |
| `zval` | 24 | 48 |
| Zend allocation header | 24 | 48 |

Buckets and hash tables cross the cache boundary. Although `_emalloc` entries
stay nearly constant (340,307 plain, 340,340 Capstone stock), outer
`__libc_malloc_impl` entries rise from **18,128 to 102,145**. Source cache rules,
layouts and observed call counts support this explanation; it is not an
isolated cache-policy A/B experiment. An ABI-aware PHP cache policy is a
specific follow-up with a memory-retention tradeoff. No such policy change is
included in these measurements.

## Isolating libc optimization level

Only Capstone libc was rebuilt with `-O2`. Runtime and compiler-builtins
archives remain byte-identical; all **387 GCC application objects** are
byte-identical. PHP is relinked from the same application objects. All GCC
validation and PHP output/repeat checks pass. This is a diagnostic variant,
not a newly qualified default allocator build.

| Workload | Default libc `-O1` | libc `-O2` | Change |
|---|---:|---:|---:|
| GCC | 23,996,732 | 23,877,889 | -0.50% |
| PHP stock Zend, churn | 316,474,036 | 314,517,914 | -0.62% |
| PHP nested Zend, churn | 396,095,779 | 394,098,205 | -0.50% |

GCC's comparison to plain becomes **+32.11%**, versus +32.77% in the default
stack. The optimization-level mismatch is measurable but is not the main
explanation for the remaining overhead.

## Evidence integrity and reproduction

The first campaign exhausted the shared host `/tmp` inode limit. The original
failed attempts remain in the evidence. lbm/nab were replayed successfully on
`/var/tmp`, and Perl was replayed there before reaching its time limit. Plain
xz had already validated all outputs and all 12 raw windows; only report
creation failed. Its attribution was regenerated from those unchanged complete
windows, with an independent guest-exit verdict recheck. No interrupted
benchmark is promoted to PASS.

The measured scripts and their hashes, arm descriptions, per-run platform and
executable hashes, original failures and replay locations are in the evidence
record. The final tool commit additionally computes the relative staging path
from guest directory depth, replacing the measurement kit's fixed root-reaching
prefix. The new verdict checks both guest routes without discarding errors and
rejects historical input-stubbing records. The old Perl input-mutation helper
is deleted. Regression checks cover successful routes, failed/missing exits,
domain faults and both forms of recorded test intervention.

Rebuild through the normal entry points, with the frozen private source
packages already installed:

```sh
source capstone/tests/capstone-test-env.sh
JOBS=16 bash capstone/runtime/virtual/build-sdk.sh "$OUT" "$MUSL" \
  -DCAPSTONE_APPLICATION_DATA_BYTES=33554432 \
  -DCAPSTONE_APPLICATION_STACK_BYTES=1048576
# runcpu build: -O2, static musl; per-arm config/output_root and label.
bash capstone/ports/spec-cpu2017/tools/profile.sh \
  arms.json cap test 502.gcc_r "$RUN"
```

Use the recorded compilers/platform for reproducing these numbers; rebuilding
with newer compiler revisions is a new experiment. Private replay scripts and
all raw evidence remain in `/var/tmp/capstone-spec-php-20261105` and the paths
listed by `evidence_location` on p12. The baseline SDK hashes remain unchanged
throughout PHP relinking and the separate `-O2` experiment. Other application
ports and RTL were outside this task.
