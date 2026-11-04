# Experimental native domain exec

**Experimental, default off:** round 3 closes the demonstrated procfs memory
channel with a module resource policy. `scalar-memory.c` fails on the old module
and passes with an explicit open refusal on the new module. Full API
compatibility and p13 gate parity remain separate requirements.

The module parameter `dom_binfmt=1` enables the root-only loader; its default is
zero. This experiment uses the existing SDK v7 images and Linux 6.1 platform.
It has one domain thread, fixed pinned transport and no new seccomp filter.
Round 3 grants the initial cwd and safe stdio only; paths cannot follow symlinks,
traverse parents or cross mounts. Non-stdio inherited fds are inaccessible.
Inherited filters still apply. It refuses setuid/setgid and mutable remote
filesystem images. Stage binaries on guest tmpfs before executing them.

The dev replay also supports anonymous `pipe2` with `O_CLOEXEC` and
`O_NONBLOCK`. Both returned descriptors become grants before domain re-entry;
unsupported flags and results beyond the descriptor table are refused. Scalar
output clears tags, including the returned fd pair. Legacy read/write file
operations use the private bounce path. Initial stdio may include the first
serial terminal (device 4:64), as well as null/console, regular files and pipes;
this does not permit opening device paths. The pipe contract fails on the
prerequisite module and passes on the candidate. Hosted serial output is
restored, but file access outside the initial cwd and full gate parity remain
unresolved. Export `LANE_ROOT` to select a private artifact directory.

The 152-byte scalar trampoline forwards module-validated delegated calls through
private bounce memory, in addition to the protection bridge. The module retains
thread-self, anonymous mapping and whole-grant retirement, node reservation,
read/write and exit. The [design and exact syscall set](../../../../docs/design/virtual-capstone/native-exec.md)
explain the transport hold, CV_TOUCH, argument policy and security limitations.
Caught capability signals, threads, fork, capability ptrace and raw brk authority
remain unresolved. No production security qualification is implied.

The scripts use the lane's p12 artifact layout. Export the current runtime and
`capstone/tests/capstone-test-env.sh` under `~/lanes/laneAG2/src`, with an empty
`llvm` directory for environment validation. Build the adapter with the matching
prepared kernel, toolchain and revision-table ABI header. Copy platform-dev4
with `cp -rL` into `~/lanes/laneAG2/platform-native`, replace its adapter/module,
and refresh the platform manifest hashes. Export this test directory as
`~/lanes/laneAG2/contracts`, then run `build-contracts.sh` and `run-contract.sh`.
The latter disables binfmt_misc to prove execution and refusals use the native
loader. `NATIVE_PLATFORM` and `CONTRACT_OUT` select the platform and output;
the default output is `contracts-review`. The driver shuts its guest down.

For measurement, compile the unchanged `lqmicro.c` and `lq-hog.c` from branch
`perf/linux-question-micro`, directory
`capstone/ports/spec-cpu2017/tools/linux-micro`, into
`~/lanes/laneAG2/micro/{lqmicro-cap,lqmicro-plain,lq-hog}` with the baseline SDK
and plain compiler. `measure.sh` runs three pairs of exec 10/100 per arm. The
parent always uses the plain binary. The plain child is that same binary; the
launcher and native child are the same Capstone binary. Every valid invocation
must have rc=0, bad=0 and exactly one complete profiler window.

`run-micro.sh` derives from that branch's harness. Its additions stage a separate
child and record image/trampoline placement outside the measured windows.
`profile.py` extends the kit profiler with these images and preserves ambiguous
byte matches as an explicit bucket. The native placement is sampled from module
parameters after each window; launcher placement is probed by a separate
unmeasured exec. ASLR is disabled. Per-process cost is `(W100-W10)/90`, with
kernel-idle work reported separately. Historical Capstone-parent numbers use a
different denominator and must not be presented as this experiment's control.

## SPEC replay and gate invocation

On p12, source `capstone/tests/capstone-test-env.sh`, then run
`python3 contracts/spec-run.py BENCH ARM` from the lane artifact layout. BENCH is
one of `502.gcc_r`, `538.imagick_r`, `519.lbm_r`, `544.nab_r`; ARM is `plain`,
`launcher` or `native`. The runner reuses census4 executable identities and SPEC
oracles, copies only input data into private work directories, and rejects missing
or incomplete invocation/oracle manifests. It never edits the SPEC tree.
`SPEC_RUN_ROOT` and `NATIVE_PLATFORM` select fresh outputs and the real-file
platform. `test-spec-replay.py` exercises invalid-manifest refusals before VM boot.

The hosted/runtime runners accept `--native-exec`. They make the domain images
executable, enable the module binfmt, disable binfmt_misc fallback and record the
execution path. Their existing expectations remain intact: unsupported semantics
must remain visible as failures. The shared gate driver accepts `NATIVE_EXEC=1`
only for these two gates, plus `MODULE_ARGS` for the runtime module. On p13 submit
each arm through `~/retire-gate/gate-queue.sh bash .../gates/run-arm.sh OUT`.

## M4 round 2 measurements

Saved p12 results at `~/lanes/laneAG2/final` were independently checked against
the census invocations, SPEC oracle commands, executable hashes and complete
profile sums. All twelve benchmark/arm combinations report PASS. The executable
is identical between launcher and native arms. The measured module hash is
`b66ff16ca270d402dc4f78254e44c47112be52d5b2383438cb201f4b10d804ca`.
The native contracts in `contracts-wire-final-r2` report 24 PASS lines,
`native contract failures=0` and `unload_rc=0`.

Matched lqmicro exec uses a plain parent and three pairs of 10/100 children.
Per-process attempts are `(W100-W10)/90`; these are QEMU attempt counts, not time.

| Child path | Mean attempts/process | Observed range |
|---|---:|---:|
| Plain | 198,348.70 | 197,678.19–199,063.96 |
| Launcher | 2,581,932.94 | 2,553,448.14–2,637,190.64 |
| Native | 1,161,382.65 | 1,161,017.02–1,161,681.29 |

Native reduces the matched launcher cost by 55.019%. The largest reductions
include kernel `do_seccomp` (163,113 attempts/process), kernel `__memset`
(80,956.11), kernel `bpf_convert_filter` (57,262), and launcher `memcpy` (49,850).
Native remains substantially above plain Linux process cost.

The full test-input profiles below use `prof-northstar.py` from
`tools/quick-census`. P is program work, K is boundary work and H is excluded
harness/idle/share work. Percentages use the plain denominator `P_plain+K_plain`.
H is retained separately; it must not be treated as a boundary saving.

| Benchmark | Launcher program % | Native program % | Launcher boundary % | Native boundary % | Native − launcher total attempts | Native − launcher K | Native − launcher H |
|---|---:|---:|---:|---:|---:|---:|---:|
| gcc | +25.6769 | +25.6704 | +48.0614 | +39.3503 | −2,762,037 | −2,763,029 | +3,077 |
| imagick | +3.3219 | +3.3200 | −2.4888 | −2.9570 | +7,311,100 | −904,346 | +8,219,003 |
| lbm | +2.9253 | +2.9253 | +3.7482 | +3.4150 | −20,024,173 | −22,122,596 | +2,099,801 |
| nab | +7.0869 | +7.0869 | +0.0746 | +0.0003 | −5,058,233 | −5,116,292 | +59,419 |

These are full SPEC test inputs, not reference workloads or rate results.
Imagick's total rises despite a lower K because H rises. No requested benchmark
is excluded in this single-thread configuration. Perlbench subprocess paths and
threaded/OpenMP configurations needing clone remain outside coverage.
Retained evidence: `final/verified-results.txt`, `final/measurement-summary-continued.txt`,
`final/compare-*.txt`, `final/buckets-*.txt`, and the per-arm invocation, oracle and
profile JSON files. Full p13 gate parity and the security boundary remain separate
acceptance requirements.

## Scalar-memory negative probe

Compile `scalar-memory.c` with the same Capstone SDK, copy the resulting image to
guest tmpfs, and execute it directly with `dom_binfmt=1` and binfmt_misc disabled.
It uses positioned I/O to find the private trampoline in `/proc/self/maps`, reads
its first instruction through `/proc/self/mem`, and writes the same word back.
It does not execute modified instructions or change another process.

The continued-round p12 run reports:

```
SCALAR_MEMORY base_found=1 opened=1 read=4 write=4
FAIL scalar host memory accessible from delegated I/O
probe_rc=1
unload_rc=0
```

Evidence: `~/lanes/laneAG2/scalar-memory-pread-result` and
`logs/scalar-memory-pread.log`. This rebuilt module used the same implementation
with a reusable prepared kernel; its SHA-256 is
`fce475cd7464aacaec820803300cb1854f01d1e76395b8dab9d912da343fdeab`.
The first probe using stdio reported `base_found=0` because ordinary module read
does not implement the procfs legacy read operation; that is an inconclusive
probe, preserved in `scalar-memory-result`. An explicit EACCES/EPERM on the memory open now returns 0. An unexplained
non-reproduction still returns 2. The historical failed run remains the required
negative control; pathname refusal alone is not the full qualification.


## Round 3 resource probes

`resource-probe.c` uses the wrapped transport boundary to send raw wire requests,
so refusals are checked in the module rather than only in libc. Compile it with
`-Wl,--wrap=__capstone_vm_delegate_transport` and the runtime include directory,
alongside `scalar-memory.c`, using the same SDK as the native contracts. Compile
`dac-driver.c` with the plain static RISC-V compiler. It retains UID 0 for the
experimental loader but drops all capabilities and disables root capability
regain across exec, to test ordinary file permissions without DAC bypass.
`run-security.sh` creates a private guest fixture with symlink aliases, a nested
proc mount and device nodes. It passes inherited proc and directory fds, including
unsafe fd 0, and verifies policy refusals plus ordinary-file positive controls.
ASLR is disabled for two identical probe execs; the second tests the recorded
scalar trampoline address, and the runner verifies that placement stayed equal.
The fixture needs the root-only native experiment. It tears down its guest.

The revised native contracts check overlapping pathname/stat output buffers and
inaccessible non-stdio descriptors. They retain owned VM, malformed wire, normal
file I/O, default/ignored signals, Linux termination and malformed-image checks.
The [authority audit](../../../../docs/design/virtual-capstone/native-exec.md)
lists the complete syscall and resource boundary, deliberate incompatibilities,
trust assumptions and vectors tested. The launcher file policy is unchanged.

## Round 3 final qualification

The final implementation is commit `26e5a1c13341`. The p12 module SHA-256 is
`73458974f707ef39dd2647d9364fa9ee4fba2afc7ebca9b8103a039fccd001a9`;
the p13 gate module, compiled with 32 supervisor slots, is
`15dc040e1e0a5688567e6f0551a226301eacdba6c73528e4f8568380660bcef6`.
Only `round3-final5-*` results identify this final build; earlier final/qualified/
release directories contain superseded candidates. No new seccomp filter is
installed. The trampoline remains 152 bytes.

`build-contracts.sh` also builds the security probes and the plain DAC/alarm
helper (`DAC_CC` selects its Linux-UAPI-capable compiler). Rebuilding through
this entry point produces identical probe binaries. `run-security.sh` reports
94 main resource checks, scalar-memory refusal, unsafe inherited stdio refusal,
two DAC checks and three descriptor-limit/recovery checks passing. The actual
trampoline-address probe and module unload pass. `run-contract.sh` reports 25
PASS lines, zero failures and successful unload. The descriptor probe crosses fd
258, which must not be confused with the prepared-syscall marker. Separate old
build controls reproduce scalar-memory access, DAC bypass, shared-FS acceptance
and the fd-258 dispatch failure. Six invalid SPEC manifests are refused.

Final matched lane-N lqmicro exec, three 10/100 pairs with a plain parent:

| Child path | Mean attempts/process | Observed range |
|---|---:|---:|
| Plain | 198,875.18 | 198,437.38–199,610.01 |
| Launcher | 2,582,532.63 | 2,553,282.42–2,637,956.84 |
| Native | 1,164,785.77 | 1,164,609.20–1,164,980.80 |

Native saves 54.898% (2.217x), retaining the earlier approximately 55% reduction.
Module work rises by 8,231.22 attempts/process while kernel work falls by
1,281,597.73 and launcher work by 142,607.06. Largest function reductions are
`do_seccomp` 163,113, `__memset` 80,874.39, `bpf_convert_filter` 57,262 and launcher
`memcpy` 49,850. These are instruction attempts, not elapsed time or hardware
performance. This paired experiment does not isolate the broker's marginal cost.

All four full SPEC test inputs pass through direct native exec with binfmt_misc
disabled. Original executable hashes, invocation identities, oracle commands,
staged reference-input hashes and complete profile sums are verified. Imagick's
immutable expected-image input is staged inside its cwd grant, with original and
translated argv retained. A is the saved Round 2 launcher control in `final/runs`;
B is the final Round 3 native build. These are one full sample per input, not a
new paired statistical timing experiment.

| Benchmark | A total attempts | B total attempts | B − A | B − A boundary K | B − A excluded H |
|---|---:|---:|---:|---:|---:|
| gcc | 56,585,681 | 53,282,891 | −3,302,790 | −2,926,933 | −374,097 |
| imagick | 201,730,077 | 196,508,452 | −5,221,625 | −5,453,449 | +235,543 |
| lbm | 7,115,288,167 | 7,091,282,688 | −24,005,479 | −23,495,704 | −508,488 |
| nab | 7,375,075,610 | 7,369,195,527 | −5,880,083 | −5,408,849 | −469,960 |

The p13 launcher and native hosted/runtime arms both completed through the shared
queue, one arm at a time. Both guests log `supervisor_slots=32`; neither logs
`thread create refused`. The comparison reports `DIFFERENT`, with all three
hosted and all 25 runtime exit rows present. Launcher hosted passes 93/93 and
runtime passes. Native hosted reports 41/48 with missing virtual checks; runtime
fails file operations outside the cwd grant, thread creation, application rows
and expected diagnostics. For example SQLite cannot open `/tmp/virtual.db`,
mruby cannot open `/tmp/virtual-ruby`, and Perl cannot open `/dev/null`. Expectations
were not relaxed. Seven other landing gates have no native qualification here.
M4 remains **NO-GO for integration**, despite closing the demonstrated memory
channel under the documented trust assumptions.

Retained p12 evidence: `~/lanes/laneAG2/round3-final5-{security,contracts,spec,measure}`,
`round3-results/{verified-results.txt,spec-table.json,compare-*.txt,buckets-*.txt,northstar-*.txt}`
and the negative-control logs. Retained p13 evidence:
`~/lanes/laneAG2/round3-gates-{launcher,final-native}` and
`logs/round3/gate-comparison.{txt,json}`. VM disks, copied inputs, build trees and
platform copies are scratch; result logs, hashes and profiles are retained.
