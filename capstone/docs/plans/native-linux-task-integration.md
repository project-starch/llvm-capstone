# Native Linux task integration

Status: first task-return prototype implemented on `native/linux-task-integration`,
based on measured `experiment/native-gcc-preliminary` at `a13e2ffaa9a2`.
[Verified tests and matched GCC comparison](../../runtime/virtual/tests/native-task/README.md)
cover core entry, default signals, blocking, accounting, copy races and output.
The full first-milestone acceptance list remains open: policy integration,
exhaustive lifetime and full state/raw-platform qualification are not complete.
See the [architecture decision](../design/virtual-capstone/native-linux-tasks.md).

## Prerequisite: verify the entry/escape contract

Reuse and extend the existing timer/fault qualification fixtures to establish
CSR unprivileged entry, timer interruption and unmapped/COW access behavior on
the pinned model. Record event publication, saved PC/retry PC, continuation after
CSR completion, IRQ/preemption state and pending-work visibility. Check the
actual Linux 6.1 entry/return path and context-tracking/RCU contract against that
record. Do not infer a conventional trap frame from the fact that CSRUNV returns.
This prerequisite is not a fresh redesign of the ISA or a new capstone_enter ABI.

## First implementation milestone: task-owned entry and return

Implement a default-off in-tree Linux task/architecture backend. Keep the current
prototype as a separate baseline, not as an automatic fallback. The first native
backend may use the existing privileged context transport, checked-copy engine,
restricted providers and root-only, single-hart scope. Its defining behavior is that a
Capstone task runs without an ioctl trampoline or control FD owning the context.

Before coding entry, write the concrete state-transition table for task exit,
preemption, signal delivery, service completion/restart and failed exec. Identify
the exact Linux 6.1 architecture hook and demonstrate that interrupt/preemption
state, return work and context ownership are valid at every transition. Freeze
the signal-frame authority/restart contract even if caught-signal support lands
in the next gated patch. Fail explicitly on unsupported operations.

Implementation areas:

1. Add task-owned protected execution state and mm-owned namespace lifetime,
   with reference-counted cleanup and correct pre-/post-point-of-no-return exec
   failure behavior. Move the relevant context machinery in-tree; do not tether it to a userspace FD.
2. Add architecture return/escape handling. Restore the complete protected state,
   enter through the privileged backend, and return through Linux's scheduling
   and signal work. Preserve scalar-task behavior when the feature is disabled.
   Kernel save areas must remain nonpageable and safely accessible during slot
   eviction. Integrate context tracking, RCU and accounting, and specify the
   non-authoritative scalar policy/debug view of the protected frame.
3. Adapt the `.dom` loader to install native task state directly. Do not install
   `dom_trampoline`, fabricate a scalar user return, or require a control descriptor.
   Keep the existing loader-established exchange/metadata capabilities and startup
   register contract for the first GCC gate, under task/mm ownership.
   Coordinate mapping/TLB changes and OOM/final-mm teardown with context state;
   select concrete hooks only after checking the model/VM contract.
4. Route supported services through the existing authority-preserving providers.
   Keep resource policy and buffers initially. Separate this change from globally
   switching GCC/musl to the direct register ABI.
5. Remove dependence on the global module execution lock across sleep and define
   per-task/per-mm serialization. Transport slots may remain resident but must be
   evictable while the task blocks; task/mm references outlive any asynchronous kernel work.

Acceptance gates:

- Source and runtime evidence of no ioctl/trampoline/control-FD dependency.
- Correct task/PID/exit/wait identity and user/kernel accounting; explicit syscall
  audit/seccomp/tracing routing for the admitted ABI, without manufacturing tags.
- Compute-only execution remains schedulable; timer progress, SIGKILL and default
  SIGSTOP/SIGCONT work without a cooperative domain service call.
- Blocking pipe I/O and two Linux tasks sharing one hardware slot make progress;
  terminating a blocked task neither leaks state nor replays committed I/O.
- Lossless register, tag, raw payload, supported bounds/CSR and FP/FCSR preservation
  across preemption, eviction and restore; stale namespace/epoch restores refused.
- Repeated exec/exit, failed exec and task kill reclaim contexts, namespace
  references and mappings; scalar Linux processes remain unaffected.
- Existing authority, checked-copy, race and raw-platform negative suites pass
  against the new backend. No legacy ownership instructions return.
- Same GCC test and reference output pass for the admitted syscall set. Unsupported
  signal/process operations remain visible; GCC output alone is not qualification.

Caught user handlers and rt_sigreturn are explicitly outside this first gate;
unsupported registration returns an explicit error. Default-fatal and job-control
signals must work. Gate A adds caught signals; a successful small unsignalled GCC
run does not stand in for it.

A passing milestone is a task-return prototype, not full Linux compatibility,
SMP readiness or a performance claim. Do not mark it complete just because a
`capstone_enter` syscall or a refactored dom_ioctl compiles.

## Subsequent gates

A. Capability signal delivery, validated sigreturn/continuation handling, nested
signals, alternate stacks, siglongjmp/setcontext and restart/partial I/O. Tests
must include pending signals at every request commit and reply boundary.

B. Direct native register ABI in musl/SDK with typed mapping results and capability
layout adapters. Compare the same application builds and inputs, separating
exchange marshalling from native task entry. Preserve ioctl-based measurements
as history; do not combine results across unrelated allocator/compiler builds.

C. Linux descriptor/task resource integration, inherited descriptor policy and
broad syscall families. Use kernel-buffer-safe operations first; add checked user
iterators only with revocation/fault/scheduling/graph-mutation races tested. Add
futex, clone/exec/fork and retained-pointer APIs with explicit lifetime contracts.

D. SMP/migration/hotplug, multi-process shared-memory namespace behavior and RTL
qualification. Hardware slots remain caches, not independently owned task state.

The order of A and B may overlap once the task state contract is fixed, but neither
can silently assume the other's unimplemented signal or pointer semantics.

## Evaluation and non-goals

Use fresh-guest native-versus-current-prototype runs, matching kernel/model,
binaries, input, allocator and output oracle. Record kernel/module/user attempted
instructions, transitions, copied bytes, faults and mappings separately. Include
compute-only, scalar syscall, short/long I/O, blocking/preemption and full GCC test.
Do not infer speedup from interface cleanliness or from replacing the word ioctl.

No new syscall is required merely to rename the existing driver operation. A
future setup ABI needs its own purpose and compatibility review. vDSO work and
allocator retention experiments are independent optimizations, not prerequisites
for this task-return milestone.


## Reviewed first musl register-ABI slice

The syscall microbenchmarks justify a bounded experiment, not a forecast for
whole-GCC speedup. Keep kernel, model and task-entry mode fixed when comparing
libc ABIs. The current GCC test exercises `writev`; its proportion is not yet
quantified and the direct backend does not admit vectors.

Start with `read`, `write`, `pread64` and `pwrite64` for **0 < count <= 65536**.
Select the route once at `dl_transfer` entry, before effects. The direct route
issues exactly one request with the original count and returns its short result
or error as final. Zero-length and larger requests retain the existing exchange
route for the entire operation, including later iterations. Do not reduce a
large request to 64 KiB to force the fast path or switch its tail to direct I/O.
`close`/`lseek`, pathname and structure calls are subsequent separate slices.

Do not insert a direct call under the existing regular-file refill loop and
infer retry safety from `result < requested`. A read can consume more bytes
than checked copy-out publishes; an additional read may silently skip data.
Oversize exchange behavior stays as existing compatibility behavior; retaining
it does not claim that all old short-I/O semantics are already ideal.

Activation requires a build opt-in and a validated loader-announced version and
feature record. Design the record/decoder compatibility explicitly: the current
launch v2 structure has no native-syscall feature field. Missing, unknown or
incompatible announcements leave direct routing off. Do not discover support
by issuing an unknown selector, and never retry via exchange after a direct
call returns an error. Existing caught-signal, process and seccomp restrictions
remain; this slice does not silently broaden backend admission.

Preserve identity/clock caches, local signal/thread intercepts and cancellation
entry checks. Use the tag-preserving stub for every active argument and zero
unused slots explicitly. Do not convert pointers to cursor integers or strip
tags from malformed scalar arguments to make them pass validation. Review
negative scalar and positional-offset code generation against the actual ABI.

A direct completion must reset stale round status and record a completed
request before the existing cancellation-result and queued-signal delivery
logic runs. It must not cause replay on EINTR. In particular, do not blindly
call `__capstone_signals_take`: direct entry publishes no exchange signal block,
and that reader does not consume/clear the shared count. Importing fresh signal
handover requires an explicit publication protocol. Full caught-signal/restart
support remains a separate gate; preserving bookkeeping is not qualification
of currently unsupported pthread cancellation during caught-signal delivery.

`writev`/`readv` remain on their existing vector path. Direct vectors require
protected snapshots of nested capabilities and their lifetimes, preserving one
vector operation; do not implement them as a loop of independent read/write
syscalls. Mapping/capability returns, exit, futex and signal operations retain
their existing interfaces in this slice.

Required checks before adopting the slice:

- ABI-announcement absent/old/unknown cases execute no direct selector; an
  injected post-entry error causes no alternate-transport retry.
- Boundary sizes 0, 1, 4095, 4096, 65535, 65536, 65537 and above 1 MiB;
  file EOF, short pipe reads, nonblocking EAGAIN, large regular-file completion,
  positional offsets, bad FDs and committed file-position/output checks.
- Positive-short copy-out and interruption cause no hidden refill/replay; stale
  signal-block contents are not delivered again. Preserve errno conversion and
  cancellation completion state, with unsupported full-signal cases labelled.
- Existing 51 authority/ABI tests and 41 copy-race scenarios, plus tests reaching
  the new musl routing hook (old direct-probe success alone cannot cover it).
- Strengthen the microbenchmark oracles with patterned input, exact read
  contents and exact write output length/content, all checked outside the
  counted window. Separate first-use allocation from warmed calls; retain a
  generic-dispatch control and report repeat/interrupt variability.
- Ordinary identity/time calls still use their local cache. Count admitted
  direct and exchange calls by syscall outside profiling windows, including
  `writev`, so a passing GCC run proves the new route was used.
- Rebuild baseline and candidate GCC from identical source/compiler/flags with
  only the musl routing option changed. Run exchange/direct/direct/exchange in
  fresh guests with `native_task=1` fixed, exact assembly-output validation and
  kernel/other instruction counts. The existing entry-toggle measure-gcc.py is
  insufficient for this libc comparison. Record a neutral/regressive result
  if observed; do not substitute the microbenchmark reduction for it.


Astra xhigh and Gemini Argon reviewed this slice against the implementation and
measurement evidence on 2026-11-06. Cross-review rejected blanket 64 KiB chunking,
live ABI probing, scalar-tag stripping and post-read refill. Both endorsed the
four-call scope, single-completion rule and separate libc-ABI GCC comparison.
This is an implementation plan; no musl routing change is included yet.

Review evidence: `$CAPSTONE_TMP_ROOT/native-task/musl-review/musl-register-review.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/musl-register-review.tar.gz`), SHA-256
`de4f52af498f6feaffffa9f37a021331cb2a29e78dcc26c86aca2570f58e3d32`.

### First musl slice: verified implementation

The opt-in small-I/O slice is implemented and qualified in the
[runtime record](../../runtime/virtual/tests/native-task/README.md#opt-in-musl-small-io-migration).
Launch v3 announces support, scalar arguments are explicitly canonicalized without
laundering tagged capabilities, and direct short/error completions are final.
Fast/generic/v2 fallback gates and both pipe/regular-file directed race suites pass.
Four matched GCC runs show +0.0411% total attempts, not a speedup; only one 69-byte
read in the separate route trace reaches the slice. This does not close the broader
Linux policy, signal/cancellation, SMP or all-native syscall milestones.


## Native subprocess closure

The direct capability-image spawn/wait slice is now implemented and passes the
[QEMU process gates](../../runtime/virtual/tests/native-task/README.md#native-child-processes).
The implementation uses a dedicated authenticated `kernel_clone` bootstrap,
pinned native-only exec, inherited resource capsules and stable-PID waits.
The original design requirements below remain the review checklist. A separate
hush port replaces its NOMMU vfork/reexec boundaries with posix_spawn; shell and
full Perl output qualification are distinct from the process fixture gate.
Caller-replacing exec, general fork and caught handlers remain unsupported.


The unchanged SPEC Perl test driver uses musl `popen`, which requests
`posix_spawn("/bin/sh", ["sh", "-c", command], actions)` and then waits. The
existing spawn wire format already carries argv/envp, ordered file actions,
cwd actions and signal attributes. No new wire ABI is required to start this
work. Cwd, pipes and default/ignore signals are independently qualified; they
are not full process support. The new hush shell port is qualified separately from the kernel service.
A conventional shell also needs fork/vfork or an explicitly ported spawn model.

The first process slice should execute a capability image directly, inherit
its original authority root separately from cwd, apply file actions in order,
and wait for the exact child. A tiny capability child must qualify this before
a shell is admitted. Required implementation and acceptance gates:

- Add an explicit authenticated kernel child bootstrap. Do not exempt arbitrary
  `kernel_clone_args.fn` calls from the existing protected-task clone refusal.
  The child must exec into a fresh mm and protected context before any user
  instruction; failure must terminate without resuming copied protected state.
- Carry a refcounted heap request/completion across parent interruption. Keep
  child PID references, handle exec failure and reaping, and release `vm_lock`
  during child exec and waits. A completion on the parent's stack is unsafe.
- Open and authorize the executable once and execute that pinned file through
  the normal bprm lifecycle, including LSM, noexec and set-id checks. Calling
  `kernel_execve` on a previously checked pathname reopens a rename/symlink race.
- Inherit only admitted descriptors and their CLOEXEC state. The current loader
  admits only stdio and derives a new root from cwd; neither behavior is correct
  for an inherited child. Evaluate each open/dup/close/chdir action against the
  evolving child state, not all paths against the parent's initial cwd.
- Keep in-place exec separate until the architecture return contract explicitly
  supports context replacement. Its current synchronous callback forbids
  releasing the old state; a successful exec cannot return to freed context
  memory or bypass the `riscv_capstone_return` state-identity check.
- Qualify parent death during bootstrap, all failed-exec stages, descriptor and
  PID reuse, ordered actions, wait/status behavior, and authority escape attempts.
  Then supply a capability-built shell with a supported process model, or a
  separately reviewed Linux confinement mechanism covering scalar descendants.
  An ambient root shell, launcher fallback, or broader filesystem grants do
  not satisfy the native authority contract.

For Linux 6.1, `user_mode_thread()` is a possible non-PF_KTHREAD bootstrap
primitive, but needs the dedicated admission check above. Pinned-file execution
requires factoring `kernel_execve` into an internal execveat-style entry with
kernel argv/envp and `AT_EMPTY_PATH`; its existing pathname-only interface is
insufficient. `kernel_wait4` takes a userspace status pointer. A narrow exact-PID
wait may use `kernel_wait` with initialized kernel status storage; general
wait/options/rusage need a kernel-buffer helper and checked result publication.
These APIs do not by themselves establish the lifecycle or authority guarantees.

Astra xhigh and Gemini Argon reviewed these boundaries. Direct-image spawn
alone does not qualify `popen`, shell pipelines/substitutions, or the full Perl
test. Preserve original workloads and require official output comparison.

## Remaining Perl pointer serialization boundary

The old native `pack.t` ran through assertion 242 and faulted on its
first big-endian pointer round trip (assertion 243). Separate probes confirm
that native aligned `p`/`p<` work while byte-reversed, unaligned and UTF-8-expanded
pointer fields lose their tags. This is an application representation problem,
not evidence that the kernel should reconstruct authority from scalar bytes.
[Reproduction and pinned inputs](../../runtime/virtual/tests/native-task/spec-perl-pack-blocker.json).

The implemented [Perl port](../../ports/perl/provenance/README.md) retains exact
tagged capabilities in private per-scalar metadata while preserving upstream
encoded bytes. It explicitly propagates supported value operations and invalidates
metadata before raw mutation or untrusted callbacks. Unpack validates the complete
field and representation before copying retained authority. It does not reconstruct
capabilities from addresses or serialized integers. Native XS aliases and interpreter
thread cloning are outside its supported configuration.

Frozen revision 06 passes 89 pointer and 101 independent controls in both native
ABI modes, the unchanged 14,708-assertion pack test in direct mode, and host
sanitizer/allocation-failure gates. A review-found dangling metadata head on
allocation failure was reproduced against revision 05 and fixed before freezing
06. The full revision-06 run then exposed a retained freed-scalar lookup in
`sort.t`; revision 07 preserves the original core exception path. Revision 07
passes both ABI focused gates and the complete original Perl benchmark with
both official comparisons. The [final ten-entry record](../../runtime/virtual/tests/native-task/spec-final-qualification.json)
includes all nine other entries on the same final kernel. This establishes
functional correctness for those inputs, not a performance comparison.

The port uses the new `capstone_provenance_v2` src.alt package and 69 freshly
compiled application objects per ABI. Keep the existing frozen source package intact; the
`.cpu2017.v1.0` suffix denotes SPEC compatibility, not a port revision. Test
inputs and official comparators stay unchanged. Qualify forged/damaged bytes,
partial fields, stale metadata, encoding/copy aliases and cleanup before the
unchanged 14,708-assertion pack test and the complete Perl test workload. A
kernel-only repair cannot establish that application-port result.
