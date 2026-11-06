# Native Linux task integration

Status: design stage on `native/linux-task-integration`, based on the measured
`experiment/native-gcc-preliminary` at `a13e2ffaa9a2`. No runtime behavior change
or new native-task qualification is claimed. See the
[architecture decision](../design/virtual-capstone/native-linux-tasks.md).

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
