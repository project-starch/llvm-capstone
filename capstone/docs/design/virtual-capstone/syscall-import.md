# Syscall work imported into the copyable-only prototype

The narrow Point 2 prototype is now implemented and qualified on
`prototype/copyable-only`: kernel syscall providers, checked copies, private FD
policy, deterministic guest races and paired instruction measurements. It remains
default off, root-only, single hart and unpinned. Full native Linux task
integration (Point 3) is separate. The sections below preserve the sequence and
boundaries of the hosted/model imports; the final section records integration.

## Initial frozen source and selection

The source snapshots were clean when inspected. No active lane checkout was
modified. Selected files were read from Git objects, not from mutable worktrees.

| Work | Inspected revision | Disposition |
|---|---|---|
| `native/syscall-fastpath` (NS) | `73730ea7b2736ae9355a13982ecb5589f69286c1` | Shared dispatcher, descriptor table, interface headers and hosted tests imported. Kernel fast paths and SDK selector deferred. |
| `runtime/mapping-cache` (MC) | `c0691a6e8c6e853fbd42d71712aea1bcd6980da0` | Deferred: persistent pins and the old linear grant ABI are prerequisites; the pool expressly refuses `unpinned=1`. |
| `native/race-test` (RT) | `d034e369da8c90c649ad094662cd893e142fca3e` | Guest race matrix is required for subsequent integration; its native kernel/model dependencies are not imported. |

The selected NS revision contains the shared U checked-copy and N context
reference dependencies. Individual source Git blob IDs and hashes are in the
[import record](../../../runtime/virtual/tests/copyable-only/results/syscall-import.json).
The C sources and interface headers are imported verbatim. Test instructions are
adapted for this checkout; ASan/UBSan failures are fatal in all three runners.

## What is reusable now

`common/native-syscall.c` captures one authenticated request epoch, prepares
private input/output snapshots, calls mandatory resource and checked-copy
providers, accounts for output prefixes and irreversible effects, and prevents
replaying an issued epoch. The 106-entry descriptor inventory does not itself
authorize a Linux call. Unknown calls and missing providers fail closed.
`ns_session_admit` is also the epoch primitive used by the source lane's
specialized kernel paths; those paths themselves are not part of this import.

`virtual/checked-copy.c` holds the mapping, enters the backend's exclusion
window, authenticates each page-bounded copy chunk, drains before leaving,
resolves faults without dummy stores, and rechecks original capability authority
after faults or rescheduling. A mapping hold alone does not preserve object
liveness: the `check` callback and exclusion from revocation are mandatory.

The hosted context reference supplies identity, epoch and register-state
callbacks for these tests. It is an executable specification with synthetic
capabilities, not an authority decoder and not a replacement for QEMU or Linux.
No legacy ownership instruction is introduced by these provider-independent C
sources. The actual capability representation remains the current copyable one.

## Why the measured NS fast path is not a direct cherry-pick

NS's kernel fast paths (`ef5f84db9f94`, subsequently tested in `73730ea7b273`)
require its native binfmt/resource broker, kernel-owned register frame and escape
epoch, protected slot transport, `CSCHECKR`/`CSCHECKW`, checked-copy module glue,
and mprotect wrapper. Its QEMU pin is `ede5e467462f258775f1d1f2fb989eab3665e73f`;
the initial copyable-only import lacked checked-access instructions and
deliberately rejected legacy ownership/control instructions. The model checkpoint
below ports checked access separately. Replacing the QEMU pin or grant ABI with
the lane's would discard the copyable-only milestone.

The source lane's measurements are useful motivation, not results on this
branch. Its own integration documentation still identifies partial-protection
compatibility and guest scheduling/reuse qualification as landing conditions.

MC's pool retains pinned backing and selects heap arenas using `a->linear`.
This branch removed that field and changed the CV_ADD ioctl payload. Porting the
pool to unpinned backing needs a separate resident-page stabilization/clearing
protocol, fresh ancestor identities, revocation-before-reuse and futex/reuse
semantics; changing its admission test to accept copyable grants is insufficient.

## Copyable model checkpoint

The copyable-only model now has default-off `CSCHECKR/W` and protected context
transport. This is the processor-side prerequisite, not an active kernel
syscall path. The shared dispatcher still has no production kernel providers.
The port preserves the removed decoder entries, NONLIN-only restoration and
ordinary MIP device handling; it does not import the NS QEMU pin wholesale.

Transport versions 5/6 deliberately differ from NS versions 1/2. Besides
omitting MIP, they save the immutable escaped GPRs separately from mutable
resume/reply slots. Gemini Argon identified the latter distinction during
review: after export/import a prepared reply must not become authority under
the original request epoch. The regression now rejects that substitution.
Import also validates capabilities without trapping or consuming a rejected
image, and both transport variants check the exported epoch/caller privilege.
See the [model contract and commands](../../../capstone-qemu/tests/capstone-checked-copy/README.md)
and [qualification record](../../../runtime/virtual/tests/copyable-only/results/checked-model.json).

The current launcher calls `capstone_delegate_seccomp()` and excludes its own
private descriptors in the delegated service. A new in-kernel call must retain
these policies: `fget(domain_scalar_fd)` alone is insufficient. NS's native
resource broker and its refusal of inherited Seccomp are dependencies to adapt,
not optional performance cleanup. Do not disable the existing launcher filter
to make a microbenchmark pass. Kernel transport integration must also include
all newly tagged snapshot slots in lifetime-root accounting and sweeps.

## Integration gates (completed for the narrow prototype)

1. Processor-side checked-access/transport is qualified as above. Connect its
   versioned ABI to kernel-owned contexts, including frame-root accounting;
   retain the removed-instruction and Linux page-lifecycle regression gates.
2. Connect the shared dispatcher to kernel-owned contexts and held file
   identities. Start with the source lane's narrow scalar and flat read/write
   set. Handle partial mprotect, signals/restart and unsupported pointer graphs
   explicitly; never forward scalar addresses as authority.
3. Port RT's deterministic revoke/reuse and COW/fault-retry scenarios, including
   deliberately broken recheck/exclusion/hold controls. Run the existing
   copyable-only safety suites and paired syscall budgets on the same binaries.
4. Only then enable the specialized NS paths and compare them with the generic
   path, including authority refusals and completed-call counters.

## Kernel consumer qualification — 2026-11-06

The later integration completes the narrow Point 2 prototype on this branch.
The descriptions above retain the boundaries of the earlier hosted/model
checkpoints. Selected NS kernel providers, binfmt, resource policy, explicit SDK
stub and Linux mprotect wrapper are now adapted from the same frozen revision;
RT hooks and its frozen 41-case oracle are adapted from the RT revision above.
MC remains deferred. No active lane checkout was modified and no legacy
ownership ABI was restored.

Admission now checks the model ABI/transport version and required kernel export.
Private file references, inherited-Seccomp refusal, immutable authority snapshots,
epoch accounting, mapping retirement/holds and fault rechecks are connected.
Metadata protection changes are refused; page-subrange mprotect semantics of
this branch are retained and covered. Both generic and specialized paths pass
51 functional checks and 10 entry/slot-eviction checks each; both pass all 41
RT cases, including 17 negative controls. Pinned/unpinned launcher regressions
pass 22 checks each, and all five raw platform admission cases pass.

Matched measurements show 54%/74% fewer system-wide attempts for 1-byte/64-KiB
reads and 40%/68% for writes versus the launcher. Cached launcher getpid remains
faster. Kernel/module work decreases for large buffers but increases for small
writes; the result is not simply fewer kernel instructions in every case.
See [reproduction, scope and measurements](../../../runtime/virtual/tests/native-integration/README.md)
and [binary qualification](../../../runtime/virtual/tests/copyable-only/results/native-integration.json).
Gemini Argon reviewed the integration; admission, snapshot restoration and
terminal cleanup findings were addressed and guest-tested.

Point 1 remains independently qualified. Point 2 is complete at the explicitly
narrow, default-off single-hart prototype scope. Point 3 remains: the current
same-task ioctl trampoline is not a complete native architecture return path or
Linux process/signal/restart integration. Bounce copies and unsupported syscall
families remain measured follow-up work; no native full SPEC result is claimed.
