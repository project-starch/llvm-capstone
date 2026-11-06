# Syscall work imported into the copyable-only prototype

The first import is a hosted, provider-independent foundation for the in-kernel
syscall milestone. It supplies the actual shared dispatcher and checked-copy
algorithm plus their executable contracts. It does not activate a native syscall
ABI or change the current module, launcher, SDK or QEMU execution paths. A faster
application syscall path is not claimed from these tests.

## Frozen source and selection

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
our copyable-only model lacks those checked-access instructions and deliberately
rejects legacy ownership/control instructions. Replacing our QEMU pin or grant
ABI with the lane's would discard the milestone just qualified here.

The source lane's measurements are useful motivation, not results on this
branch. Its own integration documentation still identifies partial-protection
compatibility and guest scheduling/reuse qualification as landing conditions.

MC's pool retains pinned backing and selects heap arenas using `a->linear`.
This branch removed that field and changed the CV_ADD ioctl payload. Porting the
pool to unpinned backing needs a separate resident-page stabilization/clearing
protocol, fresh ancestor identities, revocation-before-reuse and futex/reuse
semantics; changing its admission test to accept copyable grants is insufficient.

## Next integration gate

1. Port and qualify the authenticated checked-access/slot protocol against this
   branch's copyable-only frame and lifetime rules. Keep removed instructions
   illegal and preserve the qualified Linux tag-aware page lifecycle.
2. Connect the shared dispatcher to kernel-owned contexts and held file
   identities. Start with the source lane's narrow scalar and flat read/write
   set. Handle partial mprotect, signals/restart and unsupported pointer graphs
   explicitly; never forward scalar addresses as authority.
3. Port RT's deterministic revoke/reuse and COW/fault-retry scenarios, including
   deliberately broken recheck/exclusion/hold controls. Run the existing
   copyable-only safety suites and paired syscall budgets on the same binaries.
4. Only then enable the specialized NS paths and compare them with the generic
   path, including authority refusals and completed-call counters.

Point 1 (tag-aware page maintenance without persistent application-page pins)
remains independently qualified. This import starts Point 2; it does not mark it
complete and does not implement Point 3 (native Linux task integration).
