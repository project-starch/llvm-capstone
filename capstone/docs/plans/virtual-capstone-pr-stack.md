# Virtual Capstone review and landing stack

The qualified application direction is supervised **virtual C mode**, with a
trusted Linux module/runtime, one address-space lifetime namespace and Linux
scheduling/VM. Direct capability-enabled Linux U mode remains a separate
experiment; its entry patches are not a dependency of the C-mode adapter.

## Comparison bases

The [audit manifest](virtual-capstone-pr-audit.json) lists every historical
feature commit and changed path, without author identities. Its application
baseline is `6fb2df614ca4`; the new pthread and memcached changes are separate
review commits on top of it.

| Repository/ref | Snapshot | Interpretation |
|---|---|---|
| llvm-capstone `dev` | `a8922b0979b9` | 39 historical virtual commits beyond the common base; dev also has 268 commits absent from this lane |
| llvm-capstone `main` | `88618641986d` | Older release base; land through dev before proposing a release PR |
| `virtual-addressing` | `6b5779a09072` | Other lane; compare semantic changes, not entire final trees |
| `delegation-threads` | `92d21b84324b` | Source of reused musl condition/attribute layouts and host signal mechanisms |
| `virtual-capstone-prototype` | `55708215b540` | Ancestor containing the direct-U prototype and guest tables |
| `virtual-capstone-runtime-parity` | `b7c0ad843f87` | Ancestor of the C-mode runtime/application work |
| `memcached-app` | `c839e7294e23` | Source of the complete physical server port, oracle and unchanged safety predictions |
| QEMU qualified feature | `6bf04634a2c2` | Existing qualified binary; memcached adds no QEMU source change |
| QEMU `c128-qemu-merge` | `b8f08e599330` | 26 feature-side commits and nine target-only commits, including merges |

Use three-dot feature diffs and explicit submodule diffs. A two-dot tree
replacement would remove newer dev work. In particular, preserve dev's RTL
`776d9d85900e` and Buildroot `8fd1ea1249a9` pointers rather than copying this
lane's older pins. The system pointer is unchanged. The spec is not a changed
superproject gitlink. Linux source/image qualification is based on
`a62e18f2b9aa`; this application step needs no additional Linux-core patch.

## Immediate review branches

| Head | Base | Review scope |
|---|---|---|
| `pr/virtual-linux-posix` | `virtual-capstone-libc-vm` at `6fb2df614ca4` | One commit: musl pthread bridge, independent transports/signals, native-VA futexes, stop-before-clear exit, ABI v3 and gates |
| `pr/virtual-memcached` | `pr/virtual-linux-posix` | One commit: complete server port, virtual recipes, native/worker/safety gates, results and this integration plan |

These are stacked review branches, not new versions of the already published
prototype/runtime history. Keep `virtual-capstone-memcached` as the complete
working lane. Do not force-push old branches or squash them in place.

Suggested first PR title: **Support musl pthreads in virtual Capstone applications**.
Its description should lead with the previous failure: blocking workers shared
one transport and musl clone could not create a virtual context. Explain that
Linux now schedules same-mm workers while musl owns TLS and synchronization.
Name the tested subset, ABI compatibility, shared-TLS transport check and the
signal/cancellation cases which remain unqualified. QEMU and the module ABI are
unchanged.

The pthread review also fixes per-invocation epoll conversion: private syscall
transports alone did not protect the old shared event array after copy-back.
Its controlled-interleaving mutation is a required regression check.
Likewise keep the blocked-receive mutation which detects shared host message
views/iovecs. Native-worker concurrency needs per-call descriptors on both
sides of the transport.

Suggested second PR title: **Run the complete memcached server on virtual Capstone**.
Describe the four-worker native protocol comparison, TERM/USR1 exit parity,
worker-marker control and ten fixtures against the physical Sublet predictions.
State that Slab-neighbour and Slab-reuse remain known protection gaps. Include
the imported physical baseline's provenance and preserve its comparison arms.

## Landing the historical stack into shared branches

Prepare fresh review branches; squash complete logical changes onto their
review bases. The original 39/26 commits remain provenance. Do not publish a
single PR containing the historical U-mode experiments, compiler change,
runtime, application migrations and unrelated dev reversions.

| PR | Target | Contents and dependency |
|---|---|---|
| Q1: protected memory/lifetime foundation | QEMU `c128-qemu-merge` | Physical-granule tags and kernel-store invalidation; bounded guest-node backend, fail-closed IDs, capability-fault delegation/tval and consuming protected-path transfers; legacy and model gates |
| Q2: supervised virtual C execution | Q1 | CSRUNV, user-PTE translation, per-execution PCC, service/page-fault/quantum continuations, trusted mint/retirement and stopped-namespace collection; virtual execution gates |
| L1: readable image-gp compiler profile | llvm-capstone `dev` | Two Capstone backend files, constant-pool test and deterministic drift-helper adjustment; keep capability-table refusal |
| R1: virtual Linux runtime and libc VM services | L1, pinned merged Q2 | Module ownership/pinning/context lifecycle; C-mode launcher/CRT; growing bounded/revoking heap, anonymous mmap/mprotect and explicit same-mm threads; include the pthread review commit |
| A1: Perl virtual profile | R1 | Fresh source build, preserved pointer-width patches, smoke and corpus qualification |
| A2: remaining application migrations | R1 | CPython, PostgreSQL single-user, FFmpeg and offline tshark; source recipes, nested allocator block loans and native/safety gates |
| A3: complete memcached virtual port | R1 | The second immediate review commit; preserve the physical server port and its predictions |

Q1 and Q2 are logical review slices, not an assertion that historical commits
can be cherry-picked independently without adaptation. Keep direct-U context
spills, opt-in kernel patches and abandoned selector experiments outside the
C-mode runtime's shipping contract. If retained in QEMU, keep their opt-in
qualification and describe them as experimental.

For commit triage, `4715ec129b60` and `923cc8c235ae` are delegated-buffer
foundation work for R1; `bd13d9348b75` through `da36ed0e1d8e`, plus
`320ba350f278`, supply its C-mode runtime pieces. `e2777455c4f1` is A1 and
`6fb2df614ca4` is A2. Earlier direct-U documentation, Linux-entry probes and
context-spill pins are provenance or the separate experimental lane. Compiler
edits embedded in the runtime commits must be extracted into L1 rather than
mixed into a runtime PR. Likewise, split QEMU's `dc8878185d9d` foundation
from `5adb91104352`, `49328837b0d8` and `8ad8d281583c` execution/collection
work, adapting their shared dependencies instead of blindly cherry-picking.

Before QEMU landing, reconcile the target-only physical-supervisor changes:
dead-seal status, C-mode confinement, sealed-return slot confinement and
overflow-safe sealed-return bounds. General `cap_in_bounds` already has the
subtraction check in the virtual lane; the newer sealed-return window is a
separate target-side change. Preserve both, plus the target's instrumentation.
The virtual instruction policy denies these domain transitions, but that does
not justify dropping fixes from the physical supervisor. Rerun physical,
protected-memory and virtual gates on the integrated binary, then update the
superproject pin. The current qualified binary is not claimed to be that merge.

## Processor changes, separately from libc

- Virtual C contexts interpret bounds/cursors as VAs, then walk the owning
  process's page tables with user permissions. Capability checks and PTE checks
  are both required; tags still belong to physical granules.
- `scapctl`, `srevroot` and `urevavail`, the guest-record backend and trusted
  CSMINT provide process-local lifetime authority. Missing, zero and out-of-range
  node IDs fail closed in the protected path.
- CSRUNV saves/restores complete capability contexts, binds roots/options and
  returns service, fault, quantum and collection events to the trusted adapter.
- Virtual fetch checks PCC at execution, including cached code after revocation.
  Privileged CSRs, domain transitions and debug minting are denied there.
- Protected LDC/STC consume linear sources only after successful preflight;
  capability faults have explicit tval and delegable causes.
- CSRETIRE revokes an arena ancestor. Collection stops the address space,
  clears stale tags in registered physical pages and saved contexts, pins dead
  PCC identities and only then recycles IDs. It is a software QEMU design.

Memcached's pthread bridge adds **runtime service numbers**, not CPU opcodes or
CSRs. It adds no RTL/FPGA implementation, kernel-core patch or firmware change.
The physical monitor is retained. Full POSIX, SMP, cross-process tagged sharing,
fork and tag-preserving migration remain outside these qualification results.
