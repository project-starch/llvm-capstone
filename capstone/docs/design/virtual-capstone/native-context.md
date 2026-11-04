# Native context/event interface v1

Status: executable proposal, not a frozen integration ABI. The header is
`runtime/include/capstone/native-context.h`; the linkable implementation and
test driver are in `runtime/virtual/tests/native-context`. The implementation
is a single-threaded software double. It neither implements CSRUNV slot
eviction nor changes Linux signal handling. Native exec remains default off.

## Backends and trust boundary

One context/event contract serves two distinct entry backends:

* `CN_BACKEND_IOCTL`: a scalar trampoline enters the kernel dispatcher, which
  owns the protected context. Delegated calls still require validated private
  bounce memory and the resource policy described in `native-exec.md`.
* `CN_BACKEND_ARCH_RETURN`: a future kernel return hook enters a domain and
  dispatches escape events. The double returns `-EOPNOTSUPP` for this backend.
  No direct-trap behavior or performance saving is implied.

This is a trusted C interface, **not a userspace ioctl wire format**. All actor
identities must come from the current kernel task. Frame creation and handler
installation accept only protected backend state. Ordinary userspace cannot
import serialized register bytes, supply the actor identity, modify a saved
frame, or replace a namespace root. Snapshots exist for trusted audits only.
The scalar trampoline holds no domain grants. Implementing an authenticated
hardware export/import path is still required before connecting this interface
to the module. Ordinary `pt_regs` does not contain the required state.

The C structures use explicit 64-bit and 32-bit fields. `version` must equal
`CN_ABI_VERSION`, `bytes` must equal the structure size, unknown feature bits
and nonzero reserved fields are refused. There is no packed, endian-independent
wire encoding. A representation change requires a new version; an additional
architectural state component must not be silently discarded.

## Ownership and complete state

A context belongs to one thread. Its address space owns the namespace;
multiple thread contexts may refer to that same namespace. The owner tuple is
thread ID, address-space ID, namespace ID and namespace epoch. Each context
also has a nonreused engine-local ID and a monotonically increasing epoch.
Namespace and translation epochs must be validated by the owning backend
before entry, authorization, cloning and saved-signal restoration.

`cn_frame` carries all 32 integer/capability register records, four capability
CSRs in PCC/CTVEC/CEPC/CSCRATCH order, 32 floating-point payloads, `fcsr`, PC,
status, SATP, translation root and translation/namespace identities. Register
records preserve both 64-bit payload words, tag versus opaque raw-tag cargo,
and exact-bounds side state. Scalar writes clear the entire destination record
before installing the scalar value. x0 must be completely zero. FP state is
bitwise, including NaN payloads and negative zero. Disabled FP state must be
zero. Vector state and unknown extensions are refused, not approximated.

The double never decodes Capstone capability payloads. Its synthetic test
namespace uses payload indices solely to exercise ownership and liveness.
Production tag authentication and instruction legality belong to the trusted
ISA backend; passing a double test does not establish either.

The reference engine has exactly one active slot. Switching moves every frame
field into protected storage, clears the outgoing slot, moves the incoming
frame into the slot, and clears its inactive copy. A production backend must
also save hidden ISA state, invalidate cached authority and translation state,
and complete any required fences before exposing the slot to another task.
The current CSRUNV frame does not provide this complete protected export.

## Events, replies and capability references

Events distinguish quantum expiry, service requests, translation faults,
capability faults, signal delivery, exit and unsupported-backend refusal.
Capability faults further distinguish tag, type, rights, bounds, lifetime and
namespace failures. An exception cause alone never authorizes ordinary paging.
`fault_pc` identifies the original instruction; `restart_pc` is the backend's
current PC, never a userspace-selected jump target. Event sequence numbers
increase independently of restored signal frames. Each event carries the
post-step context epoch.

`CN_RESUME_RETRY` retains the backend PC. `CN_RESUME_REPLY` is accepted only for
a pending service and invokes `complete_service` exactly once. That callback
must apply the scalar result with tag clearing and update PC and PCC together
according to the instruction ABI. The synthetic test uses a four-byte service
instruction; production instruction decoding is not supplied by this double.
Missing completion support fails closed. Old epochs and duplicate replies
fail. Invalid backend events or frame state terminate the context. Exit,
capability-fault and refusal events are terminal in v1; recoverable capability
fault handlers are unsupported.

`cn_capref` identifies a protected general-register slot plus context ID and
epoch. It contains no scalar address from which authority can be reconstructed.
`cn_authorize` checks owner, epoch, slot, tag, range overflow and the namespace
before calling the checked-access lane's authorization callback. Zero length
does not bypass authentication. Missing callbacks fail closed. This operation
only authorizes: it supplies no copy, pin, retained-pointer or sleeping-retry
guarantee. A copy consumer must maintain exclusion and revalidation across
every chunk and fault retry; the authorization result cannot be cached as a
permission to copy later.

## Protected signals, clone and teardown

The double supports eight nested protected signal frames. Push saves the full
frame and pending event; the handler must bind to exactly the same translation
and namespace. Return requires the current epoch, context ID, depth and a
fresh engine-issued nonce for the top frame. Old, foreign and out-of-order
tokens fail. Return restores the complete protected frame/event and erases
the consumed snapshot, while preserving monotonic epoch/sequence counters.
The kernel must keep the resume state protected; the token alone is not a
cryptographic authentication mechanism. Linux signal masks, altstacks,
SA_RESTART, syscall cancellation and ptrace are outside this double.

`cn_clone` requires distinct child thread, address space and namespace IDs,
with no live context already using the child's identities in this engine.
The namespace callback must take a consistent snapshot of generations, free
lists and liveness, returning a distinct root. A failed callback must roll
back its own preparation. If context creation fails after callback success,
`abort_clone` rolls it back. The parent input and child output translation
records do not alias. Clone during a signal handler or pending service fails
with `-EOPNOTSUPP`. This is not a Linux fork implementation. Namespace release
and memory COW remain the lifecycle and memory backends' responsibility.

Destroy clears resident/saved state, pending events and nested signal frames.
Destroyed objects remain tombstones until engine destruction so a stale
context pointer cannot become another live context. No calls may use pointers
after `cn_engine_free`. The double requires serialized calls; it does not
provide locks, scheduler exclusion, DMA draining or secure physical erasure.

## Running and consuming the double

On a build host, from a lane-owned source export:

```sh
CAPSTONE_TOOLCHAIN_CHECKED=1 CAPSTONE_TMP_ROOT="$HOME/lanes/progN/tmp" \
  bash capstone/runtime/virtual/tests/native-context/run.sh "$HOME/lanes/progN/context-results"
```

The output must be fresh. The script builds a sanitizer-checked executable,
four intentionally broken variants, and `libnative-context-reference.a`.
Other C consumers include `<capstone/native-context.h>` and link that archive
(or compile `reference.c`). Implement `validate_namespace` and `step`; supply
the other callbacks only when their contracts are supported. Errors use
negative errno values. Mutation macros are test-only and must never enter a
production backend.

The test alternates two namespaces, interrupts with nested signal frames,
clones a namespace and revokes only the child's synthetic grant, then verifies
the complete frame after repeated switches. It checks stale references,
scalar tag clearing, replay rejection, service completion, terminal events
and rollback. Skipped namespace restoration, dropped tags, scalar tag
retention and replay acceptance each make the unmodified checker fail.
Counters count software entries, frame moves and bytes moved, not QEMU
instruction attempts or hardware cycles.

`run-peers.sh` consumes read-only snapshots of the fault/mapping and page
lifecycle doubles. It also links the fault lane's context consumer against
this implementation, including its cause-only classification mutant. Exact
input hashes are retained. These tests do not constitute a composed kernel
implementation or independent cross-model review.

Required integration evidence remains: two real tasks through one physical
slot with complete hidden-state restoration, real nested signals and fork,
child-only revocation, deliberate root/tag restoration failures, matched
attempt/cycle costs, hosted/runtime parity, full census 14/14, all applicable
gates on both hosts, and independent review. No production or landing claim
may be derived from the reference tests alone.

The [checked-copy prototype](checked-copy.md) consumes this context double
and provides a shared chunk/retry engine plus a default-off kernel adapter.
Its CSCHECK/CPTAG encoding conflict and native syscall integration remain
unresolved; the hosted controls do not qualify guest kernel execution.
