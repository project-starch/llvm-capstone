# Page-local capability global initialization

Status: qualification fixtures only. The compiler and runtime still initialize
capability globals eagerly. This document does not authorize enabling lazy
initialization or claim an instruction-count improvement.

The desired transition is per image data page: absent, initializing, ready.
Only ready pages may be observed by application code or written by a delegated
Linux operation. The initializer must run once, before the first operation on
the page, including a first scalar store that intentionally clears a tag.
Untouched pages must stay absent. Pinning a page is not sufficient to mark it
ready.

## Metadata and compiler contract

Use an allocated, read-only relocation section and a page directory. The
loader must validate the directory, record count, monotonic offsets, slot
alignment, segment containment, duplicate slots, target ranges and arithmetic
overflow before registering it. Immutable file-backed metadata must remain
immutable for the mapping's lifetime. Do not accept a mutable application
descriptor as authoritative bounds or code addresses.

Resolve and sort records only after final link layout. Weak undefined targets
must have an explicit null representation; adding a PC-relative relocation
whose link-time target is zero to a relocated PC does not produce null.
Distinguish the target symbol from its cursor addend. Preserve the existing
materialization's bounds, permissions and lifetime identity, including aliases,
block addresses, function pointers and one-past pointers. Representability must
be checked for the final bounds and cursor operation, not merely the storage
width of the addend. The directory itself must not acquire widened bounds by
shrinking an unaligned, unpadded per-module table.

The existing capability-global pass runs before GlobalMerge. Moving it after
GlobalMerge can change the globals that are merged and therefore their bounds.
Running the same pass twice and erasing a function by its spelling is also
unsafe. A lowering pass should consume explicitly identified synthesized stores
after the existing merge decisions, without resurveying arbitrary user globals
or deleting a user function named `__capstone_cap_init`. It must retain the
existing diagnostic for a capability in a TLS template; runtime initialization
of a thread-local pointer remains valid. Constant pools still use their normal
code-generation path and need separate regression coverage.

Unsupported initializer forms require an explicit policy. An eager fallback
that writes a page containing lazy slots must initialize those slots before
its first store, and must not allow a later lazy pass to overwrite program data.

## Execution constraints

S-mode cannot use LDC/STC to install the capabilities. CSMINT creates a new
lifetime rather than deriving from the image root, so replacing each store
with CSMINT does not preserve the existing identity. A C-mode executor is
needed under the present ISA.

Merely replacing the interrupted frame's PC with the address of a C-mode
helper is not a general callback mechanism. An untagged PCC slot on CSRUNV
resume retains the saved PCC authority. A first touch may occur under a
narrowed code capability whose bounds exclude the helper. It is insufficient
to preserve the integer fault PC: the original PCC bounds, rights, type and
lifetime must survive the complete helper transaction. The gp cursor also
cannot be assumed to point to a shared scratch block at an arbitrary fault.

The global vm_lock covers a CV_STEP call, but the quantum/deadline/signal paths
release it. An initializer can itself fault on metadata or be preempted between
stores. Ownership must therefore persist across ioctls. Siblings must not
observe partial initialization, reuse the save area, or initiate a second
initializer. Repeated CV_RESOLVE must be idempotent; a failed initialization
must not mark the page ready. Cancellation and thread exit must discharge that
ownership without stranding siblings.

Admission has several routes: resolve, pin-ahead, resident scanning, eager
registration and CV_TOUCH. CV_TOUCH promises that the launcher may immediately
write the admitted span; queuing initialization for the next CV_STEP can put
initializer stores *after* the launcher's write. A thread exit can remove the
thread before any next step. This needs a synchronous completion or a defined
defer/retry protocol before the host write, not just a pending bitmap.

These constraints require a specified executor protocol before enabling page
relocations. Possibilities include an isolated C-mode execution context with
explicit lifetime sharing, or a capability-preserving callback mechanism. An
extra context must account for supervisor capacity, including the single-slot
configuration; changing the required capacity is an ABI/platform decision.
Neither alternative is implemented here. The simple saved-PC proposal is not
an accepted design.

## Qualification fixture

Build `capstone/runtime/virtual/cap-init-page-contract.c` together with
`cap-init-page-probe.S` using the virtual SDK. Each mode is a fresh process:

* `cold`: form and print the holder interval without loading any slot.
* `one`: inspect one holder page.
* `all`: inspect every holder, including interior/one-past references, function
  pointers, an alias, a string literal, and a runtime-initialized TLS pointer.
* `overwrite`: first write a pointer slot through a scalar lvalue, then verify
  that the null cursor and cleared tag persist and its neighbor is initialized.
* `broad` / `narrow`: load a fresh anonymous page under image-wide or narrowed
  PCC. The ordinary module must resume both successfully. These are controls
  for a diagnostic redirect at `cap_init_page_probe_body` to
  `cap_init_page_probe_redirect`, outside the narrowed bounds.
* `weak`: build a separate image with `-DCAP_INIT_WEAK`; undefined weak object
  and function pointers must both be null with the null encoding. This image
  is separate because a broken eager weak initializer may fault before main.

The comparisons read the capability tag and both stored encoding words against
an ordinary materialization, including the lifetime bits. Literal dereference
is also checked. The narrowed-PCC probe creates a capability return link with
CJALR before entering the restricted range; the gp-free caller's integer return
address alone would not restore the original PCC. Run with the default
compressed-bounds profile and the exact-bounds profile. With `diag_admit=1`,
count admissions only in the printed
holder interval; process-wide page counts also include libc and stack pages.
The eager control is expected to admit every holder page even in `cold` mode.
A lazy candidate must admit none in `cold`, just the requested page in `one`
with pin-ahead disabled, and all in `all`.

Still needed before acceptance: deterministic interruption between initializer
stores, concurrent first touches, delegated writes, prepopulated image tail,
read-only table tampering, failure cleanup, constant-pool and TLS diagnostic lit
tests, the complete runtime gates, and matched quick/full census measurements.
