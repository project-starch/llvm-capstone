# Image initialization interface double

This is a draft interface for the image lane's consumers. It does not install
a compiler pass, change ELF loading, redirect a faulting PCC, or enable page
initialization in the runtime. Integration and format freeze remain pending.

`reference.py` exports `Image(version, destination, symbols, records, ...)`.
Version 1 uses 4 KiB pages and 16-byte capability destinations. The typed
records are a host representation of a proposed format, not a binary wire
encoding or an implementation of the 128-bit capability codec:

| Record | Fields |
| --- | --- |
| Grant | namespace, nonreused identity, generation, base, size, rights |
| Symbol | source grant, offset, exact bounded size, weak binding |
| Relocation | destination byte offset, symbol index, signed 64-bit addend |

The loader must attest that metadata is not writable by the program. The
double rejects writable metadata and copies records into its private plan.
Records are strictly increasing by destination, hence grouped by page;
duplicates, misalignment, out-of-grant spans and unknown versions fail closed.
Undefined weak symbols produce NULL before any rebasing or addend arithmetic.
Undefined strong and unsized symbols are unsupported. Bounds must be exactly
representable, and addends must retain the tag under the R-60 cursor window.
This draft supports only low Sv39 userspace addresses: each complete source
and destination grant must lie below `2^38`, not just the relocation cursor.

`touch(namespace, page, event, cause, classify=...)` accepts a validated data
page fault. It uses the fault backend's classifier: tag, lifetime, bounds and
operand failures are never ordinary page faults. The protected `guard` callback
must exclude grant retirement and reuse through the entire transaction,
including sleeping admission. The protected `admit` callback checks the exact
namespace, identity, generation, span and permissions. Both callbacks belong to
the loader/backend; a task cannot supply them. Every relocation on the page is
validated before any becomes visible, and repeated touches are idempotent.
An initialized page still requires live destination authority.

The returned `Capability` values describe authority for assertions; they are
not serialized capability bits. Python privacy and immutable tuples model a
protected implementation; they do not enforce isolation from arbitrary host
Python code. No claim is made about hardware tags, scheduling, signals, fork,
page migration or execution under a narrowed PCC. A production implementation
needs a protected executor and lifecycle integration before it can use this
format. Unsupported operations must remain refused.

Run `python3 test-reference.py`. `--fault-peer PATH` consumes an executable
adapter to the fault lane's `cv_fault_classify(event, cause)`, whose numeric
results are denied/read/write/exec = 0/1/2/3. The qualification run also executes
that lane's unchanged grant double and all its broken variants. Source hashes
identify the consumed snapshot; the peer implementation is not vendored here.

Each `--mutant` mode deliberately removes one protection. The ordinary test
must pass, and writable-table, weak-rebase, cursor-window and destination-span
and unlocked-publication and address-policy must each exit nonzero. These controls qualify this reference contract only;
they do not qualify a compiler or runtime implementation.
