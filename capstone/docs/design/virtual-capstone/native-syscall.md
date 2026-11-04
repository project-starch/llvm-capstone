# Native syscall interface proposal

Status: default-off executable interface work. The production module and
default SDK do not select this ABI. It is not qualified for integration.

`runtime/include/capstone/native-syscall.h` consumes the protected context
interface and specifies a checked-copy callback with the frozen I3 signature.
The descriptor inventory preserves all 96 ordinary calls from the earlier
AG2 bridge. Read/write, three private wire operations and the five native
services are additional rows. Inventory membership does not grant permission:
the later resource broker refused many of the original calls. Unknown calls
and missing backends return ENOSYS; no ambient Linux forwarding is allowed.

The experimental register convention uses a7 = `NS_SERVICE_NATIVE`, a6 =
operation number and a0..a5 = arguments with their original tags. The kernel
captures only an authenticated protected frame, deriving `cn_capref` values
for pointer positions from that frame. Scalar cursors and LCC fields never
become authority. This service selector is a proposal for integration review,
not an allocation in the current image ABI. A legacy module rejects it.

All buffer traffic must use the copy callback. Its provider must authenticate
tag, type, bounds, rights, namespace and lifetime for each chunk, exclude
revocation, resolve faults without a dummy write, recheck on retry, drain
writes and clear tags on scalar output. N's `cn_authorize` alone is not a
safe copy implementation. Missing U support fails closed.

Iovec import is a distinct unresolved production interface: an ordinary byte
copy cannot authenticate nested capabilities. `snapshot_iov` must capture
tagged elements into protected references atomically with respect to changes
to their authority; lengths and references then stay fixed until
`release_iov`. The reference fixtures use protected frame slots. Serialized
capref bytes supplied by a domain are never accepted as nested authority.
The production backend must support protected spill storage for more than
the available register slots. Other hidden-pointer structures remain refused.

`acquire` is mandatory even for scalar calls. A production provider must apply
the resource policy and specified security hooks, pin actual file identities
through blocking I/O and reject descriptor replacement. An integer fd bitmap
is insufficient once shared descriptor tables or duplication are admitted.
The existing broker's component-wise O_PATH|O_NOFOLLOW resolution,
inode_permission-before-dentry_open checks and exec sharing refusal remain
required. Broadening root grants must retain those checks. Inherited Linux
filters cover actual scalar syscalls/ioctls; arbitrary per-native-call filter
compatibility is not provided by an in-kernel callback.

MAP requires a capability return installed in the protected frame by the
grant backend; returning an address as an integer is forbidden. UNMAP and
PROTECT require whole-grant identities and retire/attenuate ordering. NODES
and FUTEX require namespace/lifecycle-aware providers. These rows cannot use
the generic scalar completion and remain refused by the reference dispatcher.

## Executable reference

`runtime/virtual/tests/native-syscall/reference.c` links against N's actual
context double and the existing RV64 wire-shape/result-length code. It captures
one request per epoch, snapshots every input before issuing any operation,
keeps strings in distinct private allocations, checks multiplication and total
size limits, and publishes only the result's permitted output prefix. Explicit
NULL with a zero dynamic length is accepted; a nonzero scalar masquerading as
a zero-length buffer is refused. Tagged zero-length buffers are authenticated.
The current reference bounds total private buffers at 1 MiB and vectors at 64
elements; larger requests fail closed rather than silently truncate.

The flat structures use the shared pointer-free Linux RV64 layouts. The
backend must fill those layouts, not its host's native structures. Timespec
conversion is exercised as two explicit 64-bit words. This does not implement
every musl structure/command adaptation. Iovecs use the separate protected
import described above. Nested socket messages, word-length pointer graphs,
retained pointers, caught-signal actions and unknown ioctl commands have no
scalar fallback. Pipe2 is refused because fd-pair publication needs explicit
rollback if checked copy-out fails. Existing production pipe support is not
changed by this reference limitation.

Only byte-stream reads, positioned reads, vector reads and getrandom return
the successfully published prefix after a copy fault. Other output failures
return the error; the result retains bytes_out and effects for the lifecycle
consumer. An EINTR restart request is accepted only when the backend certifies
zero effects; no automatic retry occurs inside the dispatcher. A restarted
call needs a fresh protected epoch. Side effects from a failed output cannot
be undone or hidden by reporting a restart.

Run `tests/native-syscall/run.sh OUT` on a build host with the shared test
environment configured. It builds a linkable reference archive, runs the
sanitizers with recovery disabled, and requires specific oracle failures from
nine broken variants. Consumers link the archive with `common/delegate.c`
and `tests/native-context/reference.c`, supplying the callbacks in the header.
One session must be kept for the entire protected context lifetime; recreating
it to bypass replay detection violates the trusted-caller contract.

The test copy provider models revocation between 16-byte chunks, namespace
ownership and scalar tag clearing. Its objects and capability payloads are
synthetic. The resource fixture models safe/private file identities, held
references, explicit roots, permissions and policy refusals. It does **not**
execute Linux VFS, LSM, seccomp, DMA, page faults or an ISA instruction. Its
proc-memory test uses a simulated private-memory object. Passing it does not
replace the real native-dom proc-memory, inherited-fd, replacement and alias
guest probes. The production broker remains unchanged and unqualified here.

## Experimental musl entry

`CAPSTONE_NATIVE_SYSCALL_ABI` selects `__capstone_native_syscall` in the musl
architecture shim, only together with `CAPSTONE_RUNTIME_VIRTUAL`. Explicitly
link `ports/musl-capstone/runtime/native-syscall.S` to use it; the normal SDK
build does not include that file or define the selection macro. The stub moves
all six arguments with `movc`, then issues the experimental selector. No
capability is converted into a scalar request buffer. The check-sdk script
cross-compiles both shim choices, checks the unresolved entry symbols and
disassembles the six tagged moves. It also rejects an unselected stub build.

This is a compile-checked entry primitive, not a complete native libc. The
existing delegate.c special handling for fcntl/ioctl, memory returns, signals,
thread startup and cancellation still needs explicit native adaptation. There
is no feature negotiation or image ABI selection, and the production module
does not serve this selector. Do not enable the macro for application builds
until those integration requirements are qualified.

Qualification still requires real N/U/F/L composition, widened broker grants,
the requested resource negative controls in a guest, original application
oracles, attempts/event against plain Linux, quick and full seven-benchmark
census, all applicable queued 32-slot gates on both hosts, and independent
cross-model review. Reference counts are not instruction attempts or cycles.
