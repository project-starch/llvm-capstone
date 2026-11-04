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

Qualification still requires real N/U/F/L composition, widened broker grants,
the requested resource negative controls in a guest, original application
oracles, attempts/event against plain Linux, quick and full seven-benchmark
census, all applicable queued 32-slot gates on both hosts, and independent
cross-model review. Reference counts are not instruction attempts or cycles.
