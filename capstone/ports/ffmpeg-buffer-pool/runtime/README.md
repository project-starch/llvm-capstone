# Allocator runtime: native, Capstone and Sublet

These sources are compiled into the replay and security executables. They
share the same allocator layout across native and Capstone comparison arms.

| File | Role |
|---|---|
| `metadata-memory.c` | Bounded metadata heap and the required `av_malloc`/`av_free` compatibility functions; ordinary C in both targets |
| `pool-memory.c` | Payload allocations, out-of-band RefStruct metadata, release-authority checks and protection modes |
| `prepare.sh` | Build helper on the development machine: source verification, compiler flags and freestanding support objects |

## Where the protection is implemented

`pool-memory.c` selects the execution target with `#ifdef FFPOOL_DOMAIN`.
The native branch uses ordinary pointers for functional comparison. The
Capstone branch uses capability operations and the existing shared
[`sublet.h`](../../sqlite/sublet/sublet.h).

Within the Capstone build, `ff2_set_mode()` selects the protection:

- Mode `0`: `issue()` restricts payload bounds; returning a lease does not revoke it.
- Mode `1`: `ff2_payload_free()` revokes the backing allocation using the outer handle.
- Mode `2`: `issue()` additionally obtains a lease with `sublet_take()` and
  `ff2_payload_return()` revokes it with `sublet_give()` on return to the pool.

The shared Sublet helper header is used for capability operations in the
Capstone build generally; **pool-lease revocation is enabled only in mode 2**.
Selecting mode 2 in a native build cannot enforce this protection.
