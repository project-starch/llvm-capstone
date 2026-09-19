# A1: parent revocation of scattered child aliases

All **44 accepted executions** pass in the pinned Capstone QEMU: 20 protected
stale accesses fault, all 20 matched no-revoke accesses succeed, and four
valid-control executions complete. Each cell has one accepted execution,
not a statistical repetition series.

| Old alias location | Immediately after revoke | After same-address reuse |
|---|---|---|
| Global | Read and write blocked | Read and write blocked |
| Heap object | Read and write blocked | Read and write blocked |
| Linked-list tail | Read and write blocked | Read and write blocked |
| Independent sibling pool | Read and write blocked | Read and write blocked |
| Register, no spill across revoke | Read and write blocked | Read and write blocked |

Every run first checks that all five copies are usable. The parent then revokes
its ancestor handle without requiring the child to return or clear any copy.
The sibling remains readable and writable. New authority can read and write
at the child's old address. Dedicated controls complete without a stale access.

This demonstrates the tested **ancestor invalidation and reuse semantics**.
It does not demonstrate a performance or total-memory advantage, a full
allocator port, protection of a hostile manager in another domain, or failure
of CHERI, PoisonCap or PICASSO. The no-revoke arm is a control in the same
Capstone binary; it is not a simulation of those systems.

## Fixture and evidence

[Protocol and reproduction](../../../security-tests/capstone/README.md).
A 4 KiB parent delegates a 64-byte child. A separate 1 KiB sibling region and
the metadata heap hold aliases outside the revoked subtree. Memory-holder
reuse cases initialize the parent and issue a new 64-byte child grant. Register
reuse cases access recovered parent authority at the exact old child address
inside a leaf assembly routine. They keep the old child alias in `a0`
throughout, without calls or spills.

Each expected fault requires the setup marker and exact instruction PC.
Register cases also require a marker after revocation/reuse and live sibling
checks. [The built register span](register-span.txt) contains one REVOKE,
no calls, no stack accesses and no stores of the retained alias. Its INIT
loop can zero the recovered parent; it does not search alias-holder storage.

[measurements.json](measurements.json) records every accepted case, raw-log
and binary/input hashes, expected/observed PCs, and failed matrix attempts.
[provenance.json](provenance.json) pins binaries, final component sources,
runtime headers and build settings. Raw captures remain outside Git under
`/tmp/capstone/alias-scatter-work/`; their hashes attest identity but the
captures themselves are needed to independently recheck the fault oracles.
The exporter refuses incomplete matrices, duplicate accepted cases, mixed
domain binaries and changed evidence.

Six existing pool-lifetime regressions also pass (cases 0, 3, 5 in modes 0 and
2), as do the three native CTest checks and a native pool control.
The 14-execution pilot is kept separately and is not counted in the matrix.
Failed boot attempts are excluded from semantic conclusions and retained in
the matrix export and [attempts.json](attempts.json).
