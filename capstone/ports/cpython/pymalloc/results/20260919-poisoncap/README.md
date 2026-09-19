# PoisonCap pymalloc pilot — 2026-09-19

The extracted CPython 3.13.7 allocator passes the complete 33-process PoisonCap
QEMU suite, including the same 115-event native recording in spatial and
protected modes. Both match native logical counts, completion and checksum.
This is functional evidence for the trusted allocator adapter, not protection
of a full Python interpreter or an untrusted nested manager.

## What ran

| Group | Processes | Required outcome |
| --- | ---: | --- |
| Purecap ABI and bounds | 2 | 16-byte pointers; deliberate bounds access faults |
| Published PoisonCap instructions | 5 | Live/reused authority works; poisoned and stale accesses fault |
| Linked allocator example | 1 | Allocation, calloc and pointer-bearing moved realloc succeed |
| Additional API controls | 5 | Sizes, pointer-bearing realloc, snapshot failure, unwritten reuse and arena turnover succeed |
| Nine lifetime cases, two modes | 18 | Expected success, exact rejection or SIGPROT in each arm |
| Complete native recording, two modes | 2 | All events and native logical oracle match |

The nine paired cases are live/sibling preservation, stale read after free,
stale write after reuse, stale free after reuse, stale in-place-realloc alias,
bounds, raw fallback, pool reclassification, and repeated address reuse.
Protected stale free must exit 1 with both its setup marker and rejection code
719. Faulting cases require their setup marker and SIGPROT; arbitrary failure
is not accepted. The repeated-address case uses eight iterations, not 2,000.
Arena turnover separately allocates and frees 2,300 objects of 512 bytes and
checks that more than one arena was allocated and an arena was released.

## Replay measurements

The existing standard-library driver captured one round with one JSON item,
including JSON/regex processing, bytearray resizing and collection. The trace
has 55 allocations, 55 frees, four reallocations and one END: 115 events and
zero recorded live allocations at END. A second capture produced identical
bytes. This is a complete small recording, not a prefix of the existing
123,622-event recording. The latter has not been validated on this adapter.
Replay writes synthetic payloads and checks their preservation; it does not
recreate Python object graphs. The report checksum fingerprints the event
sequence; payload correctness is enforced separately by the guest byte checks.

| Metric, same recording | Spatial mode 0 | Protected mode 1 |
| --- | ---: | ---: |
| Events completed | 115 | 115 |
| Explicit sweeps | 0 | 63 |
| Bytes poisoned | 0 | 80,336 |
| Bytes with poison access state cleared | 0 | 80,336 |
| Bytes overwritten to erase retired poison | 0 | 80,336 |
| Additional snapshot bytes copied | 0 | 0 |
| Private metadata high-water, bytes | 5,275,200 | 5,275,200 |
| Arenas allocated / released | 1 / 0 | 1 / 0 |

The final values are in [measurements.json](measurements.json). These are
cumulative operation byte counts, not distinct resident bytes. Protection
work is accounted for separately from normal allocation traffic. In particular,
`copied_bytes` covers only snapshot save/restore, not ordinary moved-realloc
copies, application writes, calloc or kernel scans. The small recording does
not exercise in-place snapshot copying; the dedicated realloc control does.
Payload storage is a fixed 64 MiB reservation. `metadata` is private-heap
high-water, including replay scratch and cached buffers, not live bytes,
physical protection metadata, total process memory or RSS. Both PoisonCap
modes retain the same authority-record layout. Native/CHERI metadata differences
also include ABI and backend choices and are not a protection-overhead ratio.

## A failure found and fixed

The first complete attempt passed 31 of 32 processes, then rejected a valid
zero-size allocation in the protected replay at event index 25. Instrumentation
confirmed that an unrelated raw free's sweep had removed the live pointer's
tag. The published QEMU's `cclearpoison` resets access state but does not erase
the stored poison capability; the kernel's later sweep still recognizes its
payload. Reusing such storage without a client store exposed the leftover
poison. The adapter now zeros the cleared payload before issuing new authority.
Those additional writes are counted as `zeroed_bytes`.

The targeted `unwritten-reuse` control preserves a fresh, unwritten zero-size
or raw allocation across an unrelated sweep. It and the previously failing
complete recording pass after the fix; the full final suite then passes with
that regression included. [attempts.json](attempts.json) retains unsuccessful
and selected bring-up attempts with binary hashes. Their passes are not added
to the final 33. Raw guest logs and diagnostic source remain outside Git.

## Configuration and reproducibility

Follow the [build, link and run guide](../../host/cheribsd/poisoncap/README.md).
The suite uses the reconstructed published PoisonCap LLVM/QEMU/CheriBSD
platform with no new platform changes. Both the automatic guest default and
per-process libc revocation are explicitly off; the adapter's explicit kernel
sweeps remain on. This reuses the FFmpeg pilot's documented workaround for a
kernel VM-locking failure. It is not whole-process temporal protection.

[provenance.json](provenance.json) records source and compiler hashes, unchanged
upstream patches, platform pins, workload parameters and build configuration.
[cases.json](cases.json) binds every verdict to binary, platform, input and
output hashes. [replay-validation.json](replay-validation.json) records the
native oracle comparison. Run `sha256sum -c SHA256SUMS` here to verify this
export. Regeneration requires external platform builds and the native CPython
recorder described in the guide; no benchmark suite or platform is vendored.

[regressions.json](regressions.json) records five passing native CTests, all
24 shared Python tests, the standard CheriBSD example plus ABI/bounds controls,
and four Capstone QEMU live/stale controls across its two modes. All four
backend configurations build. These are regression checks, not a complete
repeat of the existing CPython defect corpus or every Capstone lifetime case.
The protected Capstone stale-read probe has capability-fault cause 24 exactly
at its announced load PC. Its guest stops before the normal shell completion
marker (`runner_exit=1`); the existing lifetime oracle accepts the exact fault,
not normal completion. The other three controls complete normally.

The current adapter synchronously sweeps on every protected release and on
pool/arena transitions. Its sweep count and storage choices are not an
architectural minimum. Larger recordings, quarantine/batching policies and
complete matched memory accounting remain separate experiments. QEMU elapsed
time is not reported as hardware performance.
