# Four CheriBSD allocator ports: verification

The shared build and run interface produces usable purecap libraries and
direct-link examples for FFmpeg, PostgreSQL, CPython and Whisper/ggml.
[Usage and protection scopes](../../../host/cheribsd/README.md).

The main suite passes **13 guest processes**: ABI/runtime and bounds controls,
seven direct allocator examples (four PostgreSQL managers), and four replays.
A second suite passes **six processes**: the two controls and four external
client programs supplied through `--client`. Each external client is a copy
of its component's standalone example, built as the independent
`allocator-client` target. This validates the supported link interface.

| Replay | Input scope | Verified result |
|---|---|---|
| FFmpeg | Native decoder recording | 2,379 events; exact observed event sequence matches native replay |
| CPython | Native interpreter allocation recording | 123,622 events; logical counts and payload checksum match |
| Whisper/ggml | Native context-allocation recording | 60,179 events; logical counts and payload checksum match |
| PostgreSQL | Synthetic four-manager fixture | 5,176 records; requested counts match, 1,050 payload checks |

ABI-dependent resource values are retained, not required to equal x86:
CPython's metadata watermark is 4,865,664 bytes versus 4,602,752 natively.
ggml's object header is 48 bytes versus 32; its recorded peak-used value is
240,112 bytes versus 236,976. These are component/harness observations, not
total-process overhead or causal performance estimates.

All four native builds and their CTest suites pass (3 FFmpeg, 14 PostgreSQL,
3 CPython and 3 ggml CTest entries), alongside the shared support suite's
22 unit tests. All four Capstone domain configurations still build.
The CheriBSD preset and shell entry points were checked for every component.
This change does not claim new Capstone QEMU runtime measurements.

`summary.json` contains the accepted guest outcomes and platform/binary
fingerprints. `replay-validation.json` records the cross-ABI oracles.
`provenance.json` pins build files and relevant final source inputs.
Raw logs, binary replay reports and ephemeral SSH keys remain under
`/tmp/capstone/cheribsd-ports-work/` and are not committed.

The earlier seven-example pilot is excluded from the accepted totals. The
first full suite stopped at the bounds control: the child produced the
expected ready marker but SSH reported 255 when translating an unsupported
SIGPROT exit-signal name. Keeping a guest shell waiting for the child yields
the numeric exit status 162. The replacement full suite passes this control.
This was a collector transport failure, not accepted protection evidence.

CheriBSD's libc revocation is explicitly disabled and checked in these runs.
Only FFmpeg's default adapter adds per-payload bounds here; the other ports
preserve capability-compatible allocator/backing authority. Inner free/reset
operations do not automatically gain Sublet temporal semantics. These are
functional setup and linking results, not a ranking of defenses.
