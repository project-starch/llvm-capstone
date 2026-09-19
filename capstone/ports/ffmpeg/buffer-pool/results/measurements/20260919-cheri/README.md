# CHERI spatial arena replay comparison

Nine CheriBSD spatial replays use the same three FFmpeg 9.0.1 recordings as
the [Capstone spatial/Sublet campaign](../20260919-replay/README.md).
Three repetitions per recording match the complete native event sequences
and produce identical output within each arm. These are extracted allocator
replays with different lifetime guarantees, not protected video decoding.

This export contains only Capstone spatial, Capstone Sublet and CHERI spatial
results. It revalidates the original accepted `campaign-spatial-5` and
`campaign-2` captures; rebuilding this export and its plots is not a new
measurement. Recorded binary and platform hashes identify the original runs.

## Memory behavior

| Recording | Capstone payload carved, both modes | CHERI payload carved | Extra carved | Metadata carved, all arms |
|---|---:|---:|---:|---:|
| short, 320x180, 1s | 372,352 B | 372,544 B | 192 B | 12,288 B |
| turnover, 320x180, 2s | 371,904 B | 372,096 B | 192 B | 11,648 B |
| larger, 640x360, 2s | 1,242,240 B | 1,242,816 B | 576 B | 12,288 B |

![Requested pool payload](payload-comparison.png)

![Additional payload arena carving](arena-padding.png)

The extra carving includes alignment and padding for CHERI compressed bounds.
The largest expansion of an issued pointer is respectively 49, 49 and 177
bytes. Bounds cover requested payload and stay inside its backing block.
Accumulated bounds slack counts repeated issues, including callbacks; it is
not simultaneous waste. These figures do not measure total protection cost.

## Protection and controls

Every CHERI replay confirms 16-byte pointers and disabled libc revocation.
Five separate companion processes check valid references/callbacks, stale
buffer reuse, stale RefStruct return and two bounds violations. The first
three complete; the bounds probes reach their setup markers and then SIGPROT
(exit 162). Pool returns do not invalidate stale pointers in this spatial
backend. The earlier Sublet controls reject stale pool accesses.

## Accounting and provenance

`measurements.json` contains recording, binary and platform hashes and all
accepted points. `elf-storage.json` retains the corresponding Capstone and
CHERI executable section inventories. Static bookkeeping, dynamic libc and
loader storage, stacks, OS memory, capability tags and Capstone nodes need
separate accounting. Arena carving is not a complete memory ledger.

Both replay harnesses reserve 64 MiB payload, 16 MiB metadata and 128 MiB each
for input/output. The CHERI guest has 2 GiB RAM and one CPU; the Capstone guest
has 8 GiB and one CPU. This is not matched OS memory pressure. Emulator elapsed
time does not establish hardware performance.

`attempts.json` retains the earlier spatial bring-up and failed attempts.
Raw captures stay outside the repository. Recreate this export and the plots
using the commands in the [collector guide](../../../host/cheribsd/README.md).
