# ffmpeg/carved-repros on the virtual profile, 2026-11-11

Both virtual arms, re-measured on the processor model with the one-operand `CREVOKE` and `CRENEW`.
The corpus's carve driver (`shared/driver.c`, `-DFFC_SUBLET_CARVE`) changed with the instruction:
it ends a region with `capstone_cap_release(region)`, the one-operand `CREVOKE`, where it named the
block's lifetime and the region before. It still derives the block's lifetime and each region with
`CDERIVE`.

| arm | configuration | cases | tally | against `../2026-10-10-virtual` |
|---|---|---|---|---|
| `virtual-nested-pools` | `virtual-ffmpeg-carve` (`--cc-arg=-DFFC_SUBLET_CARVE`) | 13 | CAUGHT 13 | every verdict equal |
| `virtual-malloc` | `virtual-mallocng` | 13 | MISSED 13 | every verdict equal |

Controls, each in the arm's own boot: `bounds-malloc` and `uaf-malloc` fault on both arms. On
`virtual-nested-pools`, `live-carve` completes and `uaf-carve` faults at the labelled probe, as
`tools/arms.json` declares: a region that the driver ended and carved again stays unreachable
through its old reference while its sibling stays live.

Every catch is cause 28 at the case's labelled probe, the same probe function as in the earlier
bundle: `ffc_write_probe_u8` (00, 06, 07, 09, 10), `ffc_write_probe_u32` (01, 04, 05, 08),
`ffc_read_probe_u32` (02, 03, 11, 12). Every fixed run completes under the same protection.

Not in `inputs.json`: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`; module loaded without parameters (`unpinned=0`, `supervisor_slots=1`). Exact bounds is the profile of the 2026-10-10 bundles. The SDK is a
corpus SDK (`-fno-builtin`).
