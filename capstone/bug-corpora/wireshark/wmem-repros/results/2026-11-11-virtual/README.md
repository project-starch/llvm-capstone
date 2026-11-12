# wireshark/wmem-repros on the virtual profile, 2026-11-11

Both virtual arms, measured again after patch 0001 and the processor model changed their lifetime
instructions: `CREVOKE` takes the object alone, and the reset that keeps a block renews its
generation with `CRENEW`. The block header's `life` field is gone. The successor of this record is
[`../2026-11-12-virtual-release/`](../2026-11-12-virtual-release/README.md), which reads the Release
images after a repair of the symbol lookup.

| arm | configuration | build | controls (uaf-chunk, uaf-jumbo, bounds-chunk) | cases | against the newest record |
|---|---|---|---|---|---|
| `virtual-nested-pools` | `virtual-wmem-pools` | `-DWM_SUBLET=ON`, Debug | fault, fault, fault, as declared | CAUGHT 22, NO-READING 1 | `../2026-10-19-q15-repair/B-eb-repaired` (Debug, exact bounds): every verdict and cause equal |
| `virtual-malloc` | `virtual-wmem` | `-DWM_SUBLET=OFF`, Debug | complete, fault, complete, as declared | MISSED 23 | `../2026-10-11-virtual-23/virtual-malloc`: every verdict equal |

The pools faults: cause 25 at `wm_probe` for cases 00 to 12, cause 28 at `wm_probe` for 13, 14, 15
and 19, cause 28 at `wm_write_probe` for 16, 17, 18, 20 and 21. N = 1 per cell.

**Case 22** reads NO-READING:unattributed, as in the 2026-10-19 Debug record. The second
`wmem_free` faults with cause 25 at the revoke of the already ended object. In a Debug image the
revoke is the out-of-line wrapper, so the pc lies in `capstone_cap_release` (before the conversion:
`capstone_cap_revoke_child`) and not in the declared sites `wmem_block_sublet_free` and
`wmem_block_sublet_release`. One fact changed with the conversion: the reset no longer calls this
wrapper. `wmem_sublet_block_renew` calls `capstone_cap_renew`, so every caller of
`capstone_cap_release` in wmem is a free or a realloc. `../2026-10-11-virtual-23` read this case
CAUGHT from an image that inlined the revoke into `wmem_block_sublet_free`. The Release image of
this date does the same (next section): its case 22 faults with cause 25 at
`wmem_block_sublet_free+76`.

## The Release preset gives no reading with this compiler

`release-preset/virtual-nested-pools` is the pools arm built exactly as the corpus README says,
with the port's `capstone-application` preset, which is a Release build. It reads
NO-READING:control-failed 23. Every control and every case faults as in the Debug run, with the
same causes, at the labelled instruction. The compiler now emits `wm_probe` without a frame, so
the label `wm_defect_probe` and the function `wm_probe` share one address. `tools/verdicts.py`
resolves the pc to the first symbol at that address, `wm_defect_probe`, and the runner counts a
control's fault only in `wm_probe` or `wm_write_probe`. The write probe keeps its label at
`wm_write_probe+4` and resolves.

This does not come from the conversion. `dev-release-preset/virtual-nested-pools` is the same
arm built from the unconverted tree (patch 0001 with the two-operand `CREVOKE`, its own SDK, the
same compiler, the Release preset) and run on that tree's platform, the model with the two-operand
`CREVOKE` (`inputs.json`: qemu `35529815bf4b`). It reads NO-READING:control-failed 23 in the same
way: 13 faults with cause 25 and 4 with cause 28 at `wm_defect_probe+0`, 5 with cause 28 at
`wm_write_probe+4`, and case 22 with cause 25 at `wmem_block_sublet_free+84`. `llvm-nm` shows
`wm_defect_probe` and `wm_probe` at one address there too (`0x1e4fc` in `01-uaf-jumbo`). That
platform carries the gate copy of the module, so its CPU line adds
`x-capstone-supervisor-slots=32`. The 2026-10-11 images had the label at `wm_probe+20`.

## Lifetime instructions in the images

Counted from the disassembly of `00-rpcrdma-write-offsets-global`:

| image | `CDERIVE` | `CREVOKE`, one operand | `CREVOKE`, two operands | `CRENEW` |
|---|---:|---:|---:|---:|
| pools, Debug | 10 | 5 | 0 | 2 |
| pools, Release | 26 | 6 | 0 | 2 |
| pools, Release, before the conversion (same compiler) | 42 | 0 | 8 | 0 |
| stock, Debug | 6 | 3 | 0 | 0 |

In the Release pools image wmem's own instructions are: `CREVOKE` in `wmem_block_sublet_free`,
`wmem_block_sublet_realloc` and `wmem_block_fast_sublet_realloc`, `CRENEW` in
`wmem_block_free_all` and `wmem_block_fast_free_all`. The other three `CREVOKE` are mallocng's
(`free`, `realloc`).

The jumbo overhead that `capstone/tests/exact-bounds-iii/wmem-q15-check.py` assumes was measured
with the build's own flags: with patch 0001 the block header is 48 bytes in both allocators, the
jumbo header of `block_fast` is 48 bytes and the chunk header is 16 bytes, so a jumbo block is the
request plus 64 bytes. Without the patch the headers are 32, 32 and 16 bytes. The check passes on
the patched header (6,000 and 60,000 issues, no failure, both controls fire).

## Platform

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`.
`inputs.json` of each bundle carries every file hash. Two settings are not in `inputs.json`:

* CPU properties: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`,
  so the `-cpu` line reads
  `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
  The 2026-10-11 bundles and run B of 2026-10-19 record exact bounds.
* Module: loaded without parameters (`unpinned=0`, `supervisor_slots=1`).

Each arm ran in one boot through `tools/board/appvm-session.py` with the kit as a virtual kit
(collect, boot, read): the three controls first, then the 23 cases, one process each.

## Build

    cmake --preset capstone-application -S capstone/ports/wireshark/wmem -B <out> -DCAPSTONE_SDK=<corpus SDK> \
        -DWM_CORPUS_DIR=<corpus or corpus/controls> -DWM_SUBLET=ON|OFF -DCMAKE_BUILD_TYPE=Debug
    cmake --build <out>

`release-preset` is the same without `-DCMAKE_BUILD_TYPE=Debug`. The SDK is a corpus SDK
(`-fno-builtin`). Patch 0001 is `bec33460c5ed`.
