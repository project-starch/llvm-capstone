# wireshark/wmem-repros on the virtual profile, Release build, 2026-11-12

Both virtual arms, built with the port's documented preset and run on the platform of
[`../2026-11-11-virtual/`](../2026-11-11-virtual/README.md). The preset `capstone-application` is a
Release build (`-O3 -DNDEBUG`). These are the bundles `corpus.json` names under `verdict_bundles`.

| arm | configuration | build | controls (uaf-chunk, uaf-jumbo, bounds-chunk) | cases | against `../2026-11-11-virtual` (Debug) | against `../2026-10-11-virtual-23` |
|---|---|---|---|---|---|---|
| `virtual-nested-pools` | `virtual-wmem-pools` | `-DWM_SUBLET=ON`, Release | fault, fault, fault, as declared | CAUGHT 23 | 22 verdicts and causes equal, case 22 differs | every verdict, cause and function equal |
| `virtual-malloc` | `virtual-wmem` | `-DWM_SUBLET=OFF`, Release | complete, fault, complete, as declared | MISSED 23 | every verdict equal | every verdict equal |

The pools controls fault in `wm_probe` with cause 25 (uaf-chunk, uaf-jumbo) and cause 28
(bounds-chunk). The uaf-jumbo control of the malloc arm faults in `wm_probe` with cause 25.
N = 1 per cell.

## Every verdict against the two records

Each cell gives the verdict, the cause and the function with the pc's offset in it. The offsets
differ between the three builds because the label lies at `wm_probe+0` in these Release images,
at `wm_probe+40` in the Debug images and at `wm_probe+20` in the images of 2026-10-11.

| case | pools, this record | pools, 2026-11-11 Debug | pools, 2026-10-11 | malloc, this record | malloc, 2026-11-11 Debug | malloc, 2026-10-11 |
|---|---|---|---|---|---|---|
| 00 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 01 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 02 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 03 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 04 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 05 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 06 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 07 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 08 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 09 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 10 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 11 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 12 | CAUGHT, 25, `wm_probe+0` | CAUGHT, 25, `wm_probe+40` | CAUGHT, 25, `wm_probe+20` | MISSED | MISSED | MISSED |
| 13 | CAUGHT, 28, `wm_probe+0` | CAUGHT, 28, `wm_probe+40` | CAUGHT, 28, `wm_probe+20` | MISSED | MISSED | MISSED |
| 14 | CAUGHT, 28, `wm_probe+0` | CAUGHT, 28, `wm_probe+40` | CAUGHT, 28, `wm_probe+20` | MISSED | MISSED | MISSED |
| 15 | CAUGHT, 28, `wm_probe+0` | CAUGHT, 28, `wm_probe+40` | CAUGHT, 28, `wm_probe+20` | MISSED | MISSED | MISSED |
| 16 | CAUGHT, 28, `wm_write_probe+4` | CAUGHT, 28, `wm_write_probe+44` | CAUGHT, 28, `wm_write_probe+24` | MISSED | MISSED | MISSED |
| 17 | CAUGHT, 28, `wm_write_probe+4` | CAUGHT, 28, `wm_write_probe+44` | CAUGHT, 28, `wm_write_probe+24` | MISSED | MISSED | MISSED |
| 18 | CAUGHT, 28, `wm_write_probe+4` | CAUGHT, 28, `wm_write_probe+44` | CAUGHT, 28, `wm_write_probe+24` | MISSED | MISSED | MISSED |
| 19 | CAUGHT, 28, `wm_probe+0` | CAUGHT, 28, `wm_probe+40` | CAUGHT, 28, `wm_probe+20` | MISSED | MISSED | MISSED |
| 20 | CAUGHT, 28, `wm_write_probe+4` | CAUGHT, 28, `wm_write_probe+44` | CAUGHT, 28, `wm_write_probe+24` | MISSED | MISSED | MISSED |
| 21 | CAUGHT, 28, `wm_write_probe+4` | CAUGHT, 28, `wm_write_probe+44` | CAUGHT, 28, `wm_write_probe+24` | MISSED | MISSED | MISSED |
| 22 | CAUGHT, 25, `wmem_block_sublet_free+76` | NO-READING:unattributed, 25, `capstone_cap_release+40` | CAUGHT, 25, `wmem_block_sublet_free+96` | MISSED | MISSED | MISSED |

## Case 22

    capstone-exec: domain fault cause=25 pc=0x3f8891093c address=0x0 ra=0x3f8890e414 fp=0x3f887ffe40 entry=0x3f889007d4 code=0x3f88900000-0x3f88980000 last=0x42 preparing=0x42 sha256=8fe7b8f111635697fb8f2d0dbd7b37c257807aaaaf0a00f26cb7ceb261eb84b7 image=/mnt/vm/images/22-usbhid-output-usages-freed-twice-8fe7b8f111635697.dom chain=0x3f8890e414,0x3f8890eafc,0x3f88900a38,0x3f889000a8

The pc is link address `0x2093c`, `wmem_block_sublet_free+76`, a fault site the case declares, so
the attribution is `function`. The instruction there is the one-operand `CREVOKE` of the freed
object (`rs1 = a1`), which the compiler inlined from `wmem_block_sublet_release`. The return
address lies in `wm_case_run` (`+512`, the first entry of `chain`): the fault is the case's second
`wmem_free` of the same chunk, whose lifetime the first free had ended. The Debug image of 2026-11-11 makes the same revoke in the
out-of-line wrapper, so its pc lies in `capstone_cap_release+40` and the case reads
NO-READING:unattributed there, with the same cause. That record stays as it was measured.

## The judging rule that changed: a pc resolves to the function that holds it

`tools/verdicts.py` `Symbols.lookup` now resolves a fault's pc to the function that holds it, and
to a label (a symbol without a size) only where no function holds the pc. Until 2026-11-12 it
returned the first symbol that `llvm-nm` lists at the pc's address. The compiler of 2026-11-11
emits `wm_probe` without a frame in a Release image, so the asm label `wm_defect_probe` lies on the
function's first instruction, `llvm-nm` lists the label first, and the old lookup named the label.
The runner counts a fault at the probe only when the lookup names `wm_probe` or `wm_write_probe`.
It therefore read all three controls as `none` and every case as NO-READING:control-failed, 23 of
23 (`../2026-11-11-virtual/release-preset`, and `dev-release-preset` for the tree before the
conversion). That reading was an artefact of the symbol lookup: each of those faults lay at the
labelled instruction, with the cause of the Debug run. The pools images of this record are the
images of that bundle byte for byte (sha256 of the 23 case programs and of the 3 controls), so
only the lookup differs. Resolving the 26 fault lines of that boot again with both lookups, without
a boot, changes 20: 15 with cause 25 and 5 with cause 28 move from `wm_defect_probe+0` to
`wm_probe+0`. The other 6, the five faults in `wm_write_probe` and case 22, resolve as before. No
expectation, control or case changed. The unit tests of the rule are the class `SymbolLookup` in
`tools/test_verdicts.py`.

## The lifetime controls

`controls/lifetime/run-lifetime-controls.py` and its `block_fast` sibling judge the stale read by
the label's own address and use the lookup only to record the function. The four results of
`../2026-11-11-controls/` were judged again from their recorded output lines, on rebuilt images
with the recorded hashes (14 of 14), without a boot. Every program reads AS PREDICTED with either
lookup, 14 of 14. One recorded field differs: the `function` of the two stale-read faults reads
`wm_probe` with the new lookup and `wm_defect_probe` in the recorded files, and `label` reads
`wm_defect_probe` in both. The recorded files stay as they are.

## Lifetime instructions in the images

The pools images hold the instructions `../2026-11-11-virtual/` counted for its Release image
(`CDERIVE` 26, one-operand `CREVOKE` 6, `CRENEW` 2). The stock Release image of
`00-rpcrdma-write-offsets-global` holds `CDERIVE` 6 and one-operand `CREVOKE` 3, all in libc's heap
code, the counts of the stock Debug image.

## Platform

The kit and its settings are those of `../2026-11-11-virtual/`. Every hash in `inputs.json` equals
that record's except `judge`, the hash of `tools/verdicts.py`. Two settings are not in
`inputs.json`:

* CPU properties: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`,
  so the `-cpu` line reads
  `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
* Module: loaded without parameters (`unpinned=0`, `supervisor_slots=1`).

Each arm ran in one boot through `tools/board/appvm-session.py` with the kit as a virtual kit
(collect, boot, read): the three controls first, then the 23 cases, one process each.

## Build

    cmake --preset capstone-application -S capstone/ports/wireshark/wmem -B <out> -DCAPSTONE_SDK=<corpus SDK> \
        -DWM_CORPUS_DIR=<corpus or corpus/controls> -DWM_SUBLET=ON|OFF
    cmake --build <out>

The SDK is a corpus SDK (`-fno-builtin`). Patch 0001 is `bec33460c5ed`.
