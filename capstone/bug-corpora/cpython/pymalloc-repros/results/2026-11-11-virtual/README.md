# cpython/pymalloc-repros on the virtual profile, 2026-11-11

Both virtual arms, measured again after the processor model changed its lifetime instructions:
`CREVOKE` takes the object alone, and the two-operand form is an illegal instruction.

| arm | configuration | build | control `uaf-block` | cases | against `../2026-10-10-patch0003` |
|---|---|---|---|---|---|
| `virtual-nested-pools` | `virtual-pymalloc-pools` | `-DPYMALLOC_SUBLET=ON` (patches 0003 and 0004) | fault in `read_probe`, as declared | CAUGHT 20 / 20 | every verdict, cause and site equal |
| `virtual-malloc` | `virtual-pymalloc` | `-DPYMALLOC_SUBLET=OFF` | complete, as declared | MISSED 20 / 20 | every verdict equal |

Every pools fault is cause 25 at `read_probe`, the labelled stale read, as in the earlier bundle.
N = 1 per cell.

## Patch 0003 alone gives no reading on this model

`patch0003-alone/virtual-nested-pools` is the pools arm built from patch 0003 without 0004, which
was what `-DPYMALLOC_SUBLET=ON` built until this date. It reads NO-READING:control-failed 20 / 20.
The control and every case fault with cause 2 (illegal instruction) in
`capstone_cap_revoke_child`, at the first `pymalloc_free`: the faulting word is `0xa4b5105b`,
`CREVOKE` with rs1 = a0 and rs2 = a1, the two-operand form of the capstone-sublet bitstream.
Patch 0004 replaces the two revocations by `capstone_cap_release`, and
`cmake/prepare-source.py --variant protected-virtual` applies it. `--variant protected` stays
patch 0003 alone, byte for byte, for the silicon arm.

## Lifetime instructions in the images

Counted from the disassembly of `defect-00` (funct7 0x51 `CDERIVE`, 0x52 `CREVOKE`, 0x54 `CRENEW`):

| image | `CDERIVE` | `CREVOKE`, one operand | `CREVOKE`, two operands | `CRENEW` |
|---|---:|---:|---:|---:|
| pools, patches 0003 and 0004 | 6 | 2 (`capstone_cap_release`, `__libc_free`) | 0 | 0 |
| pools, patch 0003 alone | 6 | 1 (`__libc_free`) | 1 (`capstone_cap_revoke_child`) | 0 |
| stock | 4 | 1 (`__libc_free`) | 0 | 0 |

The two `CDERIVE` that the pools image adds are obmalloc's (`capstone_cap_derive`, out of line in
this Debug build, with its zero-size refusal form). The other four belong to the runtime and to
mallocng.

## Platform

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`.
`inputs.json` of each bundle carries every file hash. Two settings are not in `inputs.json`:

* CPU properties: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`,
  so the `-cpu` line reads
  `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
  The earlier bundles record `exact_bounds: true`. With the model's default properties (neither
  set) the first launch halted the machine in the monitor (cause 24), on both arms.
* Module: loaded without parameters (`unpinned=0`, `supervisor_slots=1`).

Each arm ran in one boot through `tools/board/appvm-session.py` with the kit as a virtual kit
(collect, boot, read): the control first, then the twenty cases, one process each.

## Build

    export CAPSTONE_SDK=<corpus SDK>
    shared/build-cases.sh capstone-application OUT-ON  -DPYMALLOC_SUBLET=ON
    shared/build-cases.sh capstone-application OUT-OFF -DPYMALLOC_SUBLET=OFF
    PYC_CASES=$PWD/controls shared/build-cases.sh capstone-application CTL-ON  -DPYMALLOC_SUBLET=ON
    PYC_CASES=$PWD/controls shared/build-cases.sh capstone-application CTL-OFF -DPYMALLOC_SUBLET=OFF

The SDK is a corpus SDK (`-fno-builtin`). The patch series of the pools build:
0001 `c65885a5883f`, 0002 `54df540ab50f`, 0003 `34125b58651a` (unchanged), 0004 `ad297e2a1241`.
