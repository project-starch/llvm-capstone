# 2026-10-29: all 35 cases on both Capstone arms, a dev-image kit on p13

The first Capstone reading of cases 33-35, taken as one full-corpus run per arm so that the new rows
sit in a bundle with every old one on the same platform. `verdict_bundles` still names
`2026-10-10-sublet-isa`. Moving the corpus's record to another platform is the lead's decision, as the
2026-10-22 batch pair already recorded (061a3256942c). So this bundle is beside the record, and
case.json carries no verdict derived from it.

## Platform

| part | sha256 | origin |
|---|---|---|
| QEMU | `e8a89c014e5a...` | capstone-qemu at dev's pin `a9936e7037` |
| kernel, firmware, rootfs | `e833dfb1...`, `4dd4a2ad...`, `62049800...` | images-v7 |
| launcher | `b533a0689e31...` | `build-adapter.sh` from this branch (a4a4d8f8b1df) |
| module | `74c60cd60df0...` | the same build; the paged table |
| compiler | `500f854f46a6` | unchanged on dev since |
| image, virtual-malloc | `a650d8c1712e4648fd22622a677c69ee07e135c7b14b6c037a58788a4ffd34fa` | `build-virtual.sh cpython`, `CPY_TEST_CAPI=1` |
| image, virtual-nested-pools | `1e3674938089bb61fa375a92e16e71ba6e402eac7a0a7d3cb5cd14c098feb753` | the same with `CAPSTONE_VIRTUAL_LIFETIMES=1` |

QEMU's `-cpu` was
`rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,x-capstone-exact-bounds=true,x-capstone-test-csrs=true,x-capstone-addr=sv39`.
`inputs.json` does not record it. Without `x-capstone-test-csrs` the paged module's first application
run oopses the kernel. The module ran at its defaults, one supervisor slot. The batch path can set module
parameters only through sysfs after load, and the two the PHP platform passes at load
(`node_initial_pages=1024`, `supervisor_slots=32`) were refused that way. No trigger starts a thread.
The guest booted without `norandmaps`, unlike the 2026-10-22 pair, with `PYTHONHASHSEED=0` as there.

Both images are corpus builds with `CAPSTONE_SDK_CORPUS=1`. They also needed
`CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin`, because `ports/common/application/build.py`'s relink SDK reads only
that variable. Without it the runner's C-81 gate refused both images before anything ran.

One batch boot per arm, via `tools/board/appvm-session.py` with `--virtual-kit`: 60 steps, every one
run, the boot completed. In each arm the workload ran, and both controls did what `tools/arms.json`
declares: `virtual-malloc` completed `uaf-block` and `bounds-block`, `virtual-nested-pools` faulted on
both.

## Cases 33-35

| case | issue | virtual-malloc | virtual-nested-pools |
|---|---|---|---|
| 33 | gh-151295 | CAUGHT, cause 25 in `slot_bf_getbuffer` | CAUGHT, cause 25 in `slot_bf_getbuffer` |
| 34 | gh-151403 | CAUGHT, cause 25 in `PyOS_FSPath` | CAUGHT, cause 25 in `PyOS_FSPath` |
| 35 | gh-153539 | MISSED, completed | CAUGHT, cause 25 in `memcpy` |

Each attribution is by function: the fault lies in one of the case's `fault_sites` from host ASan
(`../2026-10-29-asan/sites.tsv`).

## The 32 older cases against their records

| arm | tally | cells that differ |
|---|---|---|
| virtual-malloc | CAUGHT 17, MISSED 15, NO-READING 3 | 2 of 32 against the 2026-10-22 declared run: 09 CAUGHT -> NO-READING inconclusive (reached, then neither completed nor faulted), 18 MISSED -> CAUGHT (cause 28 in `mmap_ass_subscript`, its ASan site). 11 of 32 against 2026-10-10: ten of the eleven cells the 2026-10-22 pair moved, moved the same way, and 18; 09 reads as it did on 2026-10-10 |
| virtual-nested-pools | CAUGHT 30, MISSED 3, NO-READING 2 | 5 of 32 against 2026-10-10: 01, 19, 21 NO-READING unattributed -> CAUGHT, 04 CAUGHT -> MISSED, 18 MISSED -> CAUGHT. Each moves as it does on virtual-malloc. There is no batch-platform pools record to compare with |

The run was not pre-registered case by case. The runner's arm-level prediction is in each
`inputs.json`.
