# 2026-10-29: all 36 cases on both Capstone arms, a dev-image kit on p13

The first Capstone reading of cases 33-36, taken as one full-corpus run per arm so that the new rows
sit in a bundle with every old one on the same platform. The images here carry zlib and the
_capstone_nomemory injector (case 36 needs them), so their hashes differ from the 35-case run; every
other cell is unchanged but case 15 on virtual-malloc (see below). `verdict_bundles` still names
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
| image, virtual-malloc | `6f19a8bc07f828b5d6ccc43e4ad4b3d2924f6410ae24f325e84efefbf76d254e` | `build-virtual.sh cpython`, `CPY_TEST_CAPI=1`, zlib + `_capstone_nomemory` |
| image, virtual-nested-pools | `23eebbe99aa28969ff43bdfee172fdf3a440fbe740182927e26159c0bf6c0f96` | the same with `CAPSTONE_VIRTUAL_LIFETIMES=1` |

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

One batch boot per arm, via `tools/board/appvm-session.py` with `--virtual-kit`: 62 steps, every one
run, the boot completed. In each arm the workload ran, and both controls did what `tools/arms.json`
declares: `virtual-malloc` completed `uaf-block` and `bounds-block`, `virtual-nested-pools` faulted on
both.

## Cases 33-36

| case | issue | virtual-malloc | virtual-nested-pools |
|---|---|---|---|
| 33 | gh-151295 | CAUGHT, cause 25 in `slot_bf_getbuffer` | CAUGHT, cause 25 in `slot_bf_getbuffer` |
| 34 | gh-151403 | CAUGHT, cause 25 in `PyOS_FSPath` | CAUGHT, cause 25 in `PyOS_FSPath` |
| 35 | gh-153539 | MISSED, completed | CAUGHT, cause 25 in `memcpy` |
| 36 | gh-148395 | MISSED, completed | CAUGHT, cause 25 in `memcpy` |

Each attribution is by function: the fault lies in one of the case's `fault_sites` from host ASan
(`../2026-10-29-asan/sites.tsv`).

## The 35 older cases against the 35-case run

| arm | tally | cells that differ |
|---|---|---|
| virtual-malloc | CAUGHT 16, MISSED 17, NO-READING 3 | vs the 35-case run of this day, one cell moved: 15 CAUGHT -> MISSED. 15 (gh-148973) is a known run-to-run-unstable cell on this arm (its vp2 record was NO-READING), not a zlib effect; the other 34 old cells are identical |
| virtual-nested-pools | CAUGHT 31, MISSED 3, NO-READING 2 | identical to the 35-case run of this day in every old cell; case 36 adds one CAUGHT |

The run was not pre-registered case by case. The runner's arm-level prediction is in each
`inputs.json`.
