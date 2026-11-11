# CPython pymalloc: parent, sibling and successor controls, 2026-11-11

CPython 3.13.7's row of `capstone/docs/plans/2026-10-11-nested-lifetime-controls.md`, measured
again after patch 0014 and the processor model changed their lifetime instructions. The protected
arm is the interpreter with patch 0014 (`CPY_SUBLET=1`), the release arm the same source without
it. The harness is `controls/nested-lifetime/`; `results.json` holds every result line.

Patch 0014 now derives every block from `arena_ptr`, the arena as mallocng returned it, and
`pymalloc_free` ends the block with `CREVOKE(block)`. The parent reference of control 1 is
therefore A's direct parent. In the 2026-10-11 record it was A's grandparent, through `arena_life`.

## Result

**Every prediction held, on both arms, in one boot per arm** (the 2026-10-11 record has three).

| control | protected (`CPY_SUBLET=1`) | release |
|---|---|---|
| 1 parent | completed: pool free-list head = A, A's link = the old head, A's last byte 0xa1, write 0x5a reads back 0x5a | completed, the same readings |
| 2 sibling | completed, B intact (0 of 48 bytes differ), write and read-back of B[47] 0x5a | completed, the same |
| 3 successor | C = A on the first allocation; 48 bytes written 0xc3 through C and read back | the same |
| stale read through A | **fault, cause 25**, pc at `pymctl_stale_read`, fault address = A | **completed**, read 0xc3, C's data |

| arm | parent A | sibling A / B | successor A = C | stale A = C | stale outcome |
|---|---|---|---|---|---|
| protected | 0x3f911e64b0 | 0x3fb419a4b0 / 0x3fb419b140 | 0x3fb5ee3140 | 0x3f9817a4b0 | fault cause=25 pc=0x3f973f9544 (image 0x409544 = `pymctl_stale_read`) |
| release | 0x3fb565a4b0 | 0x3f85f224b0 / 0x3f85f23140 | 0x3faa53b140 | 0x3f9f29a4b0 | completed, read 0xc3 |

In both boots A sat at offset 9392 in a 48-byte (szidx 2) pool of arena 0, the pool held 337
blocks after the free, and the successor took one allocation. After each boot `capstone-vexec
--stats` reported 0 live domains, 0 live regions and 0 live bytes. The stale probe is compiled
without a frame by this compiler, so the label `pymctl_stale_read` is `pymctl_stale_probe+0`
(`+20` in the earlier record); the runner compares the label's address.

## Platform and build

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`
(qemu `2d5a7107890e`, launcher `0f8e0f903725`, module `e47bca62addf`), which `run-controls.py`
registers since this date. One hart, exact bounds, `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true`,
the module loaded without parameters. `run-controls.py` gives each boot a private lock file; both
boots ran while the invocation held the host's shared QEMU lock.

    CAPSTONE_SDK_CORPUS=1 CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin controls/nested-lifetime/build-arm.sh protected OUT/protected
    CAPSTONE_SDK_CORPUS=1 CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin controls/nested-lifetime/build-arm.sh release   OUT/release
    controls/nested-lifetime/run-controls.py RESULTS --kit <kit> --llvm-bin <toolchain>/bin \
        --native OUT/release/source/build-python/python --arm protected=OUT/protected --arm release=OUT/release

Images: protected `06fd1028c624`, release `ebe3a60da74f`.
