# ffmpeg/pool-repros on the virtual profile, 2026-11-11

Both virtual arms, re-measured after the buffer-pool port's patch 0003 changed its design and the
processor model changed its lifetime instructions. Patch 0003 no longer derives a child per
hand-out. Every pool entry is its own allocation, and `pool_release_buffer` and `pool_return_entry`
renew that allocation's lifetime with `CRENEW` when a hand-out returns. The patch adds no field.

| arm | configuration | cases | tally | against `../2026-10-11-virtual` |
|---|---|---|---|---|
| `virtual-nested-pools` | `virtual-ffmpeg-pool-sublet` (`-DFFPOOL_SUBLET=ON`) | 4 | CAUGHT 4 | every verdict equal |
| `virtual-malloc` | `virtual-ffmpeg-pool-stock` | 4 | MISSED 4 | every verdict equal |

Controls, each in the arm's own boot: `bounds-malloc` (cause 28) and `uaf-malloc` (cause 25) fault
on both arms. `pool-return-90` faults on `virtual-nested-pools` (cause 25 at `read_probe`) and
completes on `virtual-malloc`, as `tools/arms.json` declares.

The four catches fault with cause 25 in `ff2_case_run`, at the stale access the line table names,
the same lines as on 2026-10-11:

| case | fault site |
|---|---|
| 00 af_join | `case.c:85` |
| 01 h264_refs | `case.c:48` |
| 02 vidstab | `case.c:33` |
| 03 vvc | `case.c:104` |

Each fixed run completes with `VERDICT FIXED` and exit 0 under the same protection. A fixed run ends
its pool, so it also frees every entry through the renewed pointer.

## Platform

The kit is the tree's virtual platform on the model with the one-operand `CREVOKE` and `CRENEW`;
`inputs.json` of each bundle carries every file hash. Two settings are not in `inputs.json`:

* CPU properties: `CAPSTONE_QEMU_CPU_EXTRA=x-capstone-test-csrs=true,x-capstone-exact-bounds=true`.
  Exact bounds is the profile the 2026-10-11 bundles ran (then `run-staged.py`'s default).
* Module: loaded without parameters (`unpinned=0`, `supervisor_slots=1`).

## Build

    CAPSTONE_SDK=<corpus SDK> shared/build-cases.sh capstone-application <cases> -DFFPOOL_SUBLET=ON
    CAPSTONE_SDK=<corpus SDK> controls/virtual/shared/build-cases.sh capstone-application <controls> -DFFPOOL_SUBLET=ON
    tools/run-virtual-cases.py --corpus . --arm virtual-nested-pools --prebuilt <cases> --prebuilt-controls <controls> ...

and the same with `-DFFPOOL_SUBLET=OFF` for `virtual-malloc`. Lifetime instructions in the nested
build's pool objects (`ports/common/application/lifetime-instructions.py`): `buffer.c.o` CRENEW 1,
`refstruct.c.o` CRENEW 1, no CDERIVE and no CREVOKE. The stock build's objects carry none.
