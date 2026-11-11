# The Sublet ports of memsys5 on the one-operand lifetime instructions, in the 256 MiB configuration (2026-11-11)

The Sublet rows of [`../2026-10-26-poisoncap-heap256/`](../2026-10-26-poisoncap-heap256/README.md), run again after the
lifetime instructions changed: `CREVOKE` takes the object alone, and `CRENEW` ends an object's lifetime and begins a new
one in its place. The three memsys5 patches were converted to them
(`ports/sqlite/sublet/sublet-3220000-memsys5.patch`, `sublet-3220000-memsys5-block.patch` and its generator
`make-sublet-block-port.py`). The block port is the only SQLite code that executes `CRENEW`: `memsys5Renew` and
`lookasideRenew` are one `capstone_cap_renew` each, where they revoked the old child and derived a sibling before.
PoisonCap was not run again; its rows stay in the 2026-10-26 record.

Same configuration as 2026-10-26 for the three Sublet builds: a 256 MiB heap from `malloc`, minimum request 16,
lookaside on, the controls with the corpus's 1 MiB static heap. Same verdict rule: a case counts when the process faults
in one of the case's `fault_sites`. Every configuration ran twice, and the two runs agree in every row (`tabulate.out`).

## Results, the 17-case population of 2026-10-26

| configuration | run | fault at site | not at site | 2026-10-26 |
|---|---|---|---|---|
| Sublet, block port | `sublet-block`, `-rep2` | **17** | none | 17, none |
| Sublet, block port, ablation (`--realloc-keeps-lifetime`) | `sublet-block-keeps`, `-rep2` | 16 | 12 | 16, 12 |
| Sublet, request-bounds port | `sublet-driver`, `-rep2` | 17 | none | 17, none |

The population is 00 03 05 06 07 10 12 14 15 16 17 19 20 21 22 23 24. Each of the 17 cells has the cause and the
function it had on 2026-10-26, in all three configurations. Case 12 is still the cell the realloc lifetime decides:
cause 25 in `fts5SegIterNextPage` on the block port, completed on the ablation.

## Controls

| control | block port | ablation | request-bounds port |
|---|---|---|---|
| `control_uaf_mem5` | cause 25 in `control_read` | cause 25 in `control_read` | cause 25 in `control_read` |
| `control_bounds_mem5` | cause 28 in `control_write` | cause 28 in `control_write` | cause 28 in `control_write` |
| `control_realloc_same_mem5` | cause 25 in `control_read` | returns, `same_address=1 kept=1` | not built |
| `control_realloc_shrink_mem5` | cause 25 in `control_read` | returns, `same_address=1 kept=1` | not built |
| `control_realloc_fail_mem5` | returns, `failed=1 kept=1` | returns, `failed=1 kept=1` | not built |
| `control_capacity_mem5` | returns | returns | not built |

`run-virtual-staged.py` checks these expectations itself and exited 0 in all six runs. The two realloc controls are the
direct reading of `CRENEW`: the realloc keeps the block, the old pointer faults with cause 25, and on the ablation, which
compiles the renewal out, the new pointer has the old address and the block's contents.

## The rows outside the population

`run-virtual-staged.py --cases temporal` runs the 25 cases `results/matrix.tsv` lists as temporal. Three of the eight
outside the population read differently from 2026-10-26, the same way in all three configurations:

| case | 2026-10-26 | 2026-11-11 |
|---|---|---|
| 02 | cause 24 in `sqlite3BtreeUpdateMeta` | cause 25 in `sqlite3BtreePager` |
| 11 | returned | cause 25 in `memcmp` |
| 13 | returned | cause 25 in `sqlite3Fts5PoslistNext64` |

The cause is the corpus, not the instructions: the triggers of 02, 11 and 13 were replaced on 2026-10-29, after the
2026-10-26 record, with ones that reach their defects (the corpus README, "Case 08 is WITHDRAWN"). The new readings are
the ones `results/2026-10-29-virtual/virtual-nested-pools` has for these three cases. The other 22 rows are unchanged.

## Lifetime instructions in the images

Counted in the executable sections with `ports/common/application/lifetime-instructions.py`. The first number of each
triple is `CDERIVE`, then `CREVOKE`, then `CRENEW`.

| image (`-O0`) | count |
|---|---|
| stock memsys5 (`control_uaf_mem5.dom`, `build-virtual.py --memsys5`) | 6, 3, 0: the C library's own, in mallocng |
| request-bounds port | 8, 4, 0 |
| block port | 8, 4, 1 |
| block port, ablation | 8, 4, 0 |

## Platform and inputs

- Tree: branch `one-operand-crevoke-crenew` at 59201bdf2, QEMU at ea57497908.
- Engine source: the adapted amalgamation 7fc92e19..., regenerated with `ports/sqlite/adapt-sqlite-322.sh`.
  `make-sublet-block-port.py` applied to the request-bounds source reproduces the source that
  `sublet-3220000-memsys5-block.patch` gives, byte for byte.
- Patches: `sublet-3220000-memsys5.patch` f96ee0329a08..., `sublet-3220000-memsys5-block.patch` 508f9f53b0de....
- SDK: a corpus SDK (`CAPSTONE_SDK_CORPUS=1`, effective `-fno-builtin`) built from the tree with
  `runtime/virtual/build-sdk.sh`; compiler 0bbb1f88bc90....
- Platform: QEMU 2d5a71078904..., kernel 48eb725a40e4..., firmware 5b5c047a0030..., rootfs 62049800ed0f...,
  launcher 0f8e0f903725..., module e47bca62addf.... The module is loaded without parameters, so it runs pinned with one
  context. The guest CPU is in each run's `cpu.txt`; `x-capstone-test-csrs=true` is what a pinned boot of this kernel
  needs. One hart, 2 GiB.
- Six boots at once, each with a lock file of its own, inside one exclusive hold of the host's shared QEMU lock.

## Files

`runs/<run>/`: `results.jsonl` (one row per process, with its output), `build.json` of the build it ran and `cpu.txt`.
`tabulate.out`: the population per run, the controls, and whether the two runs of a configuration agree.

## Limits

QEMU only. Function-level sites, as on 2026-10-26. The lookaside path of `sqlite3DbRealloc` (`lookasideRenew`) has no
control here, as before; the application gate in `ports/sqlite/app/results/2026-11-11-nested-heaps` executes it.
