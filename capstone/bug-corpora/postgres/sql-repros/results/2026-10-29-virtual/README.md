# 2026-10-29: all 11 cases on both Capstone arms, the CPython lane's dev-image kit on p13

The first Capstone reading of cases 10 and 11, taken with cases 01-09 on the same platform so the new
rows can be read against the recorded ones. The image here links test_decoding (case 11 decodes through it),
so its hash differs from the earlier ten-case run; cases 01-10 read identically to it. `verdict_bundles` still names the 2026-10-18 and 2026-10-21
bundles, and this run sits beside them.

## Platform

The kit is `../../../cpython/allocator-boundary-repros/results/2026-10-29-virtual/README.md`'s: QEMU
`e8a89c014e5a` at dev's pin `a9936e7037`, images-v7, launcher `b533a0689e31` and paged module
`74c60cd60df0`, the same `-cpu` extras, and the module at its defaults. The images and the fixture
were built from this branch (3c146cf40360), `build-virtual.sh postgres` with `CAPSTONE_SDK_CORPUS=1`
and `CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin`:

| | sha256 |
|---|---|
| image, virtual-malloc | `b942d239c562955b9d6d68c0c11759812fdaaaf92e6cca495b47e976cb2aa4bf` |
| image, virtual-pg-pools | `0f026782cfb9b99b94be01b86179dfae470b96e9913c4c29c7bb8376fc65f680` |
| fixture, `build-native16-fixture.py` | cluster stable tree `d63b081dc1fa...` |

One boot per arm (`shared/run-arm.py --virtual-kit`). Every control did what `tools/arms.json`
declares: `bounds-malloc` and `uaf-malloc` faulted on both arms, `bounds-palloc` and `uaf-palloc` on
`virtual-pg-pools`.

## Results

| case | virtual-malloc | virtual-pg-pools |
|---|---|---|
| 01-09 | identical, cell for cell, to `../2026-10-18-review/virtual-malloc` | identical, cell for cell, to `../2026-10-21-review/virtual-pg-pools` |
| 10 tuplestore double free | MISSED: reached, the one expected error, completed | CAUGHT: cause 25 in `SubletRelease`, by control (`control.sql` completed on the same image) |
| 11 reorderbuffer stale read | MISSED: reached, completed | CAUGHT: cause 25 in `ReorderBufferProcessTXN` (its host fault site) |

Case 10's pools fault is in the revoke the second `pfree` makes, below `pfree` and outside the case's
host `fault_sites` (`pfree`, `tuplestore_end`), so it counts by control. On `virtual-malloc` the
second free is the silent one the host showed: both frees stay inside an aset block.
