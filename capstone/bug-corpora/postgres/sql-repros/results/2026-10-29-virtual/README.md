# 2026-10-29: all 10 cases on both Capstone arms, the CPython lane's dev-image kit on p13

The first Capstone reading of case 10, taken with cases 01-09 on the same platform so the new row can
be read against the recorded ones. `verdict_bundles` still names the 2026-10-18 and 2026-10-21
bundles, and this run sits beside them.

## Platform

The kit is `../../../cpython/allocator-boundary-repros/results/2026-10-29-virtual/README.md`'s: QEMU
`e8a89c014e5a` at dev's pin `a9936e7037`, images-v7, launcher `b533a0689e31` and paged module
`74c60cd60df0`, the same `-cpu` extras, and the module at its defaults. The images and the fixture
were built from this branch (3c146cf40360), `build-virtual.sh postgres` with `CAPSTONE_SDK_CORPUS=1`
and `CAPSTONE_SDK_EXTRA_CODEGEN=-fno-builtin`:

| | sha256 |
|---|---|
| image, virtual-malloc | `4a74f447082c55059cb9f0a90ac4647a131b65ccc227fb84296f80156e856d70` |
| image, virtual-pg-pools | `7cb1f5c13c90c0c7e4d654b9a224611bcb72b50dbbbc1a3ac0848876a73cdb8d` |
| fixture, `build-native16-fixture.py` | cluster stable tree `d63b081dc1fa...` |

One boot per arm (`shared/run-arm.py --virtual-kit`). Every control did what `tools/arms.json`
declares: `bounds-malloc` and `uaf-malloc` faulted on both arms, `bounds-palloc` and `uaf-palloc` on
`virtual-pg-pools`.

## Results

| case | virtual-malloc | virtual-pg-pools |
|---|---|---|
| 01-09 | identical, cell for cell, to `../2026-10-18-review/virtual-malloc` | identical, cell for cell, to `../2026-10-21-review/virtual-pg-pools` |
| 10 tuplestore double free | MISSED: reached, the one expected error, completed | CAUGHT: cause 25 in `SubletRelease`, by control (`control.sql` completed on the same image) |

Case 10's pools fault is in the revoke the second `pfree` makes, below `pfree` and outside the case's
host `fault_sites` (`pfree`, `tuplestore_end`), so it counts by control. On `virtual-malloc` the
second free is the silent one the host showed: both frees stay inside an aset block.
