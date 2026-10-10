# Virtual Capstone

The 32 triggers run by the whole interpreter as a Capstone application on the virtual profile,
one process per case on a persistent VM. CPython's one heap there is musl mallocng.

| arm | configuration | build | run |
|---|---|---|---|
| `virtual-malloc` | `virtual-cpython` | `CPY_TEST_CAPI=1 build-virtual.sh cpython OUT` | as is |
| `virtual-nested-pools` | `virtual-cpython-pools` | `CPY_TEST_CAPI=1 CPY_SUBLET=1 build-virtual.sh cpython OUT` | `CPY_SUBLET_MODE=1` |

(`build-virtual.sh` is `capstone/ports/common/application/build-virtual.sh`.)

    capstone_vm --state VM up --profile virtual --exact-bounds ...
    runners/virtual/run-virtual.py results/<stamp>/virtual-malloc --arm virtual-malloc \
        --state VM --build OUT --llvm-bin <llvm>/bin --raw RAW

The runner stages the release's `Lib` under the VM share as `PYTHONHOME` (compiled by the build's
native 3.13.7), and each case's `.py` files into a directory of its own. It runs `trigger.py`
through a small launcher that prints `ABR BEGIN` before it and `ABR RETURNED`, `ABR EXIT` or
`ABR RAISED` after.

In order, each must pass before anything after it counts:

1. The build's `image/manifest.json` must say profile `virtual` and nested `none` or `cpython`
   for the arm, and the VM must be a virtual one.
2. The interpreter must run a JSON/GC workload. On the pools arm it must also report
   `CPY-SUBLET mode=1`.
3. The two controls in `../../controls/` must do what `tools/arms.json` says.
4. Then each case is run. A fault counts only in the case's `fault_sites`.

`tools/verdicts.py` judges every row.
