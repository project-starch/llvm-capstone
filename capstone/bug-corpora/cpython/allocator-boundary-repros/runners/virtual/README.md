# Virtual Capstone

The 32 triggers run by the whole interpreter as a Capstone application on the virtual profile,
one process per case on a persistent VM. CPython's one heap there is musl mallocng.

| arm | configuration | build | run |
|---|---|---|---|
| `virtual-malloc` | `virtual-cpython` | `CPY_TEST_CAPI=1 build-virtual.sh cpython OUT` | as is |
| `virtual-nested-pools` | `virtual-cpython-pools` | `CPY_TEST_CAPI=1 CPY_SUBLET=1 build-virtual.sh cpython OUT` | as is |

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
2. The interpreter must run a JSON/GC workload.
3. The two controls in `../../controls/` must do what `tools/arms.json` says.
4. Then each case is run. A fault counts when it lies in one of the case's `fault_sites`, or,
   when it does not and the case has a `negative_control.py`, when that control (the same
   traffic, the offending access made valid) then runs to its end on the same image without a
   fault and passes its own self-test.

The interpreter's main stack is 8 MiB (`build-virtual.sh` sets it for cpython): CPython 3.13 sizes
its C recursion limit for the stack Linux gives a process, and with the runtime's 1 MiB default a
deep recursion overflows the stack before the interpreter's guard raises `RecursionError`.

`tools/verdicts.py` judges every row.
