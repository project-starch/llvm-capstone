# Controls for the virtual arms

Not cases. Each directory here is a program in the corpus's own case format, built by the same port
from this directory instead of the corpus root:

    cmake --preset capstone-application -S <wmem port> -B <out> \
        -DCAPSTONE_SDK=<virtual SDK> -DWM_CORPUS_DIR=<this directory> [-DWM_SUBLET=ON]

`shared/run-defects.py --controls-hosted-build <out>` runs every control on the arm before any
case, in the same invocation and against the same port configuration, and records what it did.
What a control must do on an arm is not written here or in the runner: it is the arm
configuration's, in `tools/arms.json` (`virtual-wmem`, `virtual-wmem-pools`), and
`tools/verdicts.py` refuses to score a silence from an arm whose controls did not behave.

| control | virtual-malloc | virtual-nested-pools |
|---|---|---|
| `00_control_uaf_chunk` -- one packet-pool chunk, the packet reset, read through its alias | complete | fault at the probe |
| `01_control_uaf_jumbo` -- a packet-pool jumbo, which the reset hands to `g_free`, read through its alias | fault at the probe | fault at the probe |
| `02_control_bounds_chunk` -- one byte past a live 24-byte packet-pool chunk | complete | fault at the probe |

`01` is also the native-detect arm's positive control (`runners/run-asan.sh`): ASan must report
heap-use-after-free on it, since the jumbo is the one stock wmem object that reaches `free()`.
