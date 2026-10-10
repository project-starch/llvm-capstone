# Controls for the virtual arms

Not cases. Each directory here is a program in the corpus's own case format, built by the same port
from this directory instead of the corpus root:

    cmake --preset capstone-application -S <memory-contexts port> -B <out> \
        -DCAPSTONE_SDK=<virtual SDK> -DPG_CORPUS_DIR=<this directory> [-DPG_SUBLET=ON]

`shared/run-defects.py --controls-hosted-build <out>` runs every control on the arm before any
case, in the same invocation and against the same port configuration, and records what it did.
What a control must do on an arm is not written here or in the runner: it is the arm
configuration's, in `tools/arms.json` (`virtual-mallocng-replay`, `virtual-mallocng-replay-pools`),
and `tools/verdicts.py` refuses to score a silence from an arm whose controls did not behave.

| control | virtual-malloc | virtual-pg-pools |
|---|---|---|
| `00_control_uaf_chunk` -- one chunk, pfree'd, read through its alias | complete | fault at the probe |
