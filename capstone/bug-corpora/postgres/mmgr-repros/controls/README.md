# Controls for the Capstone domain arms

Not cases. Each directory here is a program in the corpus's own case format, built by the same port
from this directory instead of the corpus root:

    cmake --preset capstone-domain -S <memory-contexts port> -B <out> \
        -DPG_CORPUS_DIR=<this directory>

`shared/run-defects.py --controls-build <out>` runs every control on each arm before any case, in
the same invocation and against the same port build, and records what it did. What a control must
do on an arm is not written here or in the runner: it is the arm configuration's, in
`tools/arms.json` (`replay-arena`, `replay-sublet`), and `tools/verdicts.py` refuses to score a
silence from an arm whose controls did not behave.

| control | replay-arena | replay-sublet |
|---|---|---|
| `00_control_uaf_chunk` -- one chunk, pfree'd, read through its alias | complete | fault at the probe |
