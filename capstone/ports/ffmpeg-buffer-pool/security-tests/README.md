# FFmpeg pool security tests

Tests of bounds and lifetime enforcement against the ported AVBufferPool and
AVRefStructPool APIs.

| File | Purpose |
|---|---|
| `security.c` | Valid-reference controls, stale accesses, invalid releases and bounds probes |
| `security-suite.py` | Select cases/modes and check QEMU outcomes against stage, cause and fault PC |
| `run-security.sh` | Run the suite under the shared QEMU lock |

## Build and run

Follow the [main build setup](../README.md#build-and-run) to prepare the native
recorder sources and select the installed platform. `replay/build.sh` builds
both the replay and security executables. Native executables cannot enforce
capability invalidation; security verdicts come from QEMU.

```bash
source capstone/tests/capstone-test-env.sh
export FFPOOL_WORK="${CAPSTONE_TMP_ROOT}/ffmpeg-buffer-pool"
port=capstone/ports/ffmpeg-buffer-pool
bash "$port/replay/build.sh" capstone

# Default matrix: cases 0–11 in modes 0, 1 and 2 (36 outcomes).
bash "$port/security-tests/run-security.sh" "$FFPOOL_WORK/runs/security"

# Short pair: valid-reference control and stale read under Sublet.
bash "$port/security-tests/run-security.sh" \
    "$FFPOOL_WORK/runs/security-pair" --cases 0,1 --modes 2
```

Use a new output directory for each invocation. Results are written to
`verdicts.json` with per-case logs beneath it. The wrapper defaults to an
explicit enlarged `CAPSTONE_REV_NODES=1048576`; this is not a hardware capacity
claim. Select a Python interpreter with `pexpect` through `PYTHON` if needed.

## Cases

| ID | Probe |
|---|---|
| 0 | Valid shared references, callbacks and deferred pool close |
| 1–2 | Buffer read/write after return to its pool |
| 3–4 | Buffer read/write through an old reference after storage reuse |
| 5 | RefStruct read after return |
| 6 | RefStruct write through an old reference after reuse |
| 7 | RefStruct unref through an old reference after reuse |
| 8 | Nested child access after its parent's reset callback |
| 9 | Buffer read after backing allocation is freed |
| 10 | Access before the RefStruct payload into the metadata area |
| 11 | Buffer access one byte past the payload end |
| 12 | Retained stale reference after repeated reuse (explicit selection) |
| 13 | Valid reference after repeated reuse (explicit selection) |

Modes are `0` (bounds), `1` (bounds and backing lifetime), and `2` (also Sublet
pool-lease lifetime). A passing verdict means the expected outcome for that
mode: stale accesses can complete in modes without the relevant protection.
Mode 2 rejects case 7 with status 304 before modifying the current reference
count; stale-access probes require a capability fault. Bounds probes fault in
all modes. Cases 12 and 13 accept `--rounds` (default 70000) and are excluded
from the default matrix.
