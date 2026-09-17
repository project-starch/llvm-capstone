# FFmpeg pool replay on Capstone

Start here for the FFmpeg allocator experiment. This directory owns the port,
recorders, replay engines and semantic probes. The paper repository owns study
protocols and archived evidence. FFmpeg 9.0.1 sources and generated builds stay
under `${FFPOOL_WORK:-/tmp/capstone/ffmpeg-buffer-pool}`.

## Scope and entry points

| Directory | Purpose |
|---|---|
| This directory | Upstream `buffer.c` bring-up and an unprotected AVBufferPool replay |
| `combined/` | AVBufferPool and AVRefStructPool recording, replay and lifetime port |
| [Paper experiment index](https://github.com/project-starch/nested-allocators-paper/blob/eval/ffmpeg/experiments/README.md) | Study ownership and evidence bundles |
| [Archived FFmpeg audit](https://github.com/project-starch/nested-allocators-paper/tree/eval/ffmpeg/experiments/results/P2/20260917-ffmpeg-pool-replay) | Configuration limits, hashes, recorded outcomes and offline validation |

The complete FFmpeg decoder runs on the native host. A serial component replay
executes the recorded pool APIs and nested allocator effects of callbacks.
It does not replay codec computation, all reference-count operations, callback
payload computations or application concurrency. Both component targets use
FFmpeg's upstream serial atomics fallback. The port separates RefStruct metadata
from payload and checks release authority before returning a lease.

| Combined replay mode | Protection boundary |
|---|---|
| `0` | Payload bounds, without temporal revocation |
| `1` | Payload bounds and backing-allocation lifetime |
| `2` | Payload bounds, backing lifetime and Sublet pool-lease lifetime |

All three modes share the same port and allocator layout. Native replay checks
functional behavior only. It cannot enforce Capstone capability invalidation.

## Build and run

Use Bash and source `capstone/tests/capstone-test-env.sh` from the repository
root before building. Select installed dependencies explicitly when this is a
source-only worktree. Set `CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`,
`CAPSTONE_QEMU_BINARY` and `FFPOOL_MUSL` to the actual installations. The musl
directory must contain the prepared Capstone headers. Python is required for
the instruments, `pexpect` for QEMU and NumPy/Matplotlib for optional plots.
The environment's toolchain freshness warning is not a successful freshness
check. Record source revisions and binary hashes independently.

```bash
source capstone/tests/capstone-test-env.sh
export FFPOOL_WORK="${CAPSTONE_TMP_ROOT}/ffmpeg-buffer-pool"
port=capstone/ports/ffmpeg-buffer-pool

# Download and verify FFmpeg, then check the isolated buffer API.
bash "$port/build.sh" native
bash "$port/build.sh" capstone

# Build native recorders and the combined component port.
bash "$port/build-workload.sh" stock
bash "$port/combined/build-workload.sh"
bash "$port/combined/build-replay.sh" native
bash "$port/combined/build-replay.sh" capstone

# Use a new output directory for each recording.
run="$FFPOOL_WORK/runs/example"
bash "$port/combined/run-workload.sh" "$run" 300 1280x720
"$FFPOOL_WORK/combined-port-native/replay" "$run/commands.bin" "$run/native.bin" 2
python3 "$port/combined/trace-tools.py" compare "$run/recorded.bin" "$run/native.bin"

# This is an explicit enlarged emulator capacity, not the hardware capacity.
export CAPSTONE_REV_NODES=1048576
for mode in 0 1 2; do
    bash "$port/combined/run-qemu.sh" "$run/commands.bin" "$run/mode-$mode" "$mode"
    python3 "$port/combined/trace-tools.py" compare \
        "$run/recorded.bin" "$run/mode-$mode/capstone.bin"
done
bash "$port/combined/run-security.sh" "$FFPOOL_WORK/runs/security-example"
```

The shared environment supplies the QEMU lock. Do not run emulator suites
outside that lock. The security suite's default matrix is cases 0 through 11
in all three modes. Cases 12 and 13 are separate long-run probes and require
explicit selection. An expected fault has to match its stage, cause and PC.

## Recorded baseline and limits

The archived 2026-09-17 combined recording has 9,000 decoded frames, 630,309
events, 27,000 buffer leases and 108,001 RefStruct leases. Native and all three
Capstone replay reports match every recorded event. The saved semantic matrix
contains 36 of 36 expected outcomes. These are single-run exploratory results,
not completion of the paper's P2, S1, S2 or M1 protocols.

An earlier large Sublet replay exhausted the default 65,536-node QEMU budget.
The completed replay used 1,048,576 nodes and reached 135,285 allocated nodes.
It does not establish sustainable node reclamation. No hardware execution,
FFmpeg throughput or full application port is claimed. The evidence manifest
separates recorded build identities from dependency checkouts observed later.

## Ownership and publication

Development uses the `ffmpeg/1-buffer-pool` branch of `project-starch/llvm-capstone`,
reviewed against `dev`. Keep the original bring-up entry points stable.
Commit code and its documentation together after the repository checks.
Publish the feature branch to `origin` before relying on it from an evidence
manifest. Record immutable code commits in result bundles.

The paper branch is `eval/ffmpeg` in `project-starch/nested-allocators-paper`.
It links the existing W2 survey to the P2 exploratory replay evidence without
changing manuscript claims. Working branches go to GitHub `origin`. Overleaf
receives integrated manuscript changes through the paper's `main` workflow.
QEMU, monitor and Buildroot changes belong in their own repositories and must
be published there before a parent revision depends on them.

Temporary source trees and build products are disposable. Completed evidence
is archived with hashes before scratch cleanup. Full local captures may carry
machine identities and are kept outside published source trees. The portable
bundle identifies excluded captures and labels any extracted observations.
