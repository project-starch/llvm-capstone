# FFmpeg pool replay on Capstone

Record AVBufferPool and AVRefStructPool activity in a native FFmpeg decoder,
then replay those pool operations in a serial Capstone domain. Start with
`record/` and `replay/`; earlier experiments are under `baselines/`.

## Layout

| Path | Responsibility |
|---|---|
| `record/` | Build the stock/traced decoder, instrument upstream FFmpeg and record a workload |
| `replay/` | Apply the lifetime port, build the replay engine and host loader, run under QEMU |
| `runtime/` | Shared build preparation, metadata allocator and payload lifetime hooks |
| `trace/` | Shared event format and observation logic used by recorder and replay |
| `tests/` | Pool lifetime probes and their QEMU verdicts |
| `tools/` | Extract commands, compare traces, summarize and plot results |
| [baselines/](baselines/README.md) | Standalone buffer API controls and the older buffer-only replay |

FFmpeg 9.0.1 sources, generated ports, binaries and logs stay outside this
repository, under `${FFPOOL_WORK:-/tmp/capstone/ffmpeg-buffer-pool}`. Build
preparation verifies the pinned archive and pristine buffer sources. Existing
build/output directory names are retained so saved traces remain usable.

## Build and run

Use Bash. For a source-only worktree, select the installed dependencies with
`CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`, `CAPSTONE_QEMU_BINARY` and
`FFPOOL_MUSL` before sourcing the environment. The MUSL directory must contain
prepared Capstone headers. Python is required for instrumentation, `pexpect`
for QEMU and NumPy/Matplotlib for optional plots. An environment freshness
warning is not a successful check; record compiler revisions and binary hashes.

```bash
source capstone/tests/capstone-test-env.sh
export FFPOOL_WORK="${CAPSTONE_TMP_ROOT}/ffmpeg-buffer-pool"
port=capstone/ports/ffmpeg-buffer-pool

# Download/verify FFmpeg and build both native decoders.
bash "$port/record/build.sh" stock
bash "$port/record/build.sh" traced

# Build the replay and security probes for both targets.
bash "$port/replay/build.sh" native
bash "$port/replay/build.sh" capstone

# Always use a new recording directory. Use 1 160x120 for a short smoke run.
run="$FFPOOL_WORK/runs/example"
bash "$port/record/run.sh" "$run" 300 1280x720
for mode in 0 1 2; do
    "$FFPOOL_WORK/combined-port-native/replay" \
        "$run/commands.bin" "$run/native-$mode.bin" "$mode"
    python3 "$port/tools/trace-tools.py" compare \
        "$run/recorded.bin" "$run/native-$mode.bin"
done

# Explicit enlarged emulator capacity; this is not the hardware capacity.
export CAPSTONE_REV_NODES=1048576
for mode in 0 1 2; do
    bash "$port/replay/run-qemu.sh" "$run/commands.bin" "$run/mode-$mode" "$mode"
    python3 "$port/tools/trace-tools.py" compare \
        "$run/recorded.bin" "$run/mode-$mode/capstone.bin"
done
bash "$port/tests/run-security.sh" "$FFPOOL_WORK/runs/security-example"
```

`record/build.sh` defaults to `traced`. `replay/build.sh` prepares its own
freestanding support objects; it does not require a baseline experiment build.
The shared environment supplies the QEMU lock. Keep emulator suites inside
that lock. The default security matrix is cases 0–11 in all three modes;
cases 12 and 13 are separate long-run probes requiring explicit selection.
Expected faults must match their stage, cause and PC.

## What is replayed

The full decoder runs natively. Component replay executes pool APIs and the
nested allocator effects of callbacks. Codec computation, callback payload
computation, all reference-count operations and application concurrency are
outside that scope. Component builds use FFmpeg's upstream serial atomics
fallback. The port separates RefStruct metadata from payload and checks release
authority before returning a lease.

| Replay mode | Protection boundary |
|---|---|
| `0` | Payload bounds, without temporal revocation |
| `1` | Payload bounds and backing-allocation lifetime |
| `2` | Payload bounds, backing lifetime and Sublet pool-lease lifetime |

All modes share the same port and allocator layout. Native replay checks
functional behavior; it cannot enforce capability invalidation.

## Branches and evidence

The code branch is `ffmpeg/1-buffer-pool` in `project-starch/llvm-capstone`.
[PR #44](https://github.com/project-starch/llvm-capstone/pull/44) is stacked on
`musl/1-gap-survey` ([PR #43](https://github.com/project-starch/llvm-capstone/pull/43)),
which includes the C48 compiler fix. Review the FFmpeg diff against MUSL until
that dependency lands in `dev`. Push the feature branch to GitHub `origin`;
record immutable commits in evidence manifests.

This source rebase does not switch the replay to MUSL's libc: it still uses
MUSL headers and the freestanding runtime under `runtime/`. Linking
`libc-capstone.a`, replacing the metadata allocator or changing startup/hostcalls
requires a separate integration and validation step. Rebase also does not
rebuild the installed compiler, QEMU or monitor.

The paper repository's `eval/ffmpeg` branch owns the
[study index](https://github.com/project-starch/nested-allocators-paper/blob/eval/ffmpeg/experiments/README.md)
and [archived evidence](https://github.com/project-starch/nested-allocators-paper/tree/eval/ffmpeg/experiments/results/P2/20260917-ffmpeg-pool-replay).
The [release](https://github.com/project-starch/llvm-capstone/releases/tag/ffmpeg-pool-replay-20260917)
pins the original code and numeric archive. Its paths refer to that original
commit, not this reorganized tree. Paper working branches go to GitHub;
Overleaf receives integrated manuscript changes through the paper's `main`.
Platform changes belong in their own repositories and must be published before
a parent revision depends on them.

The archived recording has 9,000 frames and 630,309 events: 27,000 buffer leases
and 108,001 RefStruct leases. Saved native and Capstone reports match every
event; the saved security matrix has 36 expected outcomes. These are exploratory
results, not completion of the paper's P2, S1, S2 or M1 protocols. The completed
Sublet replay used 1,048,576 QEMU nodes and reached 135,285 allocated nodes after
the default 65,536-node budget was exhausted. It does not establish sustainable
reclamation, hardware execution, FFmpeg throughput or a full application port.
The evidence manifest records platform provenance gaps. Keep full local captures
outside published source trees; archive completed evidence with hashes before
cleaning scratch builds.

## Previous paths

| Earlier entry point | Current entry point |
|---|---|
| `combined/build-workload.sh`, `combined/run-workload.sh` | `record/build.sh traced`, `record/run.sh` |
| `combined/build-replay.sh`, `combined/run-qemu.sh` | `replay/build.sh`, `replay/run-qemu.sh` |
| `combined/run-security.sh` | `tests/run-security.sh` |
| `combined/trace-tools.py`, `combined/plot.py` | `tools/trace-tools.py`, `tools/plot.py` |
| `build.sh`, `run-qemu.sh` | `baselines/standalone/build.sh`, `baselines/standalone/run-qemu.sh` |
| Root-level buffer-only recorder/replay scripts | `baselines/buffer-only/` |

For the stock decoder, use `record/build.sh stock`. The buffer-only baseline
has its own `build-workload.sh stock|traced`; its trace format is different.
