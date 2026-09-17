# FFmpeg pool replay on Capstone

Record AVBufferPool and AVRefStructPool activity in a native FFmpeg decoder,
then replay those pool operations in a serial Capstone domain. Start with
`record/` and `replay/`.

## Native, Capstone and Sublet

The build target (`native` or `capstone`) and replay protection mode (`0`, `1`,
or `2`) are separate choices. **Sublet pool-lease protection is Capstone mode 2.**

| Variant | Where it executes | Allocator / protection |
|---|---|---|
| Stock native FFmpeg | Development machine | Full upstream decoder and native allocators |
| Instrumented native FFmpeg | Development machine | Native decoder/allocators with pool recording hooks |
| Native component replay | Development machine | Our ported pools and allocator layout; functional control with ordinary pointers |
| Capstone replay, mode `0` | Capstone domain in QEMU | Payload bounds; no temporal revocation |
| Capstone replay, mode `1` | Capstone domain in QEMU | Bounds and revocation when backing storage is freed |
| Capstone + Sublet replay, mode `2` | Capstone domain in QEMU | Also revoke each pool lease on return, before reissuing the storage |

The native replay accepts all three mode numbers for functional comparison;
none gives native pointers Capstone bounds or revocation. The three Capstone
modes use one domain binary and the same port/layout, selected at runtime.
The full decoder runs only natively in this experiment.

## Layout and execution locations

| Path | Responsibility | Execution location |
|---|---|---|
| `record/` | Build stock/traced FFmpeg and record pool operations | Development machine |
| [replay/](replay/README.md) | Shared replay engine, Linux loader and build/run scripts | Engine: native or Capstone domain; loader: QEMU guest Linux; scripts: development machine |
| [runtime/](runtime/README.md) | Metadata allocator and native/Capstone/Sublet payload handling | Compiled into the replay or security executable; `prepare.sh` runs on the development machine |
| `trace/` | Event format and observation logic | Compiled into the native recorder and both replay targets |
| [security-tests/](security-tests/README.md) | Bounds, stale-reference and invalid-release probes | C probes: Capstone domain for security verdicts; suite/runner: development machine |
| `tools/` | Compare, summarize and plot traces | Development machine |

Here, the Linux **host loader** runs inside the QEMU guest and calls the
Capstone domain. It is distinct from the development machine that starts QEMU.

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
bash "$port/security-tests/run-security.sh" "$FFPOOL_WORK/runs/security-example"
```

`record/build.sh` defaults to `traced`. `replay/build.sh` prepares its own
freestanding support objects.
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

See [Native, Capstone and Sublet](#native-capstone-and-sublet) for the
execution targets and protection modes.

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
| `combined/run-security.sh` | `security-tests/run-security.sh` |
| `combined/trace-tools.py`, `combined/plot.py` | `tools/trace-tools.py`, `tools/plot.py` |

For the stock decoder, use `record/build.sh stock`.
