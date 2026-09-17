# Earlier FFmpeg baselines

These runnable controls precede the combined AVBufferPool/AVRefStructPool
experiment. Start with [the main workflow](../README.md) for current work.
Both baselines use `../runtime/prepare.sh` for source verification, compiler
flags and freestanding support objects. Neither is a prerequisite for the
current recorder or replay build.

## Standalone buffer API

`standalone/` compiles unmodified upstream `buffer.c` with a small allocator
and semantic probe. The native build compares it with upstream libavutil and
checks a deliberately corrupted payload. The domain probe is unprotected;
its expected return is 42042. `atomic-probe.c` isolates the compiler's standard
atomics limitation; the normal component build selects FFmpeg's serial fallback.

```bash
source capstone/tests/capstone-test-env.sh
port=capstone/ports/ffmpeg-buffer-pool
bash "$port/baselines/standalone/build.sh" native
bash "$port/baselines/standalone/build.sh" capstone
bash "$port/baselines/standalone/run-qemu.sh"
```

## AVBufferPool-only recording and replay

`buffer-only/` records CREATE/GET/final RETURN/CLOSE for AVBufferPool and replays
them without Sublet. It has its own `replay-format.h`, instrument and comparison
tools. Do not pass these traces to the current combined replay.

```bash
source capstone/tests/capstone-test-env.sh
port=capstone/ports/ffmpeg-buffer-pool/baselines/buffer-only
bash "$port/build-workload.sh" stock
bash "$port/build-workload.sh" traced
bash "$port/build-replay.sh" native
bash "$port/build-replay.sh" capstone
run="${FFPOOL_WORK:-/tmp/capstone/ffmpeg-buffer-pool}/runs/buffer-only-example"
bash "$port/run-workload.sh" "$run" 1 160x120
bash "$port/check-replay.sh" "$run"
```

`check-replay.sh` regenerates commands, runs native and QEMU replay, checks an
invalid RETURN control, then compares and plots the reports. It requires
`pexpect`, NumPy and Matplotlib. Use the same installed platform overrides
documented in the main README. Existing build names (`native/`, `capstone/`,
`workload-stock/`, `workload-traced/`) remain unchanged.
