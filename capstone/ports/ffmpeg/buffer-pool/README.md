# FFmpeg buffer-pool replay

This component records native FFmpeg's AVBufferPool and AVRefStructPool
operations and replays their allocator behavior in a Capstone domain. The
decoder runs natively; this is not a capability-domain FFmpeg decoder.

[`upstream.json`](upstream.json) pins FFmpeg 9.0.1 by archive checksum.
[`patches/`](patches/) separates the upstream changes from the replay and
authority adapters. The [shared layout](../../README.md) describes `src/`,
`host/`, tests and result ownership.

## Build and run

Source `capstone/tests/capstone-test-env.sh` from the repository root, then
enter this directory. Native tools include CMake 3.25+, Ninja, Python 3.11.4+
and the native FFmpeg build prerequisites. Cross builds additionally need
`CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`, `CAPSTONE_QEMU_BINARY`
and prepared Capstone musl headers (`FFPOOL_MUSL` or `PORT_MUSL_ROOT`).

```sh
cmake --preset native
cmake --build --preset native
ctest --preset native
cmake --preset capstone-domain
cmake --build --preset capstone-domain
cmake --preset linux-guest
cmake --build --preset linux-guest
ctest --preset qemu-replay
ctest --preset qemu-security
```

Native tests first produce a recording whose decoded frame hashes agree
between stock and instrumented FFmpeg. QEMU replay tests consume that
recording. Domain and Linux builds must both exist before the QEMU tests run.
Presets use `/tmp/capstone/ffmpeg-buffer-pool/build/`; custom builds must also
set the matching `FFPOOL_*_BUILD_DIR` runner paths.

For a separate small recording, choose a new external output directory:

```sh
bash host/record.sh /tmp/capstone/ffmpeg-recording 2 320x240
python3 host/run-qemu.py /tmp/capstone/ffmpeg-recording/commands.bin \
  /tmp/capstone/ffmpeg-replay --protection sublet
```

The runner offers `spatial`, `backing` and `sublet` modes, stages the exact
inputs and binaries, and retains each attempt. This port needs the capability
atomic compiler change already in LLVM `dev` and the matching emulator checks
in [QEMU PR #5](https://github.com/project-starch/capstone-qemu/pull/5).

## Evidence and limits

The [paired measurement campaign](results/measurements/20260919-replay/README.md)
checks three fresh native recordings in spatial and Sublet QEMU, three
repetitions per arm. All accepted event sequences match native observations;
carving watermarks, requested-payload series and primitive counts are reported
separately from fixed reservations and unmeasured node/tag costs. Failed attempts
remain documented. Use `host/memory/measure.py` for a new campaign,
`export-measurements.py` for checked JSON/CSV and `plot-measurements.py` for
event-indexed figures. The [measurement plan](../../../docs/plans/replay-memory-measurements.md)
defines the scope and remaining ledger, capacity and turnover work.

[`results/archive/20260917/`](results/archive/20260917/) indexes the exploratory
numeric evidence and external raw artifacts. It includes failed and incomplete
attempts; it is not a claim that every long-run configuration passed. Memory
analysis lives under `host/memory/`. Historical evidence keeps its recorded
tool identities and node budget.

Replay checks allocator requests and synthetic payloads. It does not establish
full decoder protection, concurrent application safety, FPGA behavior or
application timing overhead. See the [integration plan](../../../docs/plans/port-stack-integration.md)
before combining pending runtime changes.
