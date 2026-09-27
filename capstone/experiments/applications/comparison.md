# Capstone ports versus default CheriBSD

This campaign compares actual applications on Capstone/Sublet malloc with the
same workloads on default CheriBSD purecap. No allocator policy overrides,
forced drains, event recording or replay are used. The currently built matching
applications are mruby 4.0.0-rc2 and the configured FFmpeg 9.0.1 Matroska/MPEG-4
decoder. This is not a claim about every FFmpeg feature or every port.

The [2026-09-27 QEMU node-reuse follow-up](../../runtime/tests/application/results/20260927-node-reuse/README.md)
reruns the same Capstone application binaries at 65,536 nodes: all 27 attempts
pass, including the six mruby failures in the original campaign. The emulator
now collects retired identities under allocation pressure during a process.
Keep these runs separate from the original 66-attempt data and its larger-node
control. Default CheriBSD is unchanged and was not rerun for this QEMU fix.
The software sweep is included in this implementation; its cost cannot be used
as evidence of scan-free reclamation or a hardware performance advantage.

[Memory-behavior analysis](memory-behavior.md) relates the selected metrics to
the Cornucopia papers. The [extended paired results](results/20260927-reuse/README.md)
contain twelve workloads, 72 passing attempts, four figures and controls checking
that every recorded Capstone memory phase is unchanged without in-process sweeps.
They include a large-retained-graph counterexample and an invalid observer-limit
attempt; address reuse is not a proxy for total memory or hardware speed.

## Build and run

Source `capstone/tests/capstone-test-env.sh` before commands. Keep prepared
upstream sources, builds and raw results under `$CAPSTONE_TMP_ROOT`.

Capstone's existing `build.py` relinks prepared port objects with the shared SDK.
Add `--allocations` to count requested bytes and address reuse. The adapter works
with every application already supported by that script; measurements of nested
object allocators require separate accounting and are not inferred from malloc.

For CheriBSD, `comparison-build.py` builds either application from the port's
prepared sources. `--cc` is a configured driver executable using the matching
SDK's `cheribsd-riscv64-purecap.cfg`, not a shell command string:

```sh
python3 capstone/experiments/applications/comparison-build.py \
  --app ffmpeg --platform cheribsd --source "$PREPARED_FFMPEG" \
  --cc "$CONFIGURED_CC" --ar "$LLVM_AR" --ranlib "$LLVM_RANLIB" \
  --allocations --out "$EXPERIMENT_ROOT/cheribsd-ffmpeg"
```

Prepare FFmpeg with `ports/ffmpeg/app/host/prepare-source.sh` without pool hooks.
For mruby, use the prepared interpreter tree and `--app mruby`. Its build matches
the Capstone port's no-boxing representation, 16-byte pool alignment, switch
dispatch and selected gems. The builder copies sources and the Ruby build
configuration into its output directory, keeping generated lock files there.
Commands, source fingerprints, logs and executable hashes accompany each build.
Also fingerprint the underlying compiler, SDK libraries, kernel and emulator;
a driver hash alone does not identify them.

`cheribsd-run.py` reuses the common CheriBSD Guest implementation. One snapshot
boot serves the whole matrix; Capstone's `run.py` uses the persistent Linux VM.

```sh
python3 capstone/experiments/applications/cheribsd-run.py \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_ROOTFS" --disk "$CHERI_DISK" \
  --port 10437 --points "$EXPERIMENT_ROOT/points.json" \
  --out "$EXPERIMENT_ROOT/run" --repeat 3
```

The runner requires pexpect, SSH/SCP and the shared QEMU lock. Its default is
one hart and 8 GiB RAM. `--key` attaches to an already owned guest. Temporary
SSH keys must be excluded from exported artifacts. Stop the owned Capstone VM
before the CheriBSD campaign and restore it afterwards; do not disturb other
guests. Neither runner retries a failed application or reboots between points.

Each point specifies `id`, `application`, `arm`, `argv`, `environment`, exact
`expected_stdout`, `expected_phases`, and `allocations: true`. CheriBSD also
requires `files` mapping host files to guest paths and the expected default
`revocation` state. Keep `environment` empty: the runner rejects allocator-policy
overrides. Capstone supplies `image` separately from `argv`. Both sides use the
same workload bytes and full output oracle. Files copied into CheriBSD are
verified with SHA-256. Missing metrics, accounting errors, changed policies,
wrong output, signals and transport failures cannot become passing results.

## Measurements and their limits

* **Requested live/peak bytes:** sizes supplied by application and static-library
  calls to malloc/calloc/realloc and, on CheriBSD, aligned allocation APIs.
  Link wrapping does not observe dynamic libc's private allocations. Failed
  nonzero realloc preserves the old allocation; nested allocator calls count
  once. Unknown frees or a full observer table invalidate the comparison.
* **Address reuse:** exact starting addresses returned again after an observed
  free. An in-place realloc is a continuation, not reuse. Distance bins count
  intervening allocation API calls (including failed calls), with one meaning
  the next allocation call after the free. This is not elapsed time or a claim
  about partial overlaps. Integer addresses are retained; heap capabilities are
  not. The observer keeps aggregate counters, not event traces.
* **Occupied allocator storage:** Capstone reports occupied buddy blocks;
  CheriBSD reports jemalloc allocated/active/resident/mapped counters. Allocated
  includes retained allocations and quarantine. Active minus allocated describes
  unused space in active pages; it is not a complete external-fragmentation
  measurement. These ledgers are not interchangeable with requested live bytes.
* **Address span:** the range covering currently observed allocations. Gaps in
  that range do not establish physical-memory waste or external fragmentation.
* **Reservations and instrumentation:** the fixed address table is 1,572,864
  bytes on both targets, plus small counters and report stack space. It affects
  memory footprint and may affect allocator collection. It allocates no heap
  storage. These are instrumented memory experiments, not runtime-overhead
  measurements. Capstone's logical pools, driver grants, static allocator tables,
  application data reservation and kernel node capacity must remain visible.

For FFmpeg, vary frames per stream (30/150/600) and repeated independent streams
(1/4/16). Check every decoded frame hash against the native oracle. For mruby,
parse 32/128/512 records across eight batches, retain a graph of 256 or 4,096
records, and issue a fourfold burst in batch five. Validate the checksum and
retained graph. Keep failed points in the completion table. Any larger Capstone
node-capacity run is a separate configuration, not a replacement for failures.

Emulator wall time is diagnostic only. The current figures do not establish a
hardware speedup, total-RSS win, or a general fragmentation advantage. Exported
data and figures live on the paper repository's `eval/application-memory` branch
under `experiments/application-exploration/`; raw captures remain external.
