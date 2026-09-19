# FFmpeg on CheriBSD

The [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md) provides
the common toolchain, build/run scripts and explicit ABI/bounds controls.
`bash host/cheribsd/build.sh BUILD` builds the `FFmpeg::BufferPool` static
library, [direct-link example](../../examples/pool.c), and existing replay.
`bash host/cheribsd/run.sh BUILD OUTPUT --sdk SDK --rootfs ROOTFS --image IMAGE`
runs the example. Add `--client /absolute/path/main.c` to the build script
to link your own client with the same CMake target.

The specialized spatial measurement workflow follows.

## CHERI arena replay in QEMU

This target runs the same patched FFmpeg AVBufferPool/AVRefStructPool sources,
replay engine and native recordings as the Capstone measurement. It adds a
CHERI purecap spatial backend: every issued payload pointer has bounds checked
to cover the requested payload without extending beyond its backing block.
It refuses Capstone modes 1 and 2; running on a temporal-capable OS does not
turn these arena leases into temporally protected allocations.

## Build

Keep SDKs, upstream archives, builds and raw results outside the checkout.
Use an installed matching CHERI-RISC-V SDK, CheriBSD purecap rootfs and disk
image. The target uses the RISC-V `l64pc128d` purecap ABI.
No kernel, runtime or emulator sources are modified by this target.

```sh
source capstone/tests/capstone-test-env.sh
PORT="$PWD/capstone/ports/ffmpeg/buffer-pool"
cmake -S "$PORT" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PORT/cmake/toolchains/cheribsd.cmake" \
  -DCHERI_SDK="$SDK" -DCHERI_SYSROOT="$ROOTFS" \
  -DFFPOOL_ARCHIVE="$ARCHIVE" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$BUILD" -j 2
python3 "$PORT/host/cheribsd/measure.py" "$NATIVE_CAMPAIGN" "$RESULTS" \
  --sdk "$SDK" --rootfs "$ROOTFS" --image "$IMAGE" --build "$BUILD"
```

The host Python needs `pexpect`; file transfers use OpenSSH. The runner boots a
fresh single-user snapshot for each workload/repetition, creates a dedicated
SSH key in the external result directory and installs only its public half
in the disposable guest. It binds the forwarded SSH port to loopback. The base
image is read through QEMU's snapshot mode. Output directories must be new;
failures are retained and stop the campaign. Raw directories contain a private
SSH key and guest banners, and must not be committed.

The default is three repetitions of each source recording, with libc heap
revocation explicitly disabled in the replay process. `--runtime-revocation on`
checks the surrounding libc allocator policy; it **does not revoke an
individual arena suballocation**.
The replay reports its actual `malloc_revoke_enabled()` state, which must agree
with the requested policy before a result is accepted.

## What is measured

- Complete observed event sequences are compared to the native recording.
  Input commands contain no forced backing allocation results.
- Requested live and pool-retained payload, reuse gaps and callbacks are
  measured exactly as in the Capstone campaign.
- Payload and metadata arena carving watermarks include this target's actual
  C layouts. CHERI payload carving includes extra alignment and padding needed
  by compressed bounds. Both metadata and payload arenas retain their existing
  16 MiB and 64 MiB budgets; input/output buffers are each 128 MiB.
- Header extension words record the number of pointers issued (including
  trusted callback access), accumulated bounds slack over those issues,
  maximum bounds slack of an issued pointer, and `sizeof(void *)`. Accumulated
  slack is work-weighted accounting, not simultaneous wasted memory.
- Companion controls cover valid references and callbacks, stale buffer reuse,
  stale refstruct return, refstruct underflow and buffer overflow. In this
  spatial backend the valid and stale lifetime cases complete; the two bounds
  cases must reach the setup marker and then terminate with SIGPROT.

These are allocator memory-behavior measurements, not QEMU wall-clock speedups,
cache/DRAM results, full protected decoding or total process memory overhead.
In particular, CHERI's bounds rounding exposes a small amount of padding for
some sizes; the adjacent backing block must remain outside the issued bounds.

## Temporal comparison boundary

A custom arena bypasses libc `free()` on pool return. Per-return temporal
protection needs an explicit adapter such as the [PoisonCap backend](poisoncap/README.md).
Ancestor/child authority and finite-metadata reclamation require their own
controls; successful lease tests do not establish them.

## Export and plots

The export rechecks source-recording hashes, accepted binary hashes, every
observed event and repetition identity. It accepts only complete campaigns.
The compact export contains no raw guest banners, private keys or host paths.

```sh
python3 "$PORT/host/cheribsd/export.py" "$NATIVE_CAMPAIGN" "$EXPORT" "$RESULTS"
python3 "$PORT/host/cheribsd/plot.py" "$EXPORT/measurements.json" "$PLOTS"
# Add --language de for German labels.
```

The payload chart compares live-plus-idle requested bytes. The separate arena
chart shows the additional carved bytes in the CHERI spatial arm relative to
Capstone Sublet. It must not be captioned as total CHERI protection overhead.

The default CHERI guest has 2 GiB RAM and one CPU. The previous Capstone guest
has 8 GiB RAM and one CPU. All arenas have the same reservations, but these are
not matched OS-memory-pressure experiments. Code size, static harness storage,
libc and kernel storage need their own accounting; none belongs implicitly in
a ratio computed from payload carving alone.

For an independent inventory of allocated ELF sections, including static
bookkeeping excluded from arena watermarks:

```sh
python3 "$PORT/host/cheribsd/elf-storage.py" "$ELF_JSON" \
  capstone="$DOMAIN_BUILD/bin/replay.dom" cheri-spatial="$BUILD/bin/replay"
```

## Experimental PoisonCap integration

The separate [PoisonCap workflow](poisoncap/README.md) reconstructs the paper
platform and adds an explicitly selected pool-lifetime adapter and control suite.
