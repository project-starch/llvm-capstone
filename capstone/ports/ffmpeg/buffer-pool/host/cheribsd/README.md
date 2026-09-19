# FFmpeg on CheriBSD

The [shared CheriBSD workflow](../../../../common/host/cheribsd/README.md) provides
the common toolchain, build/run scripts and explicit ABI/bounds controls.
`bash host/cheribsd/build.sh BUILD` builds the `FFmpeg::BufferPool` static
library, [direct-link example](../../examples/pool.c), and existing replay.
`bash host/cheribsd/run.sh BUILD OUTPUT --sdk SDK --rootfs ROOTFS --image IMAGE`
runs the example. Add `--client /absolute/path/main.c` to the build script
to link your own client with the same CMake target.

The specialized spatial/PICASSO measurement workflow follows.

## CHERI arena replay in QEMU

This target runs the same patched FFmpeg AVBufferPool/AVRefStructPool sources,
replay engine and native recordings as the Capstone measurement. It adds a
CHERI purecap spatial backend: every issued payload pointer has bounds checked
to cover the requested payload without extending beyond its backing block.
It refuses Capstone modes 1 and 2; running on a temporal-capable OS does not
turn these arena leases into temporally protected allocations.

An optional `-DFFPOOL_PICASSO=ON` build adds explicitly colored leases on the
installed PICASSO SDK. This build requires mode 2 and active libc revocation;
the default spatial build still refuses mode 2. The [temporal reuse experiment](../../../../../docs/plans/pool-temporal-reuse.md)
defines the comparison and its separate hierarchical-protection boundary.

The adapter maps the payload arena with `mmap` to retain trusted recoloring
authority. Each issued lease obtains one 64-byte libc token and copies its
color onto the bounded payload pointer. It removes `CHERI_PERM_SW_VMEM` before
returning that pointer. Returning/freeing a lease frees the token, invalidating
that color while retaining pool storage. This is an added trusted-pool adapter,
not a feature evaluated for nested allocators in the PICASSO paper. It does
not automatically couple child colors to a parent lifetime. Tokens are adapter
overhead, not an intrinsic lower bound on PICASSO memory cost.

Use `--lease-protection picasso --runtime-revocation on` with the collector.
`--churn-rounds 300000` adds valid/stale constant-address churn controls in
every first-workload repetition. The separate `2200000` extension crosses the
installed 21-bit color threshold and uses one repetition. The collector retains
its own source snapshot so later formatting does not change the recorded tool.

```sh
python3 "$PORT/host/cheribsd/temporal-summary.py" "$PICASSO_CAMPAIGN" \
  "$NATIVE_CAMPAIGN" "$EXPORT" "$CAPSTONE_RUN_1" "$CAPSTONE_RUN_2" "$CAPSTONE_RUN_3"
python3 "$PORT/host/cheribsd/plot-temporal.py" "$EXPORT/measurements.json" "$PLOTS"
# For the separate extension: --extension, with its one Capstone run directory.
```

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
is a separate control for the surrounding libc allocator, useful with an
installed PICASSO image; it **does not revoke an individual arena suballocation**.
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

A custom arena allocator bypasses the libc `free()` path on pool return. The
[official PICASSO artifact](https://github.com/coloredcapabilities/colored-artifact)
integrates colored allocation/release in the libc allocator. Enabling it for
our few outer arenas is therefore a useful integration control, but does not
match Sublet's per-return protection. The optional PICASSO lease adapter above
provides an explicit trusted-pool comparison and accounts for its token storage.
Ancestor/child authority and finite-metadata reclamation remain separate
comparison obligations; successful lease controls do not establish them.

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

Use `--heap-probe "$BUILD/bin/heap-policy-probe"` with the collector for a
separate libc allocation/free control. Its stale load is volatile and built
at the selected Debug optimization level. The live control must complete;
with the PICASSO runtime active, the post-`free()` load must reach its setup
marker and then SIGPROT. This is separate from the stale pool-return controls.

## Experimental PoisonCap integration

The separate [PoisonCap workflow](poisoncap/README.md) reconstructs the paper
platform and adds an explicitly selected pool-lifetime adapter and control suite.
