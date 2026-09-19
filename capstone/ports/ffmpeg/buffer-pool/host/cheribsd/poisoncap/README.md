# FFmpeg pools on PoisonCap

Experimental integration of the extracted AVBufferPool and AVRefStructPool
with the published PoisonCap platform. The implementation is on
`ports/9-poisoncap-ffmpeg`; the prerequisite CheriBSD library port is on
`ports/8-cheribsd-allocators`. A successful build alone is not a protected
allocator result. The platform and lease controls below are acceptance gates.

The [first pilot](../../../results/measurements/20260919-poisoncap-pilot/README.md)
passes all 29 full-suite processes with the guest's automatic libc-revocation
default disabled. Explicit per-lease revocation remains active. Three
2,379-event replays produce identical output. Preserving the guest default
instead encounters a captured kernel VM-locking panic in longer suites.

## Reconstruct the platform

`platform.json` pins the paper artifact, complete upstream source bases and
the missing compressed-capability dependency. `prepare.py` overlays published
files onto those bases, preserving omitted upstream dependencies. It also
installs the artifact's separately supplied version-aware capability header
into the dependency directory. Python bytecode caches are excluded.

Fetched sources, SDKs, images, reports and logs stay outside this repository:

```sh
source capstone/tests/capstone-test-env.sh
PORT=capstone/ports/ffmpeg/buffer-pool
PC="$PORT/host/cheribsd/poisoncap"
WORK=/tmp/capstone/poisoncap-work
python3 "$PC/prepare.py" "$WORK"
python3 "$PC/prepare.py" "$WORK" --verify

# Host dependencies include Clang/LLD 18, CMake, Ninja and CheriBSD build tools.
# Use the cheribuild revision recorded in platform.json.
export CHERIBUILD=/path/to/cheribuild/cheribuild.py
export CHERI_BOOT_FIRMWARE=/path/to/sdk/share/qemu/bbl-riscv64cheri-virt-fw_jump.bin
bash "$PC/platform.sh" "$WORK" llvm
bash "$PC/platform.sh" "$WORK" qemu
bash "$PC/platform.sh" "$WORK" cheribsd
bash "$PC/platform.sh" "$WORK" image
```

The CheriBSD build keeps Kerberos enabled: this revision's libc includes a
GSSAPI header even in an otherwise minimal build. The port enables GNU C for
the published revocation header and explicitly selects LLD when linking.

Existing artifact archives can be supplied as `--artifact` and `--cap-library`.
The build stages do not alter existing standard CheriBSD/PICASSO installations.
The older emulator executable is named `qemu-system-riscv64xcheri`; the SDK
provides the name expected by the shared runner. Firmware is an explicit
external input and must be included in the run's platform fingerprints.

## Build and validate the allocator

```sh
export CHERI_SDK="$WORK/sdk"
export CHERI_SYSROOT="$WORK/output/rootfs-riscv64-purecap"
BUILD="$WORK/build/ffmpeg"
bash "$PC/build.sh" "$BUILD"

# Optional external client, linked against FFmpeg::BufferPool:
bash "$PC/build.sh" "$BUILD" -DPORT_CLIENT_SOURCE=/absolute/path/my-program.c

python3 "$PC/run.py" "$BUILD" "$WORK/platform-1" --stage platform \
  --disable-default-revocation \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
  --image "$WORK/output/cheribsd-riscv64-purecap.img"
python3 "$PC/run.py" "$BUILD" "$WORK/pools-1" --stage pool \
  --disable-default-revocation \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
  --image "$WORK/output/cheribsd-riscv64-purecap.img"
```

The runner first checks the purecap ABI and an ordinary bounds fault. PoisonCap
controls then test live access, a poisoned read/write, a successful sweep,
new authority after reuse, and rejection of a retained old alias. Pool tests
pair mode 0 with mode 2 and include retained references, an unrelated live
buffer, deferred destruction, RefStruct callback state and repeated reuse.
Mode 0 is the spatial control; mode 2 enables poison-on-return protection.
The runner passes mode 2 explicitly for recordings. A direct library client
must initialize its arenas and call `ff2_set_mode(2)`, as the example does;
merely linking the library does not select temporal protection.

Use `--stage replay --recording COMMANDS --native-report NATIVE_OUTPUT` for
a recording. This includes the preceding controls and compares the resulting
observed event sequence with the native report. Output directories are unique;
failures are retained. Raw guest logs and ephemeral SSH keys must not be committed.
`--case NAME` selects a diagnostic subset and records that selection explicitly.
It still runs the ABI/bounds checks; the default full suite is unchanged.

Libc automatic revocation is explicitly disabled in the test programs.
The recommended `--disable-default-revocation` configuration also sets and
verifies `security.cheri.runtime_revocation_default=0` before starting SSH,
so infrastructure/helper processes use that default. Omit this flag to
reproduce the preserved-default configuration and its recorded kernel issue.
The flag changes the disposable guest policy, not the disk image or kernel.
The adapter invokes the poison-aware kernel revoker itself. This isolates inner
pool lifetime handling; it is not a whole-process temporal-safety configuration.
Initialisation-safety enforcement is outside this experiment's scope.

## Lifetime policy and additional storage

On the last return of a lease, the adapter snapshots the rounded payload and
poisons it. Before handing that block out again, it completes synchronous
revocation, detoxes the block and restores its contents. Other already-poisoned
blocks can share that sweep. Failed revocation stops execution before reuse.
Fresh application pointers are bounded and lose the poison/VM authority bits;
the trusted manager retains wider backing authority obtained from `mmap()`.
The direct-link [example](../../../examples/pool.c) shows this setup.

Snapshots preserve FFmpeg pool semantics: RefStruct's initialisation callback
runs once, and fields can survive across lease returns. Poisoning directly
overwrites those fields. A capability-preserving copy outside the poisoned
storage preserves persistent state while still allowing the revoker to clear
expired capabilities within it. Snapshots are retained per backing block for
this process-lifetime extraction, including currently unused backing records.
This implementation snapshots both pool types for simplicity. A policy that
preserves only the state required by each pool may reduce this cost; these
bytes are not a lower bound on PoisonCap's overhead.

`FF2_POISONCAP` reports sweeps, bytes poisoned/cleared, retained snapshot bytes
and copied bytes. These are adapter counters, not total process or kernel
memory. The binary report's four extension fields hold sweep count, poison
bytes, snapshot bytes and pointer size. Static bookkeeping and system-allocator
overhead need separate accounting in a memory comparison.

This first policy deliberately sweeps before immediate reuse. It is not the
paper's tuned quarantine policy and does not establish a performance ranking.
QEMU results support functional behaviour and operation counts, not hardware
cache, latency, bandwidth or energy conclusions.
