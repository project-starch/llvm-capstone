# whisper.cpp 1.9.4: ggml context allocator

This ports the original context allocator in `ggml/src/ggml.c` into native and
Capstone replays. Native Whisper can record its actual object-allocation stream;
the domain executes allocator calls with checked synthetic payloads, not speech
recognition, tensor kernels, or the separate backend tensor-data allocator.

## Ownership and lifetime boundary

`ggml_init` creates a descriptor and either owns a new buffer or borrows a caller's
buffer. `ggml_new_object` bump-allocates aligned headers and payloads. Objects are
not individually freed. `ggml_reset` starts again at the beginning of the buffer.
`ggml_free` releases the descriptor and releases the buffer only when it owns it.

This distinction matters in Whisper: its graph builders use state-owned metadata
buffers, destroy the temporary context descriptor, return the graph, and compute
with that graph afterward. Revocation on every `ggml_free` would reject valid
execution. The adapter therefore keeps borrowed object authority alive after
descriptor destruction. A control dereferences both an object and its embedded
capability after that destruction.

The port's buffer owner retains one revocation handle per backing buffer. Reset,
owned-buffer release, or exclusive rebind of that buffer for a new context ends
its allocation epoch. A borrowed descriptor's release does not. Descriptor
storage has its own lifetime and is revoked independently in Sublet mode.

**Exclusive rebind is an adapter contract**, not a claim that upstream
`ggml_init` itself frees previous objects. The replay treats starting another
context on the same recorded buffer as the owner's next graph epoch. Concurrent
contexts on the same buffer are rejected. A full application integration would
need explicit owner/graph-completion coordination before starting that epoch;
this allocator replay does not prove all inference accesses obey it.

Both modes run the same allocator and backing layout:

- `spatial`: nonempty object payload pointers are request-bounded, but epoch
  transitions do not revoke old aliases.
- `sublet`: additionally revokes buffer epochs and freed context descriptors.

The allocator internally retains buffer-wide authority for its headers and linked
list. Only the replay object API returns narrowed payload pointers. This is not
a claim of metadata isolation from callers given the public buffer accessor.
Zero-byte allocations retain their numeric address without memory authority,
because Capstone's SHRINK cannot encode an empty interval. Their layout matches
upstream, but attempting to dereference them must fault in both domain modes.

## Source and layout

The shared `../../common` support provides verified downloads, external build
guards, cross toolchains, run staging and serialized QEMU execution. The official
release archive is pinned by SHA256 in `upstream.json`; no source is vendored.

Three versioned patches separate the extraction boundary, buffer-epoch hooks,
and native recording. Each names its source, checksum, variants and prerequisites.
Preparation rejects checksum mismatches, reversed patches and fuzzy matching;
the previous prepared source survives a preparation failure.

`src/native/`, `src/capstone-domain/` and `src/linux-guest/` identify execution
environments. `src/allocators/sublet/` implements the authority operations;
`src/shared/` holds backing policy, interfaces and replay. `context-api.inc` is
included inside the upstream translation unit solely to expose its static
allocator; the allocator itself is not reimplemented in the adapter.

The extracted native reference applies only patch 1. The native port applies
patches 1 and 2. An optional full-library reference links the ordinary upstream
ggml library and calls `ggml_new_buffer`, which uses the same object allocator;
object kind is metadata only for allocation placement. These references compare
request checksums, payload contents, peak usage and a normalized layout checksum.

## Build

Source `capstone/tests/capstone-test-env.sh` from the repository root. Cross builds
use the prepared `CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`,
`CAPSTONE_QEMU_BINARY` and `PORT_MUSL_ROOT`; fresh worktrees need not initialize
their submodules when those external paths are supplied.

From this component directory:

```sh
cmake --preset native
cmake --build --preset native
ctest --preset native
cmake --preset capstone-domain
cmake --build --preset capstone-domain
cmake --preset linux-guest
cmake --build --preset linux-guest
```

Builds default to `/tmp/capstone/whisper-ggml-context/build/`. Override with `-B`
or an untracked `CMakeUserPresets.json`. Native recording is optional:

```sh
cmake --preset native -DWG_BUILD_RECORDER=ON
cmake --build --preset native
ctest --preset native
python3 host/record.py /tmp/whisper-recording \
  --native-build /tmp/capstone/whisper-ggml-context/build/native \
  --model /path/to/ggml-tiny.en.bin --audio /path/to/sample.wav --repeat 10
```

The model must match SHA256
`921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
The checked workload repeats the upstream shipped speech sample ten times, uses
tiny.en and one CPU thread, and requires identical nonempty stock/instrumented
transcripts. The recorded source differs from stock only in compiled recording
hooks; extraction/lifetime switches are disabled in both full native builds.
The recorder serializes hooks with a native mutex, fails on table overflow, and
records no tensor-data backend allocations. Only a successful capture and
transcript comparison promote `.partial` to `trace.bin`.

## Replay and verification

```sh
/tmp/capstone/whisper-ggml-context/build/native/bin/replay \
  /tmp/whisper-recording/trace.bin /tmp/whisper-native.bin
/tmp/capstone/whisper-ggml-context/build/native/bin/replay-full-reference \
  /tmp/whisper-recording/trace.bin /tmp/whisper-reference.bin
python3 host/run-qemu.py /tmp/whisper-recording/trace.bin /tmp/whisper-spatial --protection spatial
python3 host/run-qemu.py /tmp/whisper-recording/trace.bin /tmp/whisper-sublet --protection sublet
python3 security-tests/qemu/run.py /tmp/whisper-security
```

QEMU runners require `pexpect`, accept `--domain-build`/`--linux-build`, take the
shared lock, and retain each attempted run, its binary/input hashes, serial log
and verdict. Failed attempts are not automatically retried or counted as passes.
After inspecting a failed security attempt, `--resume` explicitly retries
unfinished cases while retaining each earlier verdict. Previously passing cases
are reused only when staged inputs, test binaries, compiler, QEMU and node budget
still match. Changed binaries require a new output directory.
Raw logs and transcripts stay outside the repository. Portable numeric results
live in `results/` with checksums.

Native tests cover borrowed/owned contexts, all object kinds, reset, reuse, zero
sizes, exhaustion, malformed traces, source integrity and patch ordering. The
paired security cases cover live borrowed graphs after descriptor free, reset,
exclusive buffer rebind, owned free, interior stale writes, bounds, descriptor
lifetime, 2,000 reset/reuse epochs, failed allocation preserving live objects,
and zero-byte dereference. A fault verdict requires its setup marker, expected
cause and exact access PC; live controls must finish normally.

## Replay geometry and limits

Native object headers are 32 bytes; capability headers are 48 bytes. The domain
adds 16 bytes per object in the largest recorded allocation epoch of each
context to its recorded capacity. This preserves spare capacity attributable to
object headers instead of causing artificial failures in exactly sized contexts.
This is explicit ABI accommodation: raw capacities, offsets and peak usage are
not byte-identical across ABIs. Payload sizes remain native recorded byte counts,
not rebuilt capability-ABI tensor structures. Replay does not reconstruct an
executable tensor graph.

The workload reserves four roughly 79 MiB buffers even though these contexts
issue no recorded objects. The replay retains these reservations. Its shared
payload region is 384 MiB, with 8 MiB auxiliary metadata and 16 MiB trace regions.
It supports 128 buffer identities, 128 simultaneous contexts and 32,767 live
replay objects. Backing regions are retained for same-identity reuse; growing one
carves a new region and retires its old authority in Sublet mode. This bounded
backing policy is not a libc allocator port, and repeated growth can exhaust it.

The event stream is little-endian: a 16-word header followed by six-word events
`op, context, buffer, size, type, argument`. All words are unsigned 64-bit.
Operations are INIT=1, ALLOC=2, RESET=3, FREE=4, END=5. INIT's argument indicates
ownership; ALLOC's argument is the synthetic fill byte. END's context field is
the remaining live-descriptor count, not a fabricated teardown. The header
records the native object-header size. Input lengths, identities and lifetimes
are checked before interpreting their operations.

These are allocator-component QEMU results. They do not establish full Whisper
protection, inference correctness on Capstone, concurrent graph safety, FPGA
behavior, memory overhead of the complete application, or timing overhead.
