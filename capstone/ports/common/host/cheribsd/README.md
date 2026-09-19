# CheriBSD allocator ports

Four extracted allocator libraries share one CHERI-RISC-V purecap toolchain
and one QEMU runner. Each component builds a replay and a small program that
links the allocator directly, without a trace driver.

| Component | CMake library target | Standalone example |
|---|---|---|
| [FFmpeg](../../../ffmpeg/buffer-pool/README.md) | `FFmpeg::BufferPool` | [pool.c](../../../ffmpeg/buffer-pool/examples/pool.c) |
| [PostgreSQL](../../../postgres/memory-contexts/README.md) | `PostgreSQL::MemoryContexts` | [contexts.c](../../../postgres/memory-contexts/examples/contexts.c) |
| [CPython](../../../cpython/pymalloc/README.md) | `CPython::Pymalloc` | [pymalloc.c](../../../cpython/pymalloc/examples/pymalloc.c) |
| [Whisper](../../../whisper/ggml-context/README.md) | `Whisper::GgmlContext` | [context.c](../../../whisper/ggml-context/examples/context.c) |

## Build and run one component

Use an installed matching SDK, purecap rootfs and raw disk image. Host tools:
CMake 3.25+, Ninja, Python 3.11.4+, `pexpect`, OpenSSH and the source-preparation
tools used by the existing native ports. Upstream archives remain pinned by
each component's `upstream.json`; downloads and builds stay outside Git.
These commands run from the repository root:

```sh
source capstone/tests/capstone-test-env.sh
export CHERI_SDK=/path/to/cheri/output/sdk
export CHERI_SYSROOT=/path/to/cheri/output/rootfs-riscv64-purecap
CHERI_IMAGE=/path/to/cheri/output/cheribsd-riscv64-purecap.img
PORT=capstone/ports/cpython/pymalloc
BUILD=/tmp/capstone/cheribsd/cpython

bash "$PORT/host/cheribsd/build.sh" "$BUILD"
bash "$PORT/host/cheribsd/run.sh" "$BUILD" /tmp/capstone/cpython-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image "$CHERI_IMAGE"
```

The same two script names exist in all four components. Alternatively, from
any component directory use `cmake --preset cheribsd` and
`cmake --build --preset cheribsd`. Cross binaries run through the QEMU runner,
not host CTest. The `cheribsd` preset disables host tests and recorder builds.
Use a fresh build directory when changing SDK or target ABI.

The build produces `bin/allocator-example`, `bin/replay`, the allocator's
static archive, and `bin/cheribsd-abi-probe`. PostgreSQL additionally provides
`client-{allocset,generation,slab,bump}-cheribsd`.

## Link your own program

Copy the corresponding standalone example and change its allocator calls:

```sh
bash "$PORT/host/cheribsd/build.sh" "$BUILD" --client /absolute/path/my-main.c
```

This builds `bin/allocator-client`. Inside the component build the operation is:

```cmake
add_executable(allocator-client "${PORT_CLIENT_SOURCE}")
target_link_libraries(allocator-client PRIVATE CPython::Pymalloc)
```

Replace the target with the table's target for the selected component. Public
include paths and required compile definitions propagate through that target.
The archives are component libraries, not replacements for libc malloc or
complete Python, PostgreSQL, FFmpeg or Whisper runtimes. They are not installed
`find_package` packages; the supported integration is the component's CMake
client target, with one component per build directory.

The examples show the required initialization and callbacks:

* FFmpeg supplies separate metadata/payload arenas, an event sink and serial
  lock hooks. It exercises references and deferred pool destruction.
* CPython supplies backing arenas and a failure callback, then calls
  `pym_malloc/calloc/realloc/free`. It checks a capability-bearing realloc.
* PostgreSQL creates a root context and uses the ordinary context APIs.
  The library includes the minimal backend compatibility and printf support.
* ggml supplies backing storage and a failure callback, then exercises owned
  and borrowed contexts, reset and the extracted object-allocation API.

These extracted runtimes are single-threaded and initialized once per process.
Keep backing storage alive for the entire client lifetime.

## Run several components or a replay

```sh
python3 capstone/ports/common/host/cheribsd/run.py /tmp/capstone/all-ports-run-1 \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" --image "$CHERI_IMAGE" \
  --build ffmpeg=/tmp/capstone/cheribsd/ffmpeg \
  --build postgres=/tmp/capstone/cheribsd/postgres \
  --build cpython=/tmp/capstone/cheribsd/cpython \
  --build whisper=/tmp/capstone/cheribsd/whisper
```

The runner boots a disposable snapshot, verifies 16-byte purecap pointers and
the requested libc revocation policy, and checks that a paired out-of-bounds
load terminates with SIGPROT. Every allocator example runs in a separate guest
process. Completion requires exit status zero and the exact success line.
The base disk image remains unchanged. A shared QEMU lock serializes runs;
SSH binds only to loopback. Override the forwarded port with `--port`.

For custom programs and recordings, add `--cases /path/to/cases.json`:

```json
[
  {
    "name": "my-client",
    "program": "/tmp/capstone/cheribsd/cpython/bin/allocator-client",
    "expect": "MY_CLIENT PASS"
  },
  {
    "name": "pymalloc-recording",
    "program": "/tmp/capstone/cheribsd/cpython/bin/replay",
    "args": ["input.bin", "output.bin"],
    "inputs": {"input.bin": "/path/to/recording/trace.bin"},
    "outputs": ["output.bin"],
    "expect_regex": "PYM completed=[0-9]+ alloc=[0-9]+ free=[0-9]+ realloc=[0-9]+ arenas=[0-9]+ released=[0-9]+"
  }
]
```

Without `--build`, supply `--abi-probe BUILD/bin/cheribsd-abi-probe`.
Patterns match a complete stdout line. A successful process is only a smoke
check: validate replay outputs against expected event counts, payload checks
and the component's native recording before publishing a workload result.
The input formats remain allocator-specific and use the shared trace readers.
PostgreSQL's CheriBSD entry reports logical counts and payload checks; it does
not demand x86 backing counts after changing size classes and pointer layouts.

`summary.json` records all expected cases, outcomes, input/output hashes and
platform/binary fingerprints. Failures stop the run and remain in that output
directory. A new attempt needs a new directory. Raw directories also contain
an ephemeral SSH private key and guest banners: keep them outside Git.

## Protection scope

| Target | Scope of this integration |
|---|---|
| FFmpeg CheriBSD | Existing bounded pool payload pointers; no per-return temporal invalidation by default |
| PostgreSQL CheriBSD | Capability-compatible manager with 16-byte chunk/free-list layouts; ordinary libc backing allocation |
| CPython CheriBSD | Capability-compatible pymalloc with retained arena/pool authority; inner pool frees remain ordinary allocator operations |
| ggml CheriBSD | Capability-compatible context extraction and backing ownership; reset does not revoke old aliases |

Running on CheriBSD does not automatically make inner frees temporally safe.
The default suite explicitly disables libc revocation in test processes and verifies that state;
`--runtime-revocation on` checks a separate installed runtime configuration.
That switch does not add inner-lifetime hooks.
The guest's system default is otherwise preserved. Optional
`--disable-default-revocation` sets and verifies
`security.cheri.runtime_revocation_default=0` before starting SSH, so transport
and helper processes also use that default. The report records this separate
setting; test programs still receive their explicit `--runtime-revocation`
policy. The console is continuously drained during SSH operations.
FFmpeg's existing optional PICASSO lease adapter remains available through its
[specialized collector](../../../ffmpeg/buffer-pool/host/cheribsd/README.md).
The experimental [PoisonCap workflow](../../../ffmpeg/buffer-pool/host/cheribsd/poisoncap/README.md)
uses its own reconstructed platform and explicit per-lease hooks.
It has its own mode and measurements; the generic examples use the default
CheriBSD adapter. No cross-system security or performance equivalence is
implied by a successful build or example.
