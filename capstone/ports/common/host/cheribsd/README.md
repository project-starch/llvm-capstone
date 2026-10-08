# CheriBSD allocator ports

Six extracted allocator libraries share one CHERI-RISC-V purecap toolchain
and one QEMU runner. Each component builds a replay and a small program that
links the allocator directly, without a trace driver.

| Component | CMake library target | Standalone example |
|---|---|---|
| [FFmpeg](../../../ffmpeg/buffer-pool/README.md) | `FFmpeg::BufferPool` | [pool.c](../../../ffmpeg/buffer-pool/examples/pool.c) |
| [PostgreSQL](../../../postgres/memory-contexts/README.md) | `PostgreSQL::MemoryContexts` | [contexts.c](../../../postgres/memory-contexts/examples/contexts.c) |
| [CPython](../../../cpython/pymalloc/README.md) | `CPython::Pymalloc` | [pymalloc.c](../../../cpython/pymalloc/examples/pymalloc.c) |
| [Whisper](../../../whisper/ggml-context/README.md) | `Whisper::GgmlContext` | [context.c](../../../whisper/ggml-context/examples/context.c) |
| [Wireshark](../../../wireshark/wmem/README.md) | `Wireshark::Wmem` | [wmem.c](../../../wireshark/wmem/examples/wmem.c) |
| [APR](../../../apr/pools/README.md) | `APR::Pools` | [pools.c](../../../apr/pools/examples/pools.c) |
| [memcached](../../../memcached/allocators/README.md) | `Memcached::Allocators` | [allocators.c](../../../memcached/allocators/examples/allocators.c) |

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

The same two script names exist in all six components. Alternatively, from
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
Patterns match a complete stdout line. `exit` selects success (0), an explicit
application rejection (1), or CheriBSD SIGPROT (162). For a rejection, use an
exact error marker and `also_expect` for the setup marker; every additional
marker must appear as a complete stdout line. An unrelated failure is not a
successful rejection control. A successful process is only a smoke
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
| PostgreSQL CheriBSD | Capability-compatible manager with 16-byte chunk/free-list layouts; ordinary libc blocks |
| CPython CheriBSD | Capability-compatible pymalloc with retained arena/pool authority; frees inside pymalloc's pools remain ordinary allocator operations |
| ggml CheriBSD | Capability-compatible context extraction and backing ownership; reset does not revoke old aliases |
| APR CheriBSD | Nodes from the platform's own malloc, as upstream; APR reuses a destroyed pool's node from its own free list without ever calling free(), so libc revocation is never asked |
| memcached CheriBSD | Slab pages and cache objects from the platform's own malloc, as upstream; a freed chunk goes on its class's list and a freed object on cache.c's STAILQ without free(), so libc revocation is never asked |

Running on CheriBSD does not automatically make invisible releases temporally safe.
The default suite explicitly disables libc revocation in test processes and verifies that state;
`--runtime-revocation on` checks a separate installed runtime configuration.
That switch does not add inner-lifetime hooks.
The guest's system default is otherwise preserved. Optional
`--disable-default-revocation` sets and verifies
`security.cheri.runtime_revocation_default=0` before starting SSH, so transport
and helper processes also use that default. The report records this separate
setting; test programs still receive their explicit `--runtime-revocation`
policy. The console is continuously drained during SSH operations.
The experimental PoisonCap workflows for
[FFmpeg](../../../ffmpeg/buffer-pool/host/cheribsd/poisoncap/README.md),
[CPython pymalloc](../../../cpython/pymalloc/host/cheribsd/poisoncap/README.md)
and [memcached](../../../memcached/allocators/host/cheribsd/poisoncap/README.md)
reuse a reconstructed platform and add explicit per-lease hooks.
They have their own modes and measurements; the generic examples use the default
CheriBSD adapter. No cross-system security or performance equivalence is
implied by a successful build or example.

## `quarantine-probe.c`: what a CheriBSD miss means

A completion on this platform has two possible causes and the verdict cannot
tell them apart: the mechanism could not see the defect, or the asynchronous
sweep had not run yet. `security.cheri.runtime_revocation_every_free_default`
is 0 in this guest, so a sweep runs only once a quarantine threshold is crossed,
and a short case that frees a handful of objects may never cross one.

The probe answers it without running a sweep. The kernel exposes a shadow
bitmap, one bit per 16-byte granule, set while that granule is quarantined;
reading a bit is O(1), so the wrapped allocator can check on every call and
report at exit:

    QUARANTINE shadow=<...> mallocs=N frees=N quarantined_after_free=N
               reused_while_quarantined=N

`quarantined_after_free` greater than zero says the quarantine is live during
the case. `reused_while_quarantined` is the window an async sweep leaves open:
memory handed out again while still quarantined. A stale pointer into memory
that never reaches `free()` at all shows up in neither -- which is the
structural miss, and it is what Perl's eleven turned out to be.

Use it by giving a case an `env` of `{"LD_PRELOAD": "./quarantine-probe.so"}`
and shipping the shared object beside the program; `run.py` passes a case's own
environment through in front of the revocation policy.

Forcing a sweep instead (`_RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1`) is NOT a
substitute and was withdrawn as evidence: it makes every Perl case fault,
including an in-bounds read of a live object that no bounds and no revocation
may legitimately catch, so it cannot distinguish a catch from an artefact.
