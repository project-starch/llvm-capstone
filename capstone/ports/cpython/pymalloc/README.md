# CPython 3.13.7 pymalloc

[CheriBSD build, link example and QEMU runner](host/cheribsd/README.md)
use the shared purecap toolchain. Each hosted build exposes a CMake allocator
library and `bin/allocator-example`; a custom main can be linked through
`PORT_CLIENT_SOURCE`. Protection scope is stated separately from build support.

The real allocator in `Objects/obmalloc.c`, extracted into native and Capstone
replays. This is an allocator component port with a native Python workload
recorder. It does not execute the interpreter, its collector, or extensions in
a capability domain. The free-threaded/mimalloc configuration is outside scope.

The experimental [PoisonCap adapter](host/cheribsd/poisoncap/README.md) adds
explicit per-block invalidation with synchronous sweeps and paired controls.

## Layout and source boundary

This component uses `../../common` for verified downloads, cross toolchains,
external builds, input staging and the shared QEMU lock. `src/native/` contains
the recorder and reference entry point; `src/capstone-domain/` the domain entry;
`src/linux-guest/` the staging loader; `src/allocators/sublet/` the authority
adapter. Workloads and generated upstream sources stay outside the repository.

`upstream.json` pins the official CPython 3.13.7 archive by SHA256. Ordered patches
separate interpreter extraction, capability provenance, and lifetime hooks.
Each patch names its input, prerequisites and application command. The native
`replay-reference` applies only extraction: original sentinels, pointer arithmetic,
pool management and realloc. `replay` applies all three. Native tests compare
payloads, counters, and an allocation-decision hash normalized by arena identity
and offset, including raw fallback locations.

The pinned 64-bit configuration uses 16 KiB pools, 1 MiB arenas, 16-byte size
classes and a 512-byte small-request threshold. The earlier survey guide's
4 KiB pool description is not this release's default. Capability pointers enlarge
the pool header from 48 to 80 bytes; both retain the upstream computed
`POOL_OVERHEAD`. Cross-ABI slot offsets therefore need not match.

## Lifetime mapping

Upstream still selects size classes, pools, lazy block carving, free-list order,
empty-pool reassignment, and arena retention. Numeric arena cursors are kept
separate from pointer authority. Actual sentinel objects replace upstream's
fabricated list-header pointers. Pool lookup returns retained metadata authority;
it cannot authorize a release without checking the current allocation capability.

Each arena retains a senior handle. Each pool is split into a persistent header
and a body; the body is split into size-class blocks on demand. Free-list links
remain in free blocks, as upstream expects. The adapter revokes a caller's lease
before returning a fresh allocator alias for writing that free-list link. A pool
changing size class reclaims its body, while its header remains accessible. Arena
release reclaims the whole subtree. Adapter records reside in separate metadata.

Both domain modes use the same layout and allocator code:

- `spatial`: request-bounded pointers; no per-object revocation. Reclassifying a
  pool or releasing an arena still revokes that backing subtree.
- `sublet`: additionally revokes on individual free and every successful realloc,
  including an in-place resize. Address reuse does not restore an old lease.

Moved realloc copies at most the previous requested length and preserves aligned
capability payloads. Failed realloc leaves its old allocation valid. Zero-size
requests retain CPython's non-NULL-on-success behavior. Large/raw requests use a
bounded reusable backing allocator, with the same release validation and lease
discipline in the domain. This backing allocator is not a port of libc malloc.

The fixed regions are 64 MiB payload (32 MiB small-object arenas, 32 MiB raw
fallback), 16 MiB metadata, 8 MiB trace and 4 KiB report. There are at most 32
arenas, 4,096 raw size-specific records, and 65,536 live replay identities.
Metadata includes replay tables and radix-tree nodes as well as authority
records. `metadata` in the report is heap high-water usage, not live bytes or
whole-process RSS. Metadata exhaustion during pool setup fails explicitly.
No timing or hardware-memory-overhead claim is made.

## Build

From the repository root, source `capstone/tests/capstone-test-env.sh`. Set
`CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR`, `CAPSTONE_QEMU_BINARY` and
`PORT_MUSL_ROOT` to prepared tools. A fresh worktree may use existing external
tool builds; its uninitialized submodules are not build inputs by default.

From this directory:

```sh
cmake --preset native
cmake --build --preset native
ctest --preset native
cmake --preset capstone-domain
cmake --build --preset capstone-domain
cmake --preset linux-guest
cmake --build --preset linux-guest
```

Presets build under `/tmp/capstone/cpython-pymalloc/build/`. Override with `-B`
or an untracked `CMakeUserPresets.json`; sources and outputs must be outside the
repository. No experimental application CI is added.

## Record and replay

The optional recorder needs a native, GIL-enabled, little-endian CPython 3.13.7
with development headers. It can be built from the same verified archive using
CPython's ordinary configure/build/install workflow in external storage.

```sh
cmake --preset native -DPYMALLOC_RECORD_PYTHON=/path/to/python3.13
cmake --build --preset native
PYTHONMALLOC=pymalloc /path/to/python3.13 host/record.py /tmp/pymalloc-stdlib.bin \
  --module-dir /tmp/capstone/cpython-pymalloc/build/native/python --rounds 20
/tmp/capstone/cpython-pymalloc/build/native/bin/replay \
  /tmp/pymalloc-stdlib.bin /tmp/pymalloc-native.bin
python3 host/run-qemu.py /tmp/pymalloc-stdlib.bin /tmp/pymalloc-spatial --protection spatial
python3 host/run-qemu.py /tmp/pymalloc-stdlib.bin /tmp/pymalloc-sublet --protection sublet
python3 security-tests/qemu/run.py /tmp/pymalloc-security
```

QEMU's Python environment needs `pexpect`. Runners accept `--domain-build` and
`--linux-build`, retain each attempt separately, and take the common QEMU lock.
An infrastructure failure is not automatically retried or counted as success.

The recorder wraps both MEM and OBJ APIs in the actual interpreter. It records
successful requests in a bounded, single-interpreter GIL workload: JSON parsing,
regex matching, bytearray allocation and resizing. It does not record raw
interpreter metadata allocations, object-specific free lists above pymalloc,
or failed requests. Bootstrap frees outside the window are ignored; a successful
realloc of an unknown bootstrap allocation starts a new recorded identity.
An explicit END carries the remaining live count; it does not fabricate frees.
Incomplete captures remain `.partial` and are rejected. Capture is intended for
this single-threaded driver, not arbitrary concurrent/subinterpreter programs.

Replay starts from an empty allocator. It replays API requests, not the original
warm interpreter heap. Synthetic byte payloads check every free/realloc and the
remaining live objects at END; this is not replay of Python object graphs.
The separate pointer-bearing realloc probe checks capability copying. The native
reference establishes extraction equivalence for these request streams, not
identical allocation addresses in the original interpreter workload.

## Verification

The native suite covers all size classes, raw fallback, zero sizes, realloc,
pool reclassification, arena release, malformed traces, source integrity and
patch order. With the recorder configured it also compares a real workload
against the unadapted native reference.

The QEMU security suite pairs spatial and Sublet modes across nine cases:
live siblings and capability-bearing realloc; read after free; write after reuse;
stale free after reuse; in-place realloc; one-past-end access; raw-fallback reuse;
pool size-class reassignment; and an alias retained across 2,000 reuses. A stale
access passes only with the setup marker, expected fault cause and exact fault
instruction. Stale free must return the explicit authority-rejection status.

Recorded results live in `results/`. These are QEMU component results, not FPGA
measurements or a complete protected CPython execution.
