# Application and allocator ports

Start here to choose a component and find its build entry point. A port may
execute an application, replay one allocator, or only establish that a library
compiles. Those scopes are different; a directory's existence is not evidence
of a complete protected application.

For the four newer allocator components, start with the
[shared CheriBSD guide](common/host/cheribsd/README.md): one purecap toolchain,
matching `host/cheribsd/build.sh` and `run.sh` scripts, CMake library targets,
and standalone link examples. The guide states the protection boundary of
each adapter. The experimental FFmpeg PoisonCap adapter has its own platform
and validation workflow linked from that guide.

## Components

| Component | Scope | Entry point and organization |
|---|---|---|
| SQLite | In-memory SQL workloads and memsys5/lookaside lifetime experiments | [SQLite](sqlite/README.md), [Sublet](sqlite/sublet/README.md); existing shell drivers |
| MicroPython | Freestanding interpreter and GC test workloads | [MicroPython](micropython/README.md); existing shell drivers |
| nginx | Pool allocator replay, pool/block lifetime and stale-access probes; not a web server | [Domain runner](nginx/run-nginx-domain.sh), [Sublet runner](nginx/run-nginx-subpool.sh); existing shell drivers |
| APR | Freestanding pool-library compilation and adaptation census | [Census](apr/census-apr.sh); no end-to-end domain workload established here |
| musl | Domain libc, host-call services and functional tests; incomplete OS/thread support | [musl](musl-capstone/README.md); existing shell drivers |
| FFmpeg | AVBufferPool/AVRefStructPool replay from native decoder recordings; not domain video decoding | [Buffer pools](ffmpeg/buffer-pool/README.md); shared CMake layout |
| PostgreSQL | Memory-context replay and memory profiles; not a database server | [PostgreSQL](postgres/README.md), [CMake component](postgres/memory-contexts/CMakeLists.txt); shared CMake layout plus older shell drivers |
| CPython | Real pymalloc replay from native interpreter recordings; not a domain interpreter | [pymalloc](cpython/pymalloc/README.md); shared CMake layout |
| Whisper / ggml | Context allocator and buffer-epoch replay; not domain speech recognition | [ggml contexts](whisper/ggml-context/README.md); shared CMake layout |

This integration branch includes the pending Whisper port and PostgreSQL/runtime
PRs. Consult the selected revision's `upstream.json` or fetch script for the
source pin, and the component's result bundle for the revision actually tested.
Historical results keep their original source and binary identities.

## Shared layout for component ports

FFmpeg, PostgreSQL memory contexts and CPython use the same build support.
Whisper uses it here as well. New component ports should use this
layout; the older application ports retain their established drivers until a
separately validated migration preserves their workloads and gates.

```text
ports/<project>/<component>/
  README.md                  scope, source pin, build/run, evidence and limits
  upstream.json              archive URL, version and checksum
  CMakeLists.txt
  CMakePresets.json           native, capstone-domain, linux-guest
  cmake/                     source preparation and target definitions
  patches/                   ordered, versioned upstream changes
  src/native/                native recorder or replay entry
  src/capstone-domain/        freestanding domain entry
  src/linux-guest/            Linux launcher and staging
  src/shared/                replay interfaces and shared implementation
  src/allocators/             allocator-specific authority adapters, if needed
  host/                      recording, launch and analysis tools
  tests/                     functional checks and oracle controls
  security-tests/            lifetime/bounds fixtures with explicit fault oracles
  results/                   compact evidence, hashes and provenance
```

This is the common convention, not a claim that every historical component
already has every directory. Execution environment belongs under `src/`;
`spatial` and `sublet` are protection modes, not separate platforms. A native
execution does not acquire revocation merely by selecting a mode number.

[common/](common/README.md) owns downloads, build-location guards, toolchains,
run staging and the QEMU lock. [runtime/](../runtime/CMakeLists.txt) owns
capability operations and Sublet primitives. Put allocator policy in its port,
and real consumer-defect reproducers in [bug-corpora/](../bug-corpora/), rather
than copying either into another port or into generic runtime support.

## Shared traces

The four CMake components use [shared trace tooling](common/host/port_trace/README.md)
for format detection, structural validation, inspection and trace/result metadata.
Existing binary formats and allocator-specific replay operations remain intact.
For a new allocator, add a format adapter and its corruption controls rather
than copying a reader or flattening its lifetime semantics.

## Build and evidence

Source `capstone/tests/capstone-test-env.sh` from the repository root before
building or testing. In a shared-layout component directory:

```sh
cmake --preset native
cmake --build --preset native
ctest --preset native
cmake --preset capstone-domain
cmake --build --preset capstone-domain
cmake --preset linux-guest
cmake --build --preset linux-guest
```

Cross builds need prepared compiler, guest and header dependencies; the
component README names them. Presets default to component-specific directories
under `/tmp/capstone`. Override with `-B` or `CMakeUserPresets.json`; when doing
so, also pass matching build paths to the runner. Changing `CAPSTONE_TMP_ROOT`
does not rewrite the literal `binaryDir` values in existing presets.

QEMU test presets and runner arguments vary by component. Use its documented
entry point; do not assume one port's test preset exists in another. Sources,
models, recordings, builds and raw logs stay outside the repository. Commit
compact result records with input/tool hashes, preserve failed attempts, and
distinguish historical evidence from a fresh validation of the current branch.

See the [integration plan](../docs/plans/port-stack-integration.md) for the
remaining cross-repository prerequisites and merge conflicts. Experimental
application ports have manual validation; a skipped GitHub workflow is not a
passing port test.
