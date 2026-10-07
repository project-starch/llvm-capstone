# Application and allocator ports

Complete Linux application ports now use the
[delegated ABI-v2 build and runner](common/application/README.md): Perl, mruby,
CPython, PostgreSQL single-user, SQLite, FFmpeg, offline tshark and the
[four-worker memcached server](memcached/app/README.md). Old application
images must be rebuilt. The component/allocator and silicon targets described
below are separate from these application entry points.

The [trusted Linux compatibility plan](../docs/plans/trusted-linux-application-compatibility.md)
defines the next target: ordinary Linux OS functionality for every application,
with capability changes justified by pointer representation, object bounds or
lifetimes. Its milestones are proposed work, not additional qualified features.

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
| APR | Pool allocator in a Capstone domain with a Sublet adapter, and apr-util's bucket allocator carried on top of it (`-DAPRP_BUCKETS=ON`); a stock CheriBSD build against the platform's own malloc and a PoisonCap build; for the httpd/APR pool and bucket defect corpora; not httpd, not threaded pools | [pools](apr/pools/README.md); shared CMake layout. The [census](apr/README.md) that preceded it stays beside it |
| memcached | The slab allocator and the per-thread object cache of 1.6.45 in a Capstone domain with a Sublet adapter, and a stock CheriBSD build against the platform's own malloc, for the memcached defect corpus; not the server, not its threads, not the page mover | [allocators](memcached/allocators/README.md); shared CMake layout. The [census](memcached/README.md) that preceded it stays beside it |
| musl | Domain libc, host-call services and functional tests; incomplete OS/thread support | [musl](musl-capstone/README.md); existing shell drivers |
| FFmpeg | AVBufferPool/AVRefStructPool replay from native decoder recordings; not domain video decoding | [Buffer pools](ffmpeg/buffer-pool/README.md); shared CMake layout |
| PostgreSQL | Memory-context replay and memory profiles; not a database server | [PostgreSQL](postgres/README.md), [CMake component](postgres/memory-contexts/CMakeLists.txt); shared CMake layout plus older shell drivers |
| CPython | Real pymalloc replay from native interpreter recordings; not a domain interpreter | [pymalloc](cpython/pymalloc/README.md); shared CMake layout |
| Whisper / ggml | Context allocator and buffer-epoch replay; not domain speech recognition | [ggml contexts](whisper/ggml-context/README.md); shared CMake layout |
| Wireshark / wmem | All four wmem allocators with pool-reset epochs and lifetime fixtures; not domain packet dissection | [wmem](wireshark/wmem/README.md); shared CMake layout |

Every component carries a `port.json` stating what it is — `full-application`,
`allocator-component`, `platform-build`, `domain-libc` or `census` — the upstream
version it pins, and the text in its own recipe that decides that pin.
`capstone/bug-corpora/tools/check-ports.py` reads that line and refuses a
declaration the recipe no longer supports, so the version is machine-readable
without becoming a second source of truth. The generated
[bug-material index](../bug-corpora/INDEX.md) joins those versions to the
corpora, which is the shortest answer to "which defects does this release have".
A component's result bundle still names the revision actually tested, and
historical results keep their original source and binary identities.

**[INDEX.md](INDEX.md) is the generated list of every component** by role, with its
pinned version, the targets it runs on, its workload and its corpora. Read it before
this table: it cannot go stale, and the table above can.

## Naming

**A component is named for what it is:** `app/` for the complete application,
`<boundary>/` for one allocator (`pools`, `pymalloc`, `buffer-pool`, `wmem`,
`memory-contexts`), `cheribsd/` for the same release built for another platform.
`role` in each `port.json` states the same thing, so nothing depends on reading a path.

Three components were renamed to that rule on 2026-09-28 — `cpython/interpreter`,
`mruby/musl` and `postgres/single-user` are now each `app/` — and four are deferred with
their reasons. **[renames.json](renames.json) is the path history**, rendered in
[INDEX.md](INDEX.md#path-history) and checked by `check-ports.py`.

The rename does not touch the evidence. An archived `build-manifest.json` records which
directory a measured binary was actually built from, so it keeps quoting the old path and
is never rewritten; `docs/history/` is append-only for the same reason. Eight archived
files quote a renamed path today, the ledger lists each one, and the checker fails if one
of them stops quoting it — so the record and its resolution table cannot drift apart.

What is deferred is deferred for a measured reason, not for taste. `perl/musl` has
another lane working in it. `sqlite`, `micropython` and `nginx` are flat trees, so the
move would add a directory level that every script inside resolves its own paths
against — and for `sqlite`, 99 files quote the path and 23 of them are evidence. Neither
can be claimed to work from a worktree that cannot run the nightly or the board gates
that drive those scripts.

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
