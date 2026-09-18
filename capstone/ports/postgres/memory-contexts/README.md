# PostgreSQL memory-context component

The port and native defect corpus use **PostgreSQL 17.0**, pinned by URL and
SHA-256 in `upstream.json`. The original scripts in `../` read that same file
through `../upstream.sh`; a conflicting `PG_VERSION` is rejected. Change the
manifest and the versioned patches together, then rerun the checks below.

17.0 is intentional: it retains consumer lifetime defects fixed in later 17.x
releases (see `../../../docs/ref/postgres-nested-allocator-defects.md`). This is
an isolated research baseline, not a recommendation to deploy an old server.
`port-origin.json` describes the original **17.5** adaptation. The committed
`results/memory/qemu/20260918-qemu-pgbench/` campaign also measured **17.5** and
must not be relabeled as 17.0 evidence.

## Scope

The seven upstream memory-manager files are built outside the server. Native
replay checks the recorded allocation counts and payloads. Spatial replay adds
the capability ABI patches; Sublet adds per-context pools and chunk lifetimes
for **AllocSet only**. Generation, Slab and Bump are compiled but their Sublet
backing requests are deliberately refused. The hierarchy and subpool fixtures
test revocation; none of this constitutes a protected PostgreSQL server.

## Build and run

Run from the repository root. Sources and binaries stay outside the checkout.
Native builds need CMake 3.25+, Ninja, Python 3.11.4+, a C compiler, make, patch
and PostgreSQL's configure/header-generation prerequisites (including bison and
flex). Configure downloads and verifies the pinned release, creates independent
patched variants, and rejects modified manager source.

```sh
source capstone/tests/capstone-test-env.sh
cmake --preset native -S capstone/ports/postgres/memory-contexts
cmake --build /tmp/capstone/postgres-memory-contexts/build/native
ctest --test-dir /tmp/capstone/postgres-memory-contexts/build/native --output-on-failure
```

Set `CAPSTONE_LLVM_BUILD_DIR`, `CAPSTONE_BUILDROOT_DIR` and
`CAPSTONE_QEMU_BINARY` for the shared test environment. Use a Python interpreter
with `pexpect` for QEMU tests (for example `-DPython3_EXECUTABLE=/path/to/python`).
The guest loader and domains must have identical region settings.

```sh
cmake --preset capstone-domain -S capstone/ports/postgres/memory-contexts
cmake --build /tmp/capstone/postgres-memory-contexts/build/capstone-domain
cmake --preset linux-guest -S capstone/ports/postgres/memory-contexts
cmake --build /tmp/capstone/postgres-memory-contexts/build/linux-guest
ctest --test-dir /tmp/capstone/postgres-memory-contexts/build/capstone-domain --output-on-failure
```

The four QEMU tests cover spatial replay, Sublet replay, subpool lifetimes and
context hierarchy. They use the shared QEMU lock and retain per-run artifacts.
For custom build roots use `-B`, `-DPG_WORK=...` and
`-DPG_LINUX_BUILD_DIR=...`; pass matching build paths to `host/run-qemu.py`.

The original scripts remain for existing experiment drivers. Prepare their
separate source tree before invoking the corpus:

```sh
bash capstone/ports/postgres/build-mmgr-host.sh
bash capstone/bug-corpora/postgres/mmgr-repros/run-host-repros.sh
```

Both commands must use the same `OUT` if overridden. The corpus rebuilds its
objects every time so a version or compiler change cannot reuse stale objects.
Its plain and ASan outputs reproduce the retained alias; ASan silence is not
evidence of protection. The optional Valgrind arm is explicitly skipped unless
available and configured. A Capstone arm for that consumer reproducer is still
separate work; the component's lifetime fixtures are not that reproducer.

## Validation of the 17.0 pin (2026-09-18)

Five native CTests and all four QEMU CTests passed with fresh 17.0 builds.
The original host fixture matched its oracle (72/68/1/12 blocks taken,
returned, grown and peak), and both original domain scripts compiled. The
native `live_parts` reproducer returned its expected verdict in plain and ASan
builds; its malloc-use-after-free control fired. Valgrind was unavailable and
was reported as skipped. These are correctness checks, not a rerun of the
historical 17.5 memory-profile campaign.
