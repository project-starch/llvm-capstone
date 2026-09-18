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
the capability ABI patches; Sublet adds per-context pools and allocation
lifetimes for **AllocSet, Generation, Slab and Bump**. These are versioned
patches to the pinned upstream managers, selected by the CMake Sublet target.

| Manager | Sublet lifetime boundary |
|---|---|
| AllocSet | Individual free/reuse and context reset/delete |
| Generation | Individual free, moving realloc, recycled blocks and context reset/delete |
| Slab | Individual free/same-address reuse and context reset/delete |
| Bump | Allocation bounds and bulk context reset/delete; no individual free/realloc |

Context headers and typed block sidecars live outside revoked payload bytes.
Slab's free chain uses side-table indices; its block selection and empty-block
retention remain upstream code. Generation keeps its block recycling policy.
Bump's zero-byte result is a non-dereferenceable address and consumes no bytes.
Logical block geometry is preserved for the capability ABI; actual side-table
overhead is separate from PostgreSQL's logical context statistics. Tables have
fixed budgets; released backing bytes become reusable at context reset, not
through general-purpose block coalescing.

The new patches require PostgreSQL's release-layout configuration: they reject
`MEMORY_CONTEXT_CHECKING` and `CLOBBER_FREED_MEMORY`, whose upstream walks can
touch revoked chunks. This is distinct from the CMake build's optimization/debug
setting. A remaining libc backing call is still refused, never silently routed
to an unprotected heap. None of this constitutes a protected PostgreSQL server
or a validated protected consumer-defect reproducer.

## Build and run

For small programs that directly use the allocation APIs, start with the
[client examples](examples/README.md). Each allocator has a separate client,
built unchanged for native, Capstone spatial and Capstone/Sublet execution.

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

The QEMU tests cover the original spatial/Sublet replay and hierarchy fixtures,
mixed-manager trace replay in both arms, and a paired context-allocator matrix.
The matrix checks allocation-policy counters, payloads, live siblings, bounds,
free/reset/delete/realloc aliases, block recycling, mixed context trees and
metadata-exhaustion recovery. Unsupported upstream operations (Bump free/realloc,
variable-size Slab allocations) are excluded. Expected faults must occur at the
declared read/write instruction after successful setup; an unrelated fault or
timeout is a failure. Native tests exercise the same policy workload and prove
that the replay oracle and verdict classifier reject bad controls.

Tests use the shared QEMU lock and retain per-run artifacts.
For custom build roots use `-B`, `-DPG_WORK=...` and
`-DPG_LINUX_BUILD_DIR=...`; pass matching build paths to `host/run-qemu.py`.
`security-tests/run-contexts.py OUTPUT --domain-build DOMAIN --linux-build LINUX`
also runs the matrix directly. After inspecting a failed attempt, its exact
`suite-*` directory may be passed with `--resume`; passing inputs must retain
identical binary/runtime hashes. Failed attempt directories are retained.

The original scripts remain **AllocSet-only** for existing experiment drivers;
use this CMake component for the four-manager Sublet port. Prepare their
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

The additional allocator ports pass eight native CTests, both mixed-manager
QEMU replay arms and all 72 paired context-matrix arms (25 expected exact-access
faults). All three capability-ABI policy hashes match the spatial arm. Compact
results and binary identities are in `results/contexts/20260918-qemu/`.
The four existing QEMU regressions also pass; the original Sublet replay needed
one explicit rerun after a post-result guest stall, recorded in that result set.

The earlier pin-consistency validation, before those ports were added:

Five native CTests and all four QEMU CTests passed with fresh 17.0 builds.
The original host fixture matched its oracle (72/68/1/12 blocks taken,
returned, grown and peak), and both original domain scripts compiled. The
native `live_parts` reproducer returned its expected verdict in plain and ASan
builds; its malloc-use-after-free control fired. Valgrind was unavailable and
was reported as skipped. These are correctness checks, not a rerun of the
historical 17.5 memory-profile campaign.

## Opt-in client fault recovery

The default emulator stops the VM on a C-mode capability fault. With the matching
QEMU local-trap-delivery change, configure a separate domain build with
`-DCAPSTONE_DOMAIN_FAULT_RECOVERY=ON`. The guest launcher then terminates with SIGSEGV
when the client returns the reserved fault result, while Linux and QEMU continue.
This is cooperative client recovery, not recovery from monitor/kernel crashes.
See [the runtime contract and limitations](../../../runtime/domain-faults.md).
The build helper and Linux fault policy are shared, not PostgreSQL-specific.
The old `PG_DOMAIN_FAULT_RECOVERY` option is a deprecated compatibility alias;
remove it from an existing cache before selecting a different shared setting.

For the single-boot regression, configure smaller matching regions in separate
build directories (the current driver retains allocations across processes).
After setting the toolchain environment and `CAPSTONE_QEMU_BINARY` as above:

```sh
PG_FAULT_WORK=/tmp/capstone/postgres-fault-isolation
PG_PORT=capstone/ports/postgres/memory-contexts
cmake --preset linux-guest -S "$PG_PORT" -B "$PG_FAULT_WORK/linux" \
  -DPG_ARENA_BYTES=8388608 -DPG_TRACE_BYTES=1048576
cmake --preset capstone-domain -S "$PG_PORT" -B "$PG_FAULT_WORK/domain" \
  -DCAPSTONE_DOMAIN_FAULT_RECOVERY=ON -DPG_LINUX_BUILD_DIR="$PG_FAULT_WORK/linux" \
  -DPG_ARENA_BYTES=8388608 -DPG_TRACE_BYTES=1048576
cmake --build "$PG_FAULT_WORK/linux"
cmake --build "$PG_FAULT_WORK/domain"
ctest --test-dir "$PG_FAULT_WORK/domain" -R '^qemu-(client-|fault-isolation)' --output-on-failure
```

The fault suite verifies real SIGSEGV deaths for Generation, Slab and Bump
use-after-reset clients, followed by a healthy client in the same VM. The
[standalone runtime suite](../../../runtime/tests/fault-recovery/README.md)
owns allocator-independent fault delivery, quarantine and fallback tests.
Recovery is OFF by
default so existing firmware/emulator installations keep their current behavior.
