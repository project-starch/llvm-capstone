# Delegated application ports

Application images use ABI v2 exclusively. The SDK supplies startup, argv,
environment, cwd, standard streams, libc overrides, syscall delegation and the
existing allocator grants. `capstone-exec` rejects old images with exit 126;
there is no HostCall fallback and no SDK switch to build v1 applications.
Rebuild applications and their SDK together.

The application ports are Perl, mruby, CPython, the PostgreSQL single-user
backend, SQLite's in-memory SQL application, the configured FFmpeg decoder,
and offline tshark. Standalone allocator and FPGA instruction tests are
separate targets; their historical HostCall transport is not an application ABI.

## Verified migration (2026-09-29)

[Result lines and binary identities](results/20260929-delegation.json) record
new QEMU runs, separate from the earlier application-memory measurements:

| Application | Delegated qualification |
| --- | --- |
| Perl 5.36.3 | `t/base`: 9/9 files, 493 assertions; existing upstream objects relinked with the current SDK |
| mruby head / 4.0.0-rc2 | Regular suites: 2616 OK / 1638 OK, zero failures or crashes; 71 / 10 skips; head's 23 stress cases not run |
| CPython 3.13.7 | JSON round trip and GC for 1000 objects, normal and inner-pymalloc Sublet variants |
| PostgreSQL 17.5 | Single-user SQL: 22 rows match native, normal and inner-context Sublet variants |
| SQLite 3.53.3 | In-memory workload `100 3 10`, exact result `EXP-OK sqlite 89700` |
| FFmpeg 9.0.1 | Five stages, 30/30 native frame hashes, changed input alters 10 hashes; level0 and protected pool variants |
| tshark 4.6.8 | Five stages, 10/10 capture/control verdicts; level0 and Sublet variants |

The shared regression passes 108 mixed starts without resource growth,
six shell/exec checks, and rejection of a v1 image with exit 126. Native
ASan/UBSan tests pass 23/23, host tests 21/21, libc runner tests 2/2, and
the FFmpeg/tshark safety fixtures 41/41. libc-test accounts for all 77 cases:
46 PASS, 4 FAIL, 4 FAULT, 3 NOBUILD and 20 EXCLUDED. These are functional
results; there is no new performance, silicon or complete-POSIX claim.

The protected PostgreSQL workload exhausts 65,536 revocation nodes. It passes
at 262,144 with an observed high-water mark of 84,193 nodes. mruby's protected
GC/popen smoke passes, but the larger 10-by-3000-array GC stress faults with
cause 30 at both capacities. That stress remains unresolved; larger capacity
is not a demonstrated fix for it. These controls and failures are retained in
the result record.

## Build

Source `capstone/tests/capstone-test-env.sh` after selecting a current
`CAPSTONE_LLVM_BUILD_DIR` and matching `CAPSTONE_LLVM_BIN`. The compiler must pass the SDK's direct-call
capability and `__uintcap_t` feature probes. The port compiler used for the
migration is `compiler/sroa-keep-capability-whole` at `7d01722aab88`; the compiler
sources on the delegation-only branch do not include all its prerequisites. A compiler that builds the SDK is not necessarily
new enough to compile every upstream program.

The source recipes now compile and link through the same SDK:

| Port | Source recipe | Application output |
| --- | --- | --- |
| Perl | `ports/perl/musl/build-perl-domain.sh` | `src/perl-5.36.3/perl` |
| mruby | `ports/mruby/app/build-mruby-domain.sh` | `src/mruby/build/capstone/bin/{mruby,mrbtest}` |
| CPython | `ports/cpython/app/prepare-cpython-capstone.sh`, survey and link scripts | `build/link-attempt/python.dom` |
| PostgreSQL | `ports/postgres/app/build-domain.sh` | `link/postgres.dom` |
| FFmpeg | `ports/ffmpeg/app/host/build-domain.sh` | `domain/ffapp_m5.dom` and fixtures |
| tshark | `ports/wireshark/app/deps/build-*.sh`, `host/cross-build.sh`, `host/build-domain.sh` | `domain/tshark_m5.dom` and fixtures |
| SQLite | `ports/sqlite/build-sqlite-capstone.sh` prepares the library objects; link the application below | `sqlite.dom` |

Paths in the table are relative to `capstone/` and each recipe's external build
root. Use a new build root when the patch set changes. Compiler changes must
also rebuild upstream objects, not only the final runtime archive.

`build.py` links existing upstream objects into an application. It records
input, compiler, runtime and image hashes in `manifest.json`; it does not
pretend that cached objects have been rebuilt from source:

```sh
python3 capstone/ports/common/application/build.py \
  --app cpython --root "$CPY_ROOT" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --out "$CAPSTONE_TMP_ROOT/cpython-application"
```

The output directory must be new. `--libc-root` selects a separate musl build;
`--musl` and `--libc` select explicit source/archive paths (useful for tshark's
keyed dependency builds). SQLite needs `--include` pointing at `sqlite3.h`.
`--heap sublet` selects outer malloc protection. `--nested cpython|mruby|postgres`
retains the existing inner allocator adapters and requests their static backing
from the v2 descriptor. The study entry point in `experiments/applications/`
uses this builder with `--instrument`; normal applications need no study wrapper.

## Run

Boot the VM with the current `capstone-exec`, `capstone-job`, kernel and module
as described in [runtime applications](../../../runtime/applications.md).
Use `up --cma-mib 1024 --process-cache-mib 768` for the full matrix including nested allocators. The
256 MiB rounded mruby grant and retained blocks from other image sizes exceed
the default 384 MiB retained-storage limit and 512 MiB CMA reservation in a long-lived mixed session; allocation
failure is reported, not silently retried with a smaller protected heap.
The kernel must support binfmt_misc for shells to execute images directly.
The common runner stages an immutable image copy and preserves normal streams:

```sh
python3 capstone/ports/common/application/run.py \
  --state "$VM_STATE" --cwd /tmp --result "$CAPSTONE_TMP_ROOT/run.json" \
  -e HOME=/tmp "$MRBD_ROOT/src/mruby/build/capstone/bin/mruby" \
  -e 'IO.popen("printf child") { |p| puts p.read }'
```

`--stdin FILE` connects a host file to real application stdin. `--user UID:GID`
starts the application with that guest identity, after dropping supplementary
groups. The guest account, resource paths and writable directories must exist.
Use the host share owner's UID for share-backed PostgreSQL clusters. The JSON
result distinguishes normal exit from signal death and includes a fault record.

CPython uses an ordinary standard-library directory (`PYTHONHOME` with
`lib/python3.13`); no ZIP workaround or argv/env side files are required.
PostgreSQL uses real stdin and an unprivileged account, with the pinned 16-byte
MAXALIGN cluster and installed share files. Its qualified small configuration
uses `shared_buffers=4MB`, `max_connections=10`, GMT, and
`dynamic_shared_memory_type=sysv`. Single-user statistics flush synchronously
instead of scheduling an idle SIGALRM. Statement/transaction/lock timeouts,
server mode and general signal handling remain outside this qualification.

mruby's `IO.popen` uses `posix_spawn` with standard-stream file actions and
closes other descriptors in the child. It preserves the native build's POSIX
HAL. General `fork`, threads, sockets, dynamic modules, file-backed mmap and
asynchronous domain signal handlers are not enabled by migrating a port.

Historical measurement archives retain their original binary hashes and ABI.
Their results do not become new measurements merely because the build recipe
now links against delegation.

## Existing regression runners

FFmpeg's `host/run-qemu.sh all` uses `CAPSTONE_VM_STATE` and checks real exit
statuses plus the native/changed-input frame oracle. Offline tshark's runner
uses the same VM and compares native application stdout. Safety fixtures use
`check-safety.py`, which combines actual job status with the fixture's full
printed mark or a matching fault diagnostic. Run safety checks exclusively on
the selected VM; they retain each QEMU log slice for target attribution.

The application-memory matrix runner accepts `user`, `cwd` and `stdin` fields.
`stdin` names a `/mnt/host/` file and participates in workload hashing; old
`PG_SINGLE_INPUT` and synthetic-root settings must be removed from new points.
