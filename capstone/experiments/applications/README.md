# Application memory experiments

One runner, a reusable link adapter, and application workloads. These are
bounded discovery experiments for selecting paper workloads, not accepted
hardware performance results. They run applications, not allocator traces.

The application set is SQLite, CPython, Perl, mruby, PostgreSQL single-user,
the configured FFmpeg decode application, and tshark. The design discussion
and plotting script are in nested-allocators-paper's
`eval/application-memory` branch, `experiments/application-exploration/`.

## Build and run

Source `capstone/tests/capstone-test-env.sh` first. Use the existing per-port
recipes to build upstream objects outside the repository. `build.py` takes
those objects without modifying them and relinks through the common
application SDK. It refuses to overwrite an output directory. For example:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/applications/build.py \
  --app perl --root "$PERL_BUILD" --toolchain "$COMPILER_BUILD" \
  --input-revision "$PORT_COMMIT" --out "$EXPERIMENT_ROOT/perl-level0"
```

`--root` is the port's build root, not the source checkout. The adapter reads
Perl's built static extensions, CPython's Makefile object lists, PostgreSQL's
objfiles lists and static modules, mruby's archive, or FFmpeg's configured
libraries. SQLite takes the existing amalgamation and VFS objects, with
`--include` naming the upstream sqlite3.h directory. FFmpeg and SQLite use
`--libc-root` to select an existing musl build. tshark reads the upstream
ninja link command; its dependencies must already be built.

`--heap sublet --heap-log N` selects the common Sublet malloc implementation.
The separate `--nested cpython` and `--nested mruby` modes select already-built
ports of pymalloc and GC slots while leaving their outer malloc as level0.
They allocate the grants described in `regions.c`; they do not silently
fall back if a protected port's region is absent. Record the supplied source
revision, object hashes, protection contract and build manifest together.
For reused caches, the supplied revision identifies the port recipe; it does
not independently verify the cache's source origin. Archive the measured
images and actual input objects before treating a discovery result as durable.

Stage images and workload scripts in the VM's share under `experiments/`.
The language workloads take `normal_batch_size batches retained_records`.
Batch `batches/2` is a 4× burst. Native execution of the same workload must
produce the independently calculated checksum before its domain cell is run.
CPython also needs its prepared standard-library zip under
`experiments/pyhome/lib/python313.zip`; Perl uses its staged standard library.

```sh
python3 capstone/experiments/applications/matrix.py \
  --share "$VM_SHARE/experiments" --out "$EXPERIMENT_ROOT/points.json"
python3 capstone/experiments/applications/run.py \
  --state "$VM_STATE" --points "$EXPERIMENT_ROOT/points.json" \
  --out "$EXPERIMENT_ROOT/new-run" --repeat 3 --timeout 90
```

The JSON matrix is also the interface for application-specific workloads:
`application`, `arm`, `image` (host path under the share), `argv` (guest paths),
`environment`, expected stdout and expected phase names. PostgreSQL can supply
an ordered `expected_values` list checked against its named `oracle` columns.
The runner also rejects PostgreSQL ERROR/FATAL/PANIC output. Each attempt gets
complete stdout, stderr, actual guest exit/signal evidence, binary hashes,
resource counters and a record in `runs.jsonl`. A missing image is unavailable,
not a passed or faulted application. The runner preserves its own source.

It does not reboot or retry. Host timeout sends SIGTERM through the CLI's
guest cancellation path. Failure of VM control or resource cleanup stops the
campaign. Every returned fault is recorded, and subsequent independent cells
may run in the same boot. A changed node budget requires a separately recorded
VM configuration and complete new runs.

## Measurement contract

`memory.c` interposes `main` and `write` only in measurement images. At each
explicit `write(2, "MEMPHASE name\n", ...)` boundary it reports occupied heap
blocks before forwarding the marker. It uses stack formatting and direct
writes, with no malloc. An atexit hook captures peak usage even if main calls
exit. Fatal signals have no fabricated final sample.

For level0, live/peak count occupied blocks including headers and alignment;
`end` includes holes up to the furthest block; `pool` is the fixed arena.
For Sublet malloc, live/peak count occupied buddy blocks, including internal
slack; `tables` counts the statically reserved capability/control arrays.
Neither is RSS or total machine memory. Code, stack, nested grants, physical
rounding, revocation nodes, tag storage and the driver's cache are additional
ledger entries. Driver live bytes returning to zero does not mean cached
physical pages returned to Linux.

SQLite emits its own memsys5 requested-byte accounting in `EXP-INNER` lines;
do not add that to the 8 MiB backing block and count the same bytes twice.
PostgreSQL queries `pg_backend_memory_contexts` as an independent context
ledger. FFmpeg validates decoded frame hashes against stock native ffmpeg.
tshark's generated PCAPs vary flow count independently from packet count.

The baseline language scripts use application allocators and their normal
retention policies. Different applications' records are not interchangeable
units. The level0 controls do not provide per-object temporal safety. Current
SQLite and FFmpeg discovery images do not protect nested pool leases.

## Validation and resource limits

Run `python3 capstone/experiments/applications/test_runner.py` for false-pass
controls: wrong oracle, missing completion, missing guest status, timeout,
signal and impossible counters. `workloads/calibrate.c` checks two known
allocations followed by release; level0 occupies 1,136 bytes, while a Sublet
heap with 256-byte atoms occupies 1,280. Both must return live bytes to zero.
`workloads/files.c` exercises all 128 file slots, EMFILE and reuse twice.

The common SDK now includes the existing capability-aware atomic helpers and
compiler-rt's 128-bit integer helpers, needed by CPython and PostgreSQL.
The existing PostgreSQL file-table change is applied to the common runtime:
domain and helper use one capacity constant. These are ordinary application
prerequisites, independent of measurement instrumentation.

QEMU wall time is diagnostic only. Node allocations per launch include startup
and teardown and are cumulative, not peak live metadata. The supervisor may
recycle nodes between launches by sweeping capability tags; it does not run
that collector inside a still-executing application. A process restart cannot
be presented as another epoch in a continuous workload.

## Checked discovery, 2026-09-27

The first campaign records 183 attempts over six real applications and one
unavailable application: 147 pass, 21 signal, 3 nonzero exits and 12 unavailable.
All 171 launched attempts returned with zero live domains, regions and bytes.

| Application | Pass | Signal | Nonzero exit | Unavailable |
|---|---:|---:|---:|---:|
| Perl | 18 | 0 | 0 | 0 |
| CPython | 39 | 9 | 0 | 0 |
| mruby | 36 | 12 | 0 | 0 |
| SQLite | 18 | 0 | 0 | 0 |
| FFmpeg decode application | 30 | 0 | 0 | 0 |
| PostgreSQL single-user backend | 6 | 0 | 3 | 0 |
| tshark | 0 | 0 | 0 | 12 |

This is bounded workload exploration, not upstream test-suite acceptance.
Increasing the node budget from 65,536 to 262,144 lets the identical mruby
GC-slot 512-record cases complete and moves CPython's protected-pymalloc failure
boundary. The larger CPython cases still fault near the node limit. PostgreSQL's
512-row case reaches an unsupported FileFallocate operation during its burst;
tshark's build artifacts were absent and were not rebuilt for this campaign.

FFmpeg's Sublet malloc peak stays at 897,792 occupied bytes across 1, 4 and 16
30-frame streams in one process, with every decoded frame hash checked and live
heap returning to zero. Its 16-stream case still consumes 20,239 node allocations.
This does not establish indefinite protected reuse or total memory overhead.

The paper branch `eval/application-memory` contains all compact attempt records,
six PNG/PDF figures and the detailed analysis in
`experiments/application-exploration/results/2026-09-27/`. Its `archive.json`
fingerprints the durable raw archive, including binaries and input objects.
No CheriBSD comparison or hardware timing is claimed. Native output oracles,
four native runtime tests, twelve host CLI tests, eleven runner tests, heap
calibration and the 128-file capacity/reuse gate pass.
