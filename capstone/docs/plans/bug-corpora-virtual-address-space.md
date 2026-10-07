# Every bug corpus in the virtual address space

Branch `virtual-capstone-bug-corpora`, on `virtual-capstone-integration`.

The question this lane answers: of the third-party defects this tree has
collected, which ones does **virtual Capstone** stop — capability pointers over
user virtual addresses, per-process lifetime tables, an ordinary Linux process
rather than a bare-metal domain?

## How much material there is

`capstone/bug-corpora/` holds **188 cases in 19 corpora over 9 programs**:
CPython, FFmpeg, httpd/APR, memcached, mruby, Perl, PostgreSQL, SQLite and
Wireshark. The cross-language `xlang/` tree adds 30 more rows, which are
declaration-level shims rather than programs and are out of scope here.

Two corpora cannot be run in this address space at all, and that is a property
of the cases and not of the platform:

- `sqlite/capi-repros` (19 cases) are pointer-lifetime defects in SQLite
  **bindings** written in Ruby, Go, PHP, Python and Node. Their arm is a host
  ASan build of another language's runtime.
- `mruby/gc-slot-repros` has 0 cases; it is declared `planned`.

That leaves **169 cases in 17 corpora, across all 9 programs**, and all 169 were
built and run.

## What the arm is

A case is a Linux process: the virtual SDK's `capstone-cc` compiles it against
the Capstone musl and links the virtual application runtime, and
`capstone-vexec` launches it under the loadable module and QEMU's U-mode
Capstone. Three consequences shape everything below.

**One boot per corpus, not one boot per case.** The physical arms launch one
bare-metal domain per case, because a capability fault ends the domain and a
faulted domain cannot report beside itself. Here the fault ends one process and
the guest keeps running, so a corpus of 33 cases is one boot and the shared QEMU
slot is taken once. `tools/run-virtual-corpus.py` brackets each case in the gate
script and scores the transcript afterwards.

**The system allocator is protected.** The virtual runtime's `malloc` bounds
every allocation to the request and revokes every copy of the alias on free
(`capstone/runtime/virtual/README.md`, and `heap.c`). So the arm is the virtual
counterpart of `sysalloc-sublet`, not of `sysalloc-bounds`: both spatial and
temporal safety are on, at the system-allocator boundary.

**A nested allocator is still a nested allocator.** Where an application
sub-allocates out of one block it owns — APR's pool nodes, pymalloc's arenas,
wmem's chunks, PostgreSQL's contexts, FFmpeg's buffer pools, SQLite's memsys5 —
the system allocator sees one object and no frees, and nothing it enforces can
see inside. That is the study's thesis, and these corpora were built to test it.
This lane measures it; it does not port each application's inner allocator.

## What was measured

Run 2026-10-07, every image rebuilt with the final tools and every corpus run
once. Bundles are in each corpus's `results/20261007-virtual*/` with a
`matrix.tsv`, an `inputs.json` carrying the sha256 of every image, the gate
script, each staged resource, the QEMU binary, the launcher, the module and the
kernel, and a README explaining the verdicts.

| application | corpus | cases | detected | silent | trap | controls | status |
|---|---|---:|---:|---:|---:|:---:|---|
| CPython | `cpython/pymalloc-repros` | 20 | **0** | 20 | 0 | 0/0 | PASS |
| FFmpeg | `ffmpeg/plain-heap-repros` | 4 | **4** | 0 | 0 | 4/4 | PASS |
| FFmpeg | `ffmpeg/plane-repros` | 1 | **0** | 1 | 0 | 1/1 | PASS |
| FFmpeg | `ffmpeg/pool-repros` | 4 | **0** | 4 | 0 | 4/4 | PASS |
| FFmpeg | `ffmpeg/subobject-repros` | 10 | **0** | 10 | 0 | 10/10 | PASS |
| httpd/APR | `httpd/apr-pool-repros` | 1 | **0** | 1 | 0 | 0/0 | PASS |
| httpd/APR | `httpd/bucket-repros` | 8 | **4** | 4 | 0 | 0/0 | PASS |
| memcached | `memcached/allocator-repros` | 9 | **6** | 3 | 0 | 0/0 | PASS |
| memcached | `memcached/plain-heap-repros` | 2 | **2** | 0 | 0 | 2/2 | PASS |
| mruby | `mruby/release-differential` | 23 | **19** | 3 | 1 | 1/1 | PASS |
| Perl | `perl/release-differential` | 11 | **7** | 4 | 0 | 1/1 | PASS |
| PostgreSQL | `postgres/c-repros` | 5 | **5** | 0 | 0 | 0/0 | PASS |
| PostgreSQL | `postgres/mmgr-repros` | 5 | **0** | 5 | 0 | 0/0 | PASS |
| PostgreSQL | `postgres/sql-repros` | 9 | **1** | 8 | 0 | 1/1 | PASS |
| SQLite | `sqlite/engine-repros`, platform allocator | 31 | **22** | 9 | 0 | 0/0 | PASS |
| SQLite | `sqlite/engine-repros`, memsys5 nested | 33 | **10** | 23 | 0 | 0/0 | PASS |
| Wireshark | `wireshark/plain-heap-repros` | 2 | **2** | 0 | 0 | 2/2 | PASS |
| Wireshark | `wireshark/wmem-repros` | 22 | **0** | 22 | 0 | 0/0 | PASS |

Counting SQLite once through its platform-allocator arm: **167 cases, 72
detected, 94 silent and one `trap`** — an mruby case whose process stopped on
an instruction page fault, cause 12, which is a stop but not the capability
mechanism answering. The two SQLite cases missing from that arm reach for `sqlite_heap`
directly and are therefore nested by construction; they are measured in the
memsys5 arm, which is what brings the coverage to 169 of 169.

`postgres/sql-repros` needed a per-case limit of 1200 s rather than the 300 s
the other corpora use: `07_1af08af694_pg_trgm_picksplit_getsign_on_bitvec` is
slow, not non-terminating, and at 300 s it was cut off. A cut-off row is not a
measurement and the record said FAIL for it until the corpus was run again
whole at the larger limit — a corpus with an unmeasured row does not get to
average it away.

### The result the corpora were built to produce

`sqlite/engine-repros` gives the cleanest form of it, because the two arms are
the same 33 case sources, the same amalgamation and the same binary apart from
one `sqlite3_config(SQLITE_CONFIG_HEAP, ...)` call:

- SQLite allocating through the platform — here the virtual runtime's own
  allocator, one bounded object per allocation and a revoke on every free:
  **22 of 31 caught**.
- SQLite allocating through memsys5, its own arena carved out of one block:
  **9 of those same 31 caught**.

Thirteen cases are lost to the nested allocator, and the loss is a strict
subset: no case is caught under memsys5 that the platform allocator misses. The
designed-in example behaves exactly so —
`18_mem5-design_memsys5_inband_freelist_overwrite`, whose defect *is* memsys5
storing freelist links inside freed blocks, is silent under memsys5 and faults
with cause 24 when the same sequence runs against the protected allocator.

The same split runs through the whole table. Every corpus whose cases take
their memory straight from `malloc` is answered — FFmpeg, memcached and
Wireshark plain-heap 8 of 8, `postgres/c-repros` 5 of 5 — while every corpus
whose boundary is inside one allocation is not: `cpython/pymalloc-repros` 0 of
20, `wireshark/wmem-repros` 0 of 22, `postgres/mmgr-repros` 0 of 5,
`ffmpeg/subobject-repros` 0 of 10, `ffmpeg/pool-repros` 0 of 4. Those silences
are not non-runs: each case prints its own pre-defect marker and each defect arm
reports `DEFECT-REPRODUCED` with the damage it did, inside the allocation its
nested allocator owns.

## What had to be built

Three builders, a planner, a runner, a collector and one toolchain, all new.

- `tools/build-virtual-cases.py` — a case that calls the platform's own
  `malloc` is a complete program for this SDK; no port library, no entry
  adapter. -O0 by default, as the corpora's CheriBSD arms use, so an optimiser
  does not move the fault away from the labelled probe.
- `tools/build-virtual-component.py` — for a case that replays an
  application's own allocator, through the port's existing one-way-in CMake
  seam, so the virtual images differ from the native ones in the toolchain and
  nothing else. Two seam shapes: one source per `defects` target, and the
  wmem/mmgr shape of one configure declaring one target per case.
- `tools/build-virtual-sqlite322.py` — the SQLite engine corpus needs the
  amalgamation compiled with each case's own feature flags, and reads the group
  table and the per-case group out of `corpus322.sh` rather than restating them.
- `ports/common/cmake/toolchains/capstone-application.cmake` — a port component
  built as a Capstone **application**: hosted sources, a libc, capability
  pointers. `capstone-domain.cmake` cannot do it, because a freestanding domain
  has no libc and every corpus in this tree puts `main()` in a driver that
  calls `printf` and `malloc`.
- `tools/plan-virtual-corpus.py`, `tools/run-virtual-corpus.py`,
  `tools/collect-virtual-results.py` — plan, run, file the bundle.

### Corrections the virtual build forced

Each of these was a real defect or a real limit, found by running:

1. **`PORT_PLATFORM capstone-application` is HOSTED** (`Port.cmake`). A
   component takes the same sources as native and cheribsd; what differs is the
   pointer representation, which is the toolchain's business.
2. **The pymalloc corpus could not be built hosted on Capstone.** Its `mark()`
   emitted three raw monitor marker instructions (`.insn r 0x5b`) for every
   non-PoisonCap build, and QEMU's U-mode decoder does not implement that
   custom opcode: all twenty cases stopped with cause 2, ILLEGAL_INST, before
   any case printed a character. Its three sibling corpora already key that
   block on their own `*_DOMAIN` macro; pymalloc now does too.
3. **PostgreSQL's memory-contexts port had no capability-pointer hosted
   build.** Upstream `aset.c` asserts that an `AllocFreeListLink` fits in the
   minimum chunk, which a 16-byte pointer does not, so the condition that
   selects the capability-layout variant is now about the pointer rather than
   about the operating system. Its hosted corpus programs were likewise gated
   on `cheribsd`; they are now built on any hosted platform that has a fault
   oracle.
4. **The virtual allocator refuses a single allocation above 256 MiB**
   (`runtime/virtual/heap.c`, `power()`): blocks are powers of two and 384 MiB
   would need 512. wmem's replay asks for exactly 384 MiB, got NULL and exited
   75 on all 22 cases. `WM_PAYLOAD_BYTES` is now overridable and this arm asks
   for 256 MiB, recorded in its bundle.
5. **The application SDK does not supply the compiler's own headers.**
   `capstone-cc` compiles with `-nostdinc` and then adds only the four musl
   directories; musl carries a `stdarg.h`, which is why most sources build, but
   `stdatomic.h` is clang's and FFmpeg's `buffer.c` needs it.
   `capstone-domain.cmake` adds the resource directory for the same reason and
   `capstone-application.cmake` now does too. The SDK itself still does not,
   which is a gap worth closing in `build-sdk.sh`.
6. **`check-ports.py` rejected the virtual platform.** Two components already
   declared `capstone-virtual` as a target when the virtual stack landed, and
   `ports/sqlite/cheribsd` carries an `arms` note; the checker knew neither, so
   it was BLOCKED with three problems on the base commit. The target and the
   field are now in the checker and in `SCHEMA.md`. One problem remains and is
   not this lane's: `ports/sqlite/cheribsd`'s `pin_source` quotes a grep whose
   text does not contain the declared version, and an honest fix needs a pin
   inside the repository — its recipe defaults to a path outside it.
7. **The stand-alone PostgreSQL backend needs two things the other corpora do
   not.** `postgres` refuses to run as root, so each case runs under
   `su nobody` — and the module registers `/dev/capstone-vm` with mode 0600, so
   the launcher cannot open it as `nobody` and every row reported
   `capstone-vexec: open: Permission denied`. And PostgreSQL resolves its share
   directory relative to the binary it was invoked as, so an image under
   `images/` makes it fall back to the compiled-in prefix and fail on
   `/usr/local/pgsql/share/timezonesets`. The gate now does what the virtual
   application gate does: `chmod 666 /dev/capstone-vm`, and a
   `/tmp/pg/bin/postgres` beside a link to the staged `share`.
8. **The runner's own markers must not be anchored to a line start.** A program
   whose last write has no newline leaves its text in front of the launcher's
   fault record, which PostgreSQL's `backend> ` prompt does. Anchored patterns
   read one real CAP_OOB detection as an unexplained SIGSEGV and one timed-out
   case as a non-run. All 19 transcripts were audited for mid-line markers
   afterwards; only `postgres/sql-repros` had any, so no other record was
   affected, and the rescore of an unaffected run reproduced its verdicts
   exactly.

## Where the per-case time goes

Measured with the launcher's own counters (`CAPSTONE_VM_STATS=1`,
`runtime/virtual/exec.c`), one boot, one row each:

| row | arena asked for | `pages` | `faults` | `launch_ns` | `elapsed_ns` |
|---|---|---:|---:|---:|---:|
| `wmem` case 0 | 256 MiB | 610 | 14 | 0.10 s | **0.16 s** |
| `pymalloc` case 0 | 64 MiB | 23,140 | 22,544 | 0.11 s | **36.4 s** |
| `ffmpeg/subobject` case 0 | 64 MiB | 21,345 | 20,493 | 0.17 s | **31.8 s** |

Process start, ELF load and registration are 0.10–0.17 s regardless of how long
the case takes, so neither the QEMU boot (one per corpus; a corpus of five
cases boots and runs in five seconds) nor the per-case launch is the cost. The
cost is first-touch page resolution: about **1.6 ms per 4 KiB page**, one fault
per page, through the module's GUP-and-pin path. The arena a port *asks for*
does not predict it — wmem asks for the largest and touches 610 pages, because
the virtual heap maps lazily — the pages a case actually *touches* do. The
collector did not run in any of the three (`collections=0`).

Two consequences. A supervisor process that survives a fault instead of dying,
which `capstone_domain_exit_on_fault` explicitly allows a caller to arrange
(`CAPSTONE_DOMAIN_FAULT_RETVAL`), would save that 0.1 s per case and not the 36.
And the lever that would matter is in the runtime, not in this harness.

## Limits of this lane

- **A capability fault cannot be caught in-process, by design.** The launcher
  reports it and `capstone_domain_exit_on_fault` resets SIGSEGV to `SIG_DFL`,
  unblocks it and raises it, "even if ignored or blocked"
  (`runtime/linux/domain-fault.c`), because the process must die before Linux
  releases mappings the module still holds. A case therefore cannot survive its
  own fault and report beside it; what may survive is a supervisor.
- **mruby's own smoke control does not hold here.** It stops with cause 30,
  INSUF_RESOURCES, at its M6 step. The arm therefore carries a trimmed smoke as
  its control, which holds, the full smoke as a recorded row, and a ladder. A
  first ladder of 100,000 short Ruby strings did **not** stop — short mruby
  strings are embedded in the object and never reach the allocator, so that
  ladder measured nothing. A second ladder of 64-byte strings, which do reach
  it, passes `LIVE=30000` and then faults, so the practical ceiling is **30,000
  to 35,000 live allocations**, below the 65,536 table slots: an mruby string
  costs more than one identity. The table size is a compile-time constant shared
  by the module, the launcher and the runtime (`runtime/virtual/wire.h`,
  `CV_NODE_ORDER`); raising it is a change to the virtual stack and is not made
  here.
- **Attribution is claimed only where the corpus's probe is an external
  symbol.** Where it is, the fault's pc is checked against that symbol's extent
  resolved from the image that ran, shifted by the load address the launcher
  published — the criterion the CheriBSD arms settled on. Two corpora
  (`wmem-repros`, `mmgr-repros`) label the exact faulting instruction with
  `.globl wm_defect_probe` / `pg_defect_probe` behind `#ifdef __riscv`, which a
  `capstone64` target does not define, so that label is absent and attribution
  falls back to function granularity. Widening that guard is a small follow-up.
- **This lane adds no `virtual` arm to the cases themselves.** Each corpus's
  `case.json` still declares the arms it declared before; the results live in
  `results/<stamp>-virtual*/`, registered as `evidence` in each `corpus.json`,
  and in this document. Declaring a per-case oracle for this arm — and teaching
  `check-corpus.py` about it — is the natural next step and is deliberately not
  mixed into this change.
- Nothing here measures a ported inner allocator, the physical arms, RTL
  behaviour, more than one hart, or any arm on CheriBSD.

## Reproducing

Environment as in `virtual-capstone-integration.md`, plus `CAPSTONE_SDK`
pointing at a built virtual SDK. Then, per corpus, build → plan → run → collect:

```sh
C=capstone/bug-corpora
python3 $C/tools/build-virtual-cases.py --corpus $C/ffmpeg/plain-heap-repros \
  --sdk "$CAPSTONE_SDK" --out /tmp/kit/ffmpeg-plain-heap \
  --include "$($CAPSTONE_LLVM_BIN/clang -print-resource-dir)/include"
python3 $C/tools/plan-virtual-corpus.py --repo . \
  --corpus $C/ffmpeg/plain-heap-repros --images /tmp/kit/ffmpeg-plain-heap \
  --out /tmp/kit/plan.json --arm-argv fixed --arm-argv buggy --case-number \
  --expect-symbol ffh_read_probe --expect-symbol ffh_read_probe_u8 \
  --expect-symbol ffh_write_probe_u8
python3 $C/tools/run-virtual-corpus.py --plan /tmp/kit/plan.json \
  --work /tmp/kit/run --qemu "$CAPSTONE_QEMU_BINARY" \
  --platform-images "$VIRTUAL_INTEGRATION_IMAGES" \
  --adapter "$VIRTUAL_INTEGRATION_OUT/adapter" --nm "$CAPSTONE_LLVM_BIN/llvm-nm"
python3 $C/tools/collect-virtual-results.py --run /tmp/kit/run --stamp 20261007-virtual
```

A correction to the parser or the verdict rules does not need another boot:
`--rescore` scores an existing work directory again from its own serial log and
says in the record that it did.

Every runner serialises through the shared QEMU lock. Do not work around it.
