# PostgreSQL in a domain: what `postgres --single` would need, measured

Branch `postgres/8-single-user-survey`, on `dev`, and `postgres/9-boot-attempt` on it. Component
`capstone/ports/postgres/single-user/` (the survey scripts, the session's SQL, the build, recorder
and runner of the boot attempt, its patches, and `results/2026-09-24/`). Written 2026-09-24, the
day the question was asked; the survey says what a port meets, in three layers, with the numbers
behind each, and the last section says where the backend, booted, actually stops.

The existing PostgreSQL port ([`../../ports/postgres/README.md`](../../ports/postgres/README.md))
is the memory manager alone, and its README says the program "wants an operating system, a file
system, sockets and processes". `postgres --single` is the one way to run the backend without the
postmaster: one process, SQL on stdin, no sockets. That is the door, and this is what is behind it.

## Layer 1: the operating system (done on branches, 2026-09-24)

`survey-native.sh` builds 17.5 natively, runs `initdb`, then one single-user session over
`work.sql` (DDL, 2000 rows, an index, a scan, a join, update, delete, vacuum, count) under
`strace -f -c`. The session is **one process**: no `fork`, `clone`, `execve` or `wait` in the
trace. It ran `work.sql` to its end (`count = "1500"`).

What it asked for, against what the domain runtime serves today (`hostcall.c`):

| need | calls | served | how |
|---|---:|---|---|
| `pread64`, `pwrite64` (and `preadv`, `pwritev`) | 255, 142 | **`runtime/pread-pwrite`** | the wire always carried an offset; the domain side now uses it |
| `openat`, `close`, `lseek`, `read`, `write`, `fstat`, `newfstatat`, `getdents64`, `fsync`, `fdatasync`, `unlink`, `access` | 208 … 11 | yes (dev) | — |
| `sync_file_range` | 44 | no | refusal is tolerated (`pg_flush_data` warns) |
| `rename` | 2 | **`runtime/path-rename`** | opcode `PATH_RENAME` (27) |
| `mmap(MAP_SHARED\|MAP_ANONYMOUS)` 16.6 MB, `shmget`/`shmat` 56 bytes, `mmap(MAP_SHARED, fd)` for dynamic shared memory | 1, 1, 2 | **`runtime/mmap-shm`** for the first two | a libc override on level0 (musl's `mmap`/`shmat` return the syscall's long as a pointer); file mappings answer `ENODEV`, so `dynamic_shared_memory_type=sysv` |
| `chdir(DataDir)` | 1 | **`runtime/cwd`** | a domain-side cwd joined onto every relative path; `getcwd` too |
| `getpid`, `getppid` | 5, 1 | **`runtime/pid-timer`** | 1 and 0 |
| `rt_sigaction`, `rt_sigprocmask` | 10, 6 | refused | tolerated (`pqsignal` returns `SIG_ERR`, unchecked) |
| `setitimer` | 1 | **`runtime/pid-timer`** | accepted and never fires, listed under `NO-OP` in the exit report |
| `epoll_create1`, `signalfd4` | 1, 1 | no; **`runtime/pipe-poll`** for the alternative | build with `WAIT_USE_POLL` + `WAIT_USE_SELF_PIPE`: `pipe2`, the pipe's `read`/`write`/`fcntl`/`fstat`, and `ppoll` are served in the domain (a wait with nothing ready returns at once, `NO-OP`) |
| `mkdir` (initdb's `CREATE DATABASE`), `rmdir` | 27 in initdb | **`runtime/mkdir-rmdir`** | opcode `PATH_MKDIR` (28), `PATH_DELETE`'s directory flag |
| `prlimit64`, `umask`, `getrandom`, `readlink` | 2, 1, 1, 18 | `umask` on `runtime/pid-timer`; the rest no | tolerated; the 18 `readlink`s fail natively too (`find_my_exec`) |
| `socket`, `connect` | 2 | no | glibc's nscd lookup, not PostgreSQL; musl does not make it |
| stdin | — | fd 0 is not a file in a domain | a few lines in `InteractiveBackend` to read from a named file |

The runtime rows were done on 2026-09-24, one branch each, stacked on the runtime PR chain
(`runtime/flush-on-return`, #87) in the order above, each with a QEMU probe and a control on the
runtime before it, and each gated on the whole `run-hostcall-all.sh` suite (23 probes at the end).
They are on the remote and not in `dev`. What the runtime still does not have is what PostgreSQL
tolerates: signals, `sync_file_range`, `getrandom`, `readlink`, `getrlimit`.

`initdb` is a driver: 13 `execve`s of `postgres --boot` and `postgres --single`, 27 `mkdir`, and
the rest is writing files. In a domain it becomes a host script that runs those two backends as
domain runs with their input files. The data directory the native session left is 39 MB in 977
files; the native `postgres` is 9.7 MB of text, so the image would be about 13 MB at the ports'
usual ×1.3, and the session fits a 128 MiB block with `shared_buffers` small.

None of this was new in kind: every row is an opcode or a local answer of the size the CPython
port added for R2, R7-R10; the rows took one day, where this plan had estimated two weeks. What
remains on this layer is on the PostgreSQL side: the `InteractiveBackend` input file, the initdb
driver, and the build settings of layer 2.

## Layer 2: the compiler (nearly nothing)

`survey-capstone.sh` runs PostgreSQL's own `configure` for `riscv64-unknown-linux-musl` with the
CPython port's `capstone-cc` as `CC` and `make -k` over `src/backend`. **957 of 966 objects
compile.** The nine that do not:

| objects | cause | route |
|---:|---|---|
| 6 | `StaticAssert(sizeof(ExprEvalStep) <= 64)`: the step's union holds pointers, which are 16 bytes now | a size guard for the executor's cache behaviour, not a correctness limit; raise it |
| 2 | clang exit 139 in code generation on `gram.c` and `jsonpath_gram.c`, the bison parsers | C-52's shape (the Greedy allocator on one huge function); its fix is on `compiler/c52-frame-base-capability` |
| 1 | `aset.c`: the free-list link no longer fits the smallest chunk | the mmgr port's two-line `port/aset-capstone.patch` |

Two configure answers are wrong for this target and must be forced: `MAXIMUM_ALIGNOF 8` (PostgreSQL
takes it from `long`/`double`, not from pointers; every `palloc` and every tuple would put
capabilities at 8-byte slots, the class of CPython's patches 0010/0012/0013) and
`HAVE_COMPUTED_GOTO 1` (`execExprInterp.c` dispatches through a table of `&&label`s, the trap the
Lua and CPython ports met; there is no `--without-computed-gotos` here, the answer is the cache
variable `pgac_cv_computed_goto=no`). The survey ran on clang `595e757cb696`, a build without the C-50…C-58 fixes, at
`-O2` without `-g`; with the integration compiler the two crashes should go.

## Layer 3: `Datum` (the problem)

A `Datum` is `uintptr_t` (`postgres.h:64`), eight bytes on capstone64 (`SIZEOF_UINTPTR_T 8`,
`SIZEOF_VOID_P 16`). Every by-reference value the executor moves — every `text`, array, tuple,
`jsonb` — travels as `PointerGetDatum(p)`, which is `(Datum) p`: the capability is cut to its
address, and `DatumGetPointer` gives back an untagged pointer that faults on its first use. The
compiler says so once per file (`-Wcapstone-pointer-roundtrip` at the inline function, 912 times)
and counts the casts: 1571 `-Wvoid-pointer-to-int-cast`, 910 `-Wint-to-pointer-cast`. In the
source of `src/backend`: 374 `DatumGetPointer`, 781 `PointerGetDatum`, 729 bare `(Datum)` casts,
4750 `DatumGet*`/`*GetDatum` calls, 482 of 869 files, and 275 places that spell out `sizeof(Datum)`
or `SIZEOF_DATUM`.

This is not a patch series. It is the question `plans/intcap-uintptr-model.md` answers: a
`uintptr_t` that carries a capability (CHERI's `__intcap`), 16 bytes, under which `Datum` works
as it does on CheriBSD. That plan costs 11-20 weeks and starts with an ISA decision, because on
silicon `MOVC` nulls an untagged value in a capability register (C-32) and `SCC` refuses one: an
integer `Datum` would not survive a register move. The plan lists PostgreSQL among the costs
("uses `intptr_t`/`uintptr_t` as ordinary integers"). A `Datum` redefined as a pointer type in
plain C (the CPython patches' B2 route, pointer arithmetic keeping the capability) meets the same
ISA rule and would run on QEMU only. Either way, PostgreSQL waits for that decision.

## What this says

The two layers a port usually fails on are open doors here: single-user mode is one process, its
dozen syscall additions are served on the runtime branches above, and the backend compiles. The
layer that is closed is the value representation, and it is closed by the ISA, not by PostgreSQL.
So the honest order is: the intcap decision first; then `postgres --single` is a two-to-four-week
port on top of it, with `initdb` driven from the host and the memory manager port already in
hand. Without intcap there is no PostgreSQL in a domain, on QEMU or on silicon, that is more than
the memory manager.

The link and the boot were done the same evening; the next section says where the boot stops.
Not done here: the two crashes on the integration compiler.

## The boot attempt (branch `postgres/9-boot-attempt`, 2026-09-24)

Component `capstone/ports/postgres/single-user/` grew a build, a recorder and a runner, and the
backend was booted in a domain on QEMU, on `initdb`'s own `--boot` invocation. Result lines in
`results/2026-09-24/boot-attempt.txt`; the pieces:

- `record-initdb.sh` runs the native `initdb` (`-U pg --no-sync --no-locale -E UTF8`) with a
  recording `postgres` in front of the real one. initdb makes five backend calls: `-V`, two
  `--check`, `--boot` over the 953,038-byte catalog script, and one `--single` session over the
  249,422 bytes of setup SQL. Each call's arguments, environment, input and the data directory
  it started from are kept, so the domain replays one call from the recorded state.
- `build-domain.sh` builds musl and the runtime from a runtime branch (`RUNTIME_REPO`), configures
  and makes the backend through `toolchain/capstone-cc` (`MAXIMUM_ALIGNOF` 16, no computed goto,
  `-DWAIT_USE_POLL`, system tzdata), and links `link/postgres.dom`: 760 objects, 0 undefined
  symbols. The link needed `src/timezone`'s objects, compiler-rt's int128 builtins
  (`numeric.c`'s `sqrt_var`), and two runtime branches, `runtime/path-readlink` (musl's
  `realpath()` under `find_my_exec`) and `runtime/strchrnul-alias` (the string override lacked
  the public name PostgreSQL's `snprintf.c` calls).
- `run-domain.sh <call>` stages the image under `bin/`, the native `share/` beside it (the
  backend derives its share directory from `argv[0]`), the recorded data directory (mode 0700),
  the input as a file (patch 0003: the backend has no stdin), and the arguments and
  environment for `toolchain/domain_entry.c`. The catalog script is regenerated from the
  template with this target's substitutions -- `SIZEOF_POINTER` 16 -- and checked against the
  recording (two rows differ, `pg_ddl_command` and `internal`). The guest links
  `/usr/share/zoneinfo` to the staged tzdata. The domain block: 128 MiB from CMA (`br-snap`'s
  rootfs module); the image with a 64 MiB level0 arena asked for 256 MiB and failed
  `create_dom`, so the arena is 32 MiB.
- Patches: 0001-0004 from the compile survey and the memory-manager port; 0005 `numeric.c`'s
  sort abbreviation for a 16-byte `Datum`; 0006 the root refusal off under `PGSU_DOMAIN`
  (the runtime reports uid 0); 0007 a `TYPEALIGN` that keeps the capability, with a `_Generic`
  guard that refused, at the link, the one operand in the backend that is not a byte pointer
  (0008, `read_stream.c`); 0009 the WAL prefetcher's context as a `void *` instead of a
  `uintptr_t`.

Five boots. The first failed `create_dom` (block size). The second halted at `XLOGShmemInit`:
`TYPEALIGN(XLOG_BLCKSZ, allocptr)` rounds the pointer through `uintptr_t`, and the `memset`
that follows faults on the untagged result -- the first `uintptr_t`-as-pointer site, before any
`Datum`. The third was a repeat by mistake (no `--enable-depend`; the tree rebuilt nothing for a
patched header) and stalled in the guest. The fourth, with 0007/0008, ran through
`checkDataDir`, the lock file, `CreateSharedMemoryAndSemaphores` and `BootStrapXLOG` -- the
domain wrote `global/pg_control` (native `pg_controldata` reads it: checkpoint 0/1000030, max
alignment 16), the 16 MiB WAL segment, the clog/subtrans/multixact pages and `postmaster.pid`
through the file service -- and halted in `StartupXLOG` on the prefetcher's `uintptr_t`
context. The fifth, with 0009, completes recovery, opens the catalog script, and halts on the
script's first statement, `create pg_proc`: `GetTableAmRoutine` calls the heap handler through
the fmgr, the handler returns `PG_RETURN_POINTER(&heapam_methods)` as a `Datum`, and the first
load through `DatumGetPointer` of it faults (`GetTableAmRoutine+0x38`, x10 = `heapam_methods`
as an integer).

So the boot stops exactly at the layer the survey named, and not before it: everything the
operating-system layer had to serve, served (a 16 MiB WAL write, recovery, the lock file, the
config), the `uintptr_t` alignment idiom is three small patches, and the `Datum` is the fmgr
calling convention of the whole backend. The next step is the intcap decision, not another
patch.
