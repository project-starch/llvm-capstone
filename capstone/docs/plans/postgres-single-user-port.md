# PostgreSQL in a domain: what `postgres --single` would need, measured

Branch `postgres/8-single-user-survey`, on `dev`. Component
`capstone/ports/postgres/single-user/` (two survey scripts, the session's SQL, and
`results/2026-09-24/`). Written 2026-09-24, the day the question was asked; nothing has been
ported, and this says what a port would meet, in three layers, with the numbers behind each.

The existing PostgreSQL port ([`../../ports/postgres/README.md`](../../ports/postgres/README.md))
is the memory manager alone, and its README says the program "wants an operating system, a file
system, sockets and processes". `postgres --single` is the one way to run the backend without the
postmaster: one process, SQL on stdin, no sockets. That is the door, and this is what is behind it.

## Layer 1: the operating system (bounded, runtime work)

`survey-native.sh` builds 17.5 natively, runs `initdb`, then one single-user session over
`work.sql` (DDL, 2000 rows, an index, a scan, a join, update, delete, vacuum, count) under
`strace -f -c`. The session is **one process**: no `fork`, `clone`, `execve` or `wait` in the
trace. It ran `work.sql` to its end (`count = "1500"`).

What it asked for, against what the domain runtime serves today (`hostcall.c`):

| need | calls | served today | what a port adds |
|---|---:|---|---|
| `pread64`, `pwrite64` | 255, 142 | no (`read`/`write`/`readv`/`writev` only) | two opcodes; the wire already carries a file offset |
| `openat`, `close`, `lseek`, `read`, `write`, `fstat`, `newfstatat`, `getdents64`, `fsync`, `fdatasync`, `unlink`, `access` | 208 … 11 | yes | — |
| `sync_file_range` | 44 | no | refusal is tolerated (`pg_flush_data` warns) |
| `rename` | 2 | no | needed (`pg_stat`, `replorigin_checkpoint`; WAL and control files in general) |
| `mmap(MAP_SHARED\|MAP_ANONYMOUS)` 16.6 MB, `shmget`/`shmat` 56 bytes, `mmap(MAP_SHARED, fd)` for dynamic shared memory | 1, 1, 2 | no | one process: all three can be served in the domain from the heap; `dynamic_shared_memory_type=sysv` folds the third into the second |
| `chdir(DataDir)` | 1 | no | fatal if refused; a domain-side cwd (prefixing paths before the hostcall) |
| `getpid`, `getppid` | 5, 1 | no | any fixed value; they go into the lock file |
| `rt_sigaction`, `rt_sigprocmask` | 10, 6 | refused | tolerated (`pqsignal` returns `SIG_ERR`, unchecked) |
| `setitimer` | 1 | no | `timeout.c:339` treats failure as fatal on the schedule path; serve it as a no-op (no signals in a domain) |
| `epoll_create1`, `signalfd4` | 1, 1 | no | fatal at latch setup; build with `WAIT_USE_POLL` and a self-pipe, which needs a domain-side `pipe2` (nothing ever waited in the session) |
| `prlimit64`, `umask`, `getrandom`, `readlink` | 2, 1, 1, 18 | no | tolerated; the 18 `readlink`s fail natively too (`find_my_exec`) |
| `socket`, `connect` | 2 | no | glibc's nscd lookup, not PostgreSQL; musl does not make it |
| stdin | — | fd 0 is not a file in a domain | a few lines in `InteractiveBackend` to read from a named file |

`initdb` is a driver: 13 `execve`s of `postgres --boot` and `postgres --single`, 27 `mkdir`, and
the rest is writing files. In a domain it becomes a host script that runs those two backends as
domain runs with their input files, plus `mkdirat` served. The data directory the native session
left is 39 MB in 977 files; the native `postgres` is 9.7 MB of text, so the image would be about
13 MB at the ports' usual ×1.3, and the session fits a 128 MiB block with `shared_buffers` small.

None of this is new in kind: every row is an opcode or a local answer of the size the CPython
port added for R2, R7-R10. About two weeks of runtime work, if the other two layers were solved.

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

The two layers a port usually fails on are open doors here: single-user mode is one process with
a syscall list of about a dozen additions, and the backend compiles. The layer that is closed is
the value representation, and it is closed by the ISA, not by PostgreSQL. So the honest order is:
the intcap decision first; then `postgres --single` is a two-to-four-week port on top of it, with
`initdb` driven from the host and the memory manager port already in hand. Without intcap there
is no PostgreSQL in a domain, on QEMU or on silicon, that is more than the memory manager.

Not done here: a link of the 957 objects (the undefined-symbol list would name the libc gaps as
CPython's first link did), and the two crashes on the integration compiler.
