# CPython 3.13.7 interpreter in a Capstone domain

**Status.** The complete interpreter runs. `run-cpython-domain.sh` starts it in a domain on the
musl-capstone libc with the standard library as a zip; the four-arm inner-reuse campaign
qualified it on the `objects.py 8 3 0` JSON/GC workload, 12 of 12 processes oracle-checked
(`capstone/experiments/study/results/cpython-reuse-four-arm-20260928/`). `cheribsd/` cross-builds
the same release for CheriBSD purecap, with ordinary pymalloc or the PoisonCap component.
Fourteen patches carry the pointer-layout changes, and `port.json` states the pin and the role.

**Everything below is the compile survey of 2026-09-23**, which is how this port started. It is
kept because it is the list of what did not compile and why, and because three of those causes
are compiler defects with registry entries. Its "Nothing has run yet" was true on that date and
is not the status above.

## The compile survey, 2026-09-23

The current source recipe builds the whole interpreter with the shared delegated
ABI-v2 application SDK. Build with `prepare-cpython-capstone.sh`, run the compile
survey, and link with `link-cpython-capstone.py`. Use the
[common runner](../../common/application/README.md) with a normal standard-library
directory (`PYTHONHOME` containing `lib/python3.13`). The private compiler driver,
startup and VM runner have been removed. The dated survey below describes the
earlier bring-up, including failures subsequently fixed in the compiler/port.

## Result, 2026-09-23

clang `d030df93d4a4` (= `origin/dev`), CPython 3.13.7, the configuration below.

| | compiled | of 253 |
|---|---:|---:|
| upstream as released (`CPY_PATCHES=none`) | 27 | 10.7 % |
| with `patches/0001`–`0006` | **222** | **87.7 %** |

| group | ok | fail | notrun |
|---|---:|---:|---:|
| parser | 15 | 1 | 0 |
| objects | 39 | 4 | 0 |
| python | 73 | 11 | 0 |
| core modules | 4 | 0 | 1 |
| stdlib modules (built in) | 72 | 14 | 0 |
| bundled libmpdec / expat / HACL* | 18 | 0 | 0 |
| `Programs/python.o` | 1 | 0 | 0 |

The one NOTRUN is `Modules/getbuildinfo.o`, which CPython's Makefile makes depend on every other
object. Two runs of the same configuration gave 222 both times.

**What the 31 are.** Four causes and the NOTRUN, each visible in the survey's "error sites" report:

| objects | cause | owner | route |
|---:|---|---|---|
| 26 | `llvm.ptrmask` on a capability crashes isel; every 8/16-bit atomic reaches it, and CPython's one-byte `PyMutex` is one | compiler, **C-51** | none at source level |
| 1 | `Python/compile.c`: the Greedy register allocator segfaults on `compiler_visit_stmt` | compiler, **C-52** | compiles at `-O0`, or `-O1 -mllvm -regalloc=basic` |
| 1 | `Objects/longobject.c`: `PyLong_FromVoidPtr`/`PyLong_AsVoidPtr` `#error` -- no integer holds a capability | CPython port decision | decide what `id()` and pointer round trips mean here |
| 2 | `Modules/_multiprocessing`: semaphore POINTERS parsed as integers (`F_POINTER`) | CPython | module has no use in a one-process domain |
| 1 | `Modules/getbuildinfo.o` | NOTRUN, above | |

A third compiler defect, **C-50** (`-g` with optimisation asserts in Assignment Tracking), failed
146 objects on its own before `capstone-cc` worked around it; see Configuration.

**Integer→pointer round trips in what compiles.** clang flags each explicit conversion of an
integer to a pointer with `-Wcapstone-pointer-roundtrip`: the result is untagged and faults when
dereferenced. The survey counts distinct source sites: **62 in 15 files**. The ones on the
interpreter's normal path matter first -- `Include/internal/pycore_gc.h` and `Python/gc.c` encode
the GC list links `_gc_next`/`_gc_prev` as `uintptr_t` with flag bits, which is every GC-tracked
object -- then `Objects/obmalloc.c` (13; the pymalloc port already carries the replacement,
`../pymalloc/patches/cpython-3.13.7-0002`). `Modules/_elementtree.c` (19) and
`Python/tracemalloc.c` (8) are off the startup path. The warning sees explicit casts only, not a
pointer carried through `memcpy` or a union, so the census is a lower bound.

## First link, 2026-09-23

    source $CPY_ROOT/build/capstone-env.sh
    python3 link-cpython-capstone.py $CPY_ROOT/build --native <native CPython 3.13.7 build>
    python3 ../../musl-capstone/check-domain-support.py $CPY_ROOT/build/link-attempt/python.dom \
        --sdk "$CAPSTONE_SDK" > domain-support.txt
    python3 host/startup-syscalls.py <native>/python domain-support.txt

`link-cpython-capstone.py` links, after a survey, what CPython's Makefile would link for a static
interpreter, the way every musl domain here is linked. Two absent objects get a measured per-file
workaround first: `Python/compile.o` with `-mllvm -regalloc=basic` (C-52) and
`Modules/getbuildinfo.o` by its own rule. Each undefined symbol is attributed with the native
build of the same source: which native object defines it. Controls: `PyLong_FromVoidPtr` must be
undefined and attributed to `Objects/longobject.o`; `Py_BytesMain` must not be undefined.

**Undefined symbols: 388, and only 3 of them are new information.**

| | symbols | |
|---|---:|---|
| defined by one of the 29 absent CPython objects | 385 | 26 of them C-51; `longobject.o` alone 45 |
| from a module configure left out here | 0 | |
| compiler-rt builtins that did not compile | 0 | |
| **provided by nothing in the link** | **3** | `__atomic_compare_exchange_16`, `__atomic_load_16`, `__atomic_store_16`: atomics on a pointer, **C-54** |

musl, the port runtime and compiler-rt supply every other symbol the 206 linked objects need.
There are no link errors besides undefined symbols, and no `.init_array` (nothing in a domain
would run one).

**Size.** Linked with undefined symbols ignored, the image is **8.37 MiB** (`PT_LOAD` memsz;
5.0 MiB `.text`, 2.5 MiB `.rodata`) *without* the 29 absent objects, among them `unicodeobject`,
`typeobject` and `ceval`. Scaling their native size by the capstone/native ratio of the present
objects (1.30) adds ~3.1 MiB: **~11.5 MiB for the whole static image, an estimate**, before any
heap. A domain gets 4 MiB.

**Linking is not working.** `musl-capstone/check-domain-support.py` on that image: 100 linked
libc symbols need one of **53 syscalls the domain does not serve**. (Measured against the
HostCall v0 served set of 2026-09-23. The checker now reads the delegated runtime's shape table,
and `link-cpython-capstone.py` no longer writes `domain-support.txt`: run the checker on the
image, as above.) Which of those CPython
itself makes while starting (`-S -I -c pass`), measured natively under `strace -k` and
attributed to the first CPython frame -- glibc and a dynamic loader, so an approximation:

| syscall | calls | made by | in a domain |
|---|---:|---|---|
| `rt_sigaction` | 66 | `PyOS_getsig` / `PyOS_setsig`, signal setup | fails; whether startup tolerates it is not known |
| `mmap` | 3 | `_PyMem_ArenaAlloc`: obmalloc's arenas, mapped directly | arenas cannot be allocated; configure's `HAVE_MMAP` has to say no so obmalloc uses `malloc` |
| `getdents64` | 4 | `os_listdir`, from the import system | importing from a directory fails; needs a hostcall or frozen/zipped modules |
| `getrandom` | 1 | `_Py_HashRandomization_Init` | CPython falls back to `/dev/urandom`, or `PYTHONHASHSEED` |
| `gettid` | 1 | `PyThread_get_thread_native_id` | returns an error value |
| `getcwd` | 0-1 | `_Py_wgetcwd` (only when started from some directories) | |

The remaining rows `strace` shows are glibc's (`brk` from its `malloc`, locale `mmap`, `futex`)
or the dynamic loader's; a domain has the port's allocator and a static image instead.

## At `-Os`, 2026-09-23

`survey-cpython-capstone.sh --opt=-Os` replaces only the `-O3` in CPython's `OPT`. It compiles
**222 of 253** too, a different 222: `Python/compile.c` compiles (C-52 does not arise), and
`Objects/dictobject.c` stops on **C-55** (two cascaded selects with a null capability). C-51 still
takes 26. Over the 221 objects that compile at both levels, `-Os` is **5.4 % smaller** than `-O3`
(text+data+bss 7,580,186 against 8,014,557; `.text` 5.8 %). That does not move a ~11.5 MiB image
toward 4 MiB. With `--opt` there is no baseline gate unless `--expect-ok` is given.

## How the survey measures

    source capstone/tests/capstone-test-env.sh
    CAPSTONE_LLVM_BUILD_DIR=<llvm build with clang, lld, llvm-ar> \
      bash capstone/ports/cpython/app/survey-cpython-capstone.sh --jobs 14

`prepare-cpython-capstone.sh` unpacks the pinned archive (`upstream.json`), applies `patches/`,
builds or reuses a native CPython 3.13.7 (`configure` requires one for a cross build), builds
`libc-capstone.a` with the compiler under test through `musl-capstone/build-musl-capstone.sh`,
the port runtime exactly as libc-test builds it, and compiler-rt's builtins as an archive, then
runs CPython's own `configure` with `toolchain/capstone-cc` as `CC`.
`survey-cpython-capstone.py` asks CPython's generated Makefile which objects make up the
interpreter, deletes them, runs `make -k` on exactly those, and reads one line per compile back
from `capstone-cc`. Every object is compiled with the flags CPython chose for it.

Everything is written under `$CPY_ROOT` (default `$CAPSTONE_TMP_ROOT/cpython-interpreter`).
`CPY_BUILD_PYTHON` reuses a native 3.13.7, `CPY_ARCHIVE` a downloaded archive, and
`CPY_PATCHES=none` measures upstream. The LLVM build needs only `clang`, `lld` and `llvm-ar`;
`capstone-test-env.sh` warns "toolchain check rc=1" when the other tools its freshness check
knows are absent, and `toolchain-fresh.py --targets clang lld llvm-ar` says whether the three
that are used are current.

What keeps the number honest:

* **The denominator is the Makefile's list.** An object make never attempted, because a generated
  header or a prerequisite failed, is NOTRUN and counted, not dropped.
* **Every error site, not the first.** Each object keeps its full diagnostics (`X.o.err`, with
  `-ferror-limit=0`), and the report counts every `file:line` error and every backend assertion
  across all failing objects. Counting first errors hid a second broken header behind the first
  one until it was patched; this shows a whole layer at once. A backend failure still ends its
  object, so what sits behind one stays hidden until it is fixed.
* **Controls.** `Objects/boolobject.o` must compile: it includes every header the six patches
  touch. `Objects/longobject.o` must fail, on the `PyLong_FromVoidPtr` `#error`, which is
  structural. A flipped control exits 2. With `CPY_PATCHES=none` the survey exits 2 on the first
  control, which is how the control was shown to fire.
* **Baseline.** Fewer than `BASELINE_OK` (222) compiled exits 1; `--expect-ok 223` was run and did.
* **The link check can say no.** `configure` decides `HAVE_<func>` by linking. `capstone-cc`
  links a real domain image against the archive musl compiled to; `prepare` refuses to continue
  unless `strlen` links and both an undefined function and `-lz` are refused.
* **Settings are read back.** `prepare` checks in what `configure` wrote that the static module
  build, `config.site`, the thread-local define and the absence of computed gotos all took effect.
  (The first of these checks was added after a comment line inside a continued command silently
  dropped two environment settings.)

## Configuration, and why

| setting | reason |
|---|---|
| `--host=riscv64-unknown-linux-musl` | the domain libc is musl, whose capstone64 arch layer derives from riscv64 and whose syscall numbers are Linux's; CPython's `configure` has no capstone target |
| `CC=toolchain/capstone-cc` | compiles with the flags the working musl domains use (`+m +a`, `-ffreestanding -fno-builtin -fno-jump-tables`, musl headers); links a real domain image so `HAVE_*` answers are about this libc |
| `MODULE_BUILDTYPE=static` | no `dlopen` in a domain; every module is built in |
| `--without-computed-gotos` | a table of `&&label` values is emitted without capability-init records and loads untagged; the first dispatch faults. It compiles cleanly, so no compile survey could flag it; the flag selects CPython's `switch` dispatch instead. `docs/history/05-08-2026_06-00-00_gp-captable-lua-bringup.md` |
| `-D_Py_THREAD_LOCAL_AS_GLOBAL` (+ patch 0006) | capstone64 cannot lower TLS (C-47); a domain has one hart and no clone, so a thread-local has one instance |
| `-Xclang -fexperimental-assignment-tracking=disabled` | C-50; keeps `-g` |
| `--with-pkg-config=no` | the host's pkg-config would hand over host library flags |
| `config.site` | no `/dev/ptmx` or `/dev/ptc`; `getaddrinfo` not buggy (it is never run) |
| `--disable-test-modules --disable-ipv6 --without-ensurepip` | not part of the interpreter |
| `ac_cv_libatomic_needed=no` | left to itself the check's conftest crashes the compiler (C-51), configure reads the crash as "needed" and puts a `-latomic` in `LIBS` that no capstone64 library provides. `prepare` now fails on any configure check that crashed the compiler unless its answer was reviewed; the two x87/mc68881 FPU checks crash on C-53 and "no" is right for this target anyway |

`configure` found, with these answers: `SIZEOF_VOID_P 16`, `SIZEOF_UINTPTR_T 8`,
`SIZEOF_LONG_DOUBLE 16`, `ALIGNOF_MAX_ALIGN_T 16`. 53 stdlib modules are built in; 13 are
missing a library this target does not have (zlib, bz2, lzma, OpenSSL, libffi for `_ctypes`,
curses, readline, gdbm, dbm, tk, uuid) and 16 are test modules or disabled. The survey prints
the list on every run.

**`HAVE_*` means "links", not "works".** `HAVE_FORK`, `HAVE_DLOPEN` and `HAVE_SIGACTION` are 1
because musl defines the symbols; the domain serves none of those syscalls. For the compile
survey that selects more code, not less. A runnable interpreter needs them answered the way the
WASI configuration answers them.

## Patches

Each one moves a header gate that stopped nearly every file, so that the survey can see past it.
A failure confined to one file stays a finding, not a patch. Each patch's header states the
assumption it corrects and why the replacement is right; all six leave every platform where
`sizeof(void *) == sizeof(uintptr_t)` unchanged.

| patch | header | assumption corrected |
|---|---|---|
| 0001 | `pyport.h`, `longobject.h` | introduces `_Py_SIZEOF_ADDRESS`; CPython uses `SIZEOF_VOID_P` for pointer storage, address width and "integer that holds a pointer", and on capstone64 those are 16, 8 and 8. The `intptr_t` format unit follows the address width. |
| 0002 | `pycore_pymem.h` | the freed-pointer fill-pattern check compares an address |
| 0003 | `pycore_obmalloc.h` | the radix tree indexes 64 address bits (as `../pymalloc` decided) |
| 0004 | `pycore_pyhash.h` | the pointer hash is of the address; it is never converted back |
| 0005 | `pycore_qsbr.h` | false-sharing padding assumed the per-thread state fits in 64 bytes |
| 0006 | `pyport.h` | opt-in: thread-locals as globals in a process that cannot start a thread |
| 0015 | `_posixsubprocess.c` | opt-in: `fork_exec` through `posix_spawn` in a process that cannot fork; `preexec_fn`, umask and id changes refused |

None of them makes a pointer↔integer ROUND TRIP safe; the census above is where those are.

## Signals, 2026-09-29

On `delegation-signals` the interpreter's `signal` module works against the
delegated runtime; the record is [results/signals-2026-09-29.json](results/signals-2026-09-29.json).

- [host/signals-smoke.py](host/signals-smoke.py) passes 18/18 in the guest: a
  handler with `os.kill` on the process itself and `raise_signal`; `alarm` and
  `setitimer` against `time.sleep`, with the sleep resumed (PEP 475) or the
  handler's exception propagated out of `sleep` and out of a blocking `read`;
  `set_wakeup_fd`; `pthread_sigmask`, `sigpending` and delivery on unblock;
  `sigtimedwait` for `SIGCHLD` with `si_pid` and `si_status`; a child that
  signals its waiting parent; `SIG_IGN` inheritance and `setsigdef` through
  `os.posix_spawnp`; `signal.pause`; `faulthandler.enable`; `KeyboardInterrupt`
  from `SIGINT`. One check documents the deviation: a loop that makes no libc
  call runs the handler only at the next call, while a loop that polls
  `time.monotonic()` is interrupted, because every entry into the dispatcher
  checks the recorded-sequence hint.
- `python -m unittest test.test_signal`: 25 pass, 0 fail, 13 skipped, 19 errors,
  every error an unserved `clone` (16, through `subprocess`), `socket` (2) or a
  thread start (1). `test.test_faulthandler`: 2 pass, 35 errors, all
  `subprocess`, 9 skipped.
- Not signals: at that revision `subprocess.Popen` still ran
  `_posixsubprocess.fork_exec` over `clone`; patch 0015 (below) routes it
  through `posix_spawn`. `epoll_create1` is unserved as well.

With `delegation-pty-ioctls`, `os.openpty()` and `pty.openpty()` work (musl's
`openpty` over `/dev/ptmx`, `TIOCSPTLCK` and `TIOCGPTN`); `pty.fork` and
`pty.spawn` still need `fork`. The two `config.site` answers
`ac_cv_file__dev_ptmx=no` and `ac_cv_file__dev_ptc=no` have no effect on
that path, since the build has `HAVE_OPENPTY` and uses musl's `openpty`
directly; they only stop `configure` from probing the build host's `/dev`.

## subprocess without fork, 2026-09-29

Patch 0015 (`-D_Py_FORK_EXEC_POSIX_SPAWN`) keeps `_posixsubprocess.fork_exec`'s
signature and its contract with `subprocess.py` and expresses the child steps
of `child_exec` as `posix_spawn` attributes and file actions: the pipe ends
onto 0, 1 and 2 with the same care when one of them already is a standard
stream, `cwd` through `posix_spawn_file_actions_addchdir_np`, `pass_fds` kept
by a dup2 onto itself (which clears `FD_CLOEXEC`), `close_fds` as one close
action per inheritable descriptor from 3 up read from `/proc/self/fd`,
`restore_signals` as `SETSIGDEF` of `SIGPIPE`, `SIGXFZ` and `SIGXFSZ`,
`start_new_session` as `SETSID`, `process_group` as `SETPGROUP`. Exec failure
is synchronous and raises the error named after the program; the error pipe
is never written. Refused with ENOSYS: `preexec_fn`, `umask`, `user`, `group`
and `extra_groups`. The launcher's spawner does the rest.

[host/subprocess-smoke.py](host/subprocess-smoke.py) passes 21/21 in the
guest: `run` with captured streams and exit status, `check_output`,
`check_call`, `communicate` in both directions, a 10 KB round trip through
`cat`, `env` and `cwd`, `FileNotFoundError` for a missing program and a
missing directory, `pass_fds` kept and an inheritable descriptor closed,
`restore_signals` both ways, `terminate`, `kill`, `wait(timeout)`,
`run(timeout)`, `start_new_session`, `process_group`, `preexec_fn` refused,
a child signalling its waiting parent, `os.popen` and `shell=True`.
CPython's own suites in the guest, booted with `--process-cache-mib 768`
(the default 384 MiB refuses a second CPython domain next to its parent):
`test_subprocess` 344 tests, 237 ok, 7 fail, 56 errors, 38 skipped, up from
179 ok before the patch; `test_popen` 5/5; `test_faulthandler` 25 ok, 11 fail,
1 error, 9 skipped, up from 2 ok; `test_signal` without
`test_interprocess_signal` 37 ok, 2 fail, 4 errors, 13 skipped, up from 25 ok.
What remained then, by cause: 26 refused by design; 24 `select.select` on the
unserved `pselect6`; the thread tests; children started without `PYTHONHOME`
(`env={}`, `-E`, `-I`, which is also why `test_interprocess_signal` hung: its
sender never started); `os.getpgid` and `os.getsid` unserved; two stderr
comparisons that saw the domain's `UNSERVED syscalls: 20` report; faults that
faulthandler cannot report because a domain fault is fatal;
`signal.pthread_kill(threading.get_ident())`, which rebuilds a `pthread_t`
from an integer; and `kill` to the parent, which the launcher refused.
With `delegation-runtime-rows` (`pselect6`, `getpgid`, `getsid`, `kill` to the
parent with ESRCH for a vanished pid, the unserved report only under
`CAPSTONE_DELEGATE_STATS`, musl's `pselect` mask kept a capability) and the
stdlib beside the image: `test_subprocess` 282 ok, 0 fail, 32 errors, 38
skipped, the errors being 26 refused by design, 4 threads and 2 descriptor
limits; `test_signal` complete, 57 tests, 38 ok, 2 fail, 4 errors, 13 skipped,
`test_interprocess_signal` included; `test_faulthandler` 26 ok. A child's
stderr is empty again. Record:
`runtime/tests/application/results/20260930-runtime-rows.json`.

With `delegation-cheap-rows`, the 32 plain rows (`statfs`, `statx`, `truncate`,
`linkat`, `fchown`, `getrusage`, `sched_getaffinity`, `setpgid`, ...):
[host/rows-smoke.py](host/rows-smoke.py), 28 CPython checks over the rows,
28 pass where 4 of 30 did before, the rest ENOSYS; `test_os` 366 tests,
256 ok, 2 fail, 38 errors, 70 skipped, from 229 ok, 6 fail, 60 errors; the
37 ENOSYS left are `timerfd_create` (12), `socket.socketpair` (5, the
sendfile tests' setup), `os.spawn*` and `fork` (17), `eventfd` (3); the
other three are `setreuid`, `setregid` and a `pathconf` on a bad
descriptor. `test_posix` 29 ok of the 45 before `test_lockf`, from 22, and
stops there in both runs: the domain faults before any fcntl round, musl's
`fcntl()` narrowing the `struct flock` pointer through `unsigned long`, the
class of musl patch 0004; a fifth patch, not a row. Record:
`runtime/tests/application/results/20260930-cheap-rows.json`.

With the fifth patch, `musl-patches/0005`, `test_lockf`
passes and `test_posix` runs to its end: 170 tests, 113 ok, 1 fail, 18
errors, 38 skipped. The errors are ENOSYS from calls that are not rows
(`getresuid`, `getresgid`, `setresuid`, `setresgid`, `setgroups`,
`initgroups`, `waitid`, `fexecve`, `preadv` with flags), EOPNOTSUPP
(`posix_fallocate` on the file system, `POSIX_SPAWN_RESETIDS` refused by the
spawner), and two CPython build items (`_testcapi`, `makedev`); the failure
is `test_register_at_fork`. The same objects linked without the patch fault
at `test_lockf` as before. `test_fcntl`: 12 tests, 6 ok, 3 skipped, 3
errors (`F_NOTIFY` not an admitted command; two lock tests need
`_multiprocessing`, which is not built). Record:
`runtime/tests/application/results/20260930-cpython-lockf.json`.

With `delegation-sockets` (nineteen socket and epoll rows, the word length
rule, the msghdr block, the datagram rule, and the data-copy rule for every
buffer crossing the exchange region): [host/socket-smoke.py](host/socket-smoke.py),
16 CPython checks over the rows, 16 pass; `test_epoll` 10 of 10, from a
suite that did not import; `test_selectors` 121 tests, 77 ok, 1 fail, 43
skipped, from 6 ok; `test_socket` 740 tests, 96 ok, 2 fail, 356 errors, 286
skipped, from 22 ok, and 350 of the errors are threads, the rest the Linux
abstract namespace (4: the build targets `capstone64-unknown-elf`, which
defines no `__linux__`, so CPython's abstract-namespace code is compiled
out while the kernel binds and connects the names; a configuration item for
the port), `getservbyname` without `/etc/services` in the image,
`if_nametoindex` (`SIOCGIFINDEX` is not an admitted ioctl), `sethostname`
and `testMaxName`; `test_asyncio.test_sock_lowlevel` 12 of
39 with every error a thread, `test_asyncio.test_streams` 53 of 74,
`test_socketserver` 8 of 27, all three from an ENOSYS at import. The
socket contract passes 11 of 11 in the guest. Record:
`runtime/tests/application/results/20260930-sockets.json`.
[host/run-filtered.py](host/run-filtered.py) runs a module without named tests.
Record: [results/subprocess-2026-09-29.json](results/subprocess-2026-09-29.json).

To run the suite, stage two things the port's scripts do not:

1. **The build's own `_sysconfigdata__linux_.py`.** In the cross build tree,
   `make pybuilddir.txt` generates it in `build/lib.linux-riscv64-3.13/`
   (checked 2026-10-01 on a copy of a 3.13.7 cross build). The name has no
   multiarch part because this build's `sys.implementation._multiarch` is
   empty. Without it every suite dies importing `test.support`, which needs
   `sysconfig.get_config_var`. The native riscv64 build's
   `_sysconfigdata__linux_riscv64-linux-gnu.py` copied under this name, which
   this README gave before, also imports, but it describes another build.
2. **The standard library next to the image.** Put `lib/python3.13` on the
   share beside the image, copied from the source `Lib/` (the `test` package
   included, an empty `lib-dynload`), with the file from step 1 in it. The
   interpreter finds that tree from `sys.executable`, so children started with
   `-I`, `-E` or an empty environment find it too; with the share at
   `/mnt/host`, `PYTHONHOME=/mnt/host` names the same tree. A `PYTHONHOME`
   elsewhere works for the parent but not for those children: that is the
   `test_interprocess_signal` hang above, and in the review of PR #170 it also
   gave spurious `test_posix` failures.

## What this does not establish

* **A complete link.** 29 objects are still absent, so the link above is of what compiles.
* **Running.** Nothing has executed. The 4 MiB contiguous domain limit
  (`../../musl-capstone/README.md`, `sscanf_long`) is far below what CPython needs to start.
* **The round trips.** 62 explicit sites are listed, not fixed, and implicit ones are not seen.
* **`-O` sensitivity.** Everything is at CPython's `-O3`. C-52 is present at `-O1` to `-O3`.
