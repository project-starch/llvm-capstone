# CPython interpreter port — what stands between here and a running interpreter

Branch `cpython/6-integration`, stacked on `cpython/3`..`5`. Component
`capstone/ports/cpython/interpreter/` ([README](../../ports/cpython/interpreter/README.md)).
First written 2026-09-23 against clang `d030df93d4a4` (= `origin/dev`); rewritten the same day
after the compiler fixes landed on their branches. "Integration compiler" below is `4c7f04e3d817`:
`origin/dev` with the C-50, C-51, C-52, C-54, C-55 and C-58 branches merged, a local build that is
not itself a branch to merge. Survey and link results for it with patches 0001-0013 are in
`results/*-4c7f04e3d817.txt`; `results/*-97d5978c6402.txt` are the same without C-58 and with
0001-0009.

## Where it stands

| step | on dev `d030df93d4a4` | on the integration compiler, patches 0001-0013 |
|---|---|---|
| compile survey | 222 of 253 objects | **250 of 250** (`_multiprocessing`/`_posixshmem` n/a) |
| link | 388 undefined symbols | **0 undefined**; controls pass |
| image | 8.4 MiB lower bound | 11.9 MB file, 59.8 MiB in memory (48 MiB of it the heap arena) |
| in a domain | nothing ran | **runs a script**: imports from the stdlib zip, prints, exits 0; see "Boots" |

Owners: **compiler** = the compiler lane (`llvm/`); **port** = whoever holds this branch;
**runtime** = domain runtime / hostcall / kernel module; **lead** = the project lead's decision.

## What has to merge, in order

Each is its own branch, pushed, independent of the others unless stated. **2026-09-24:** every
row but the last two is merged into `dev` -- the compiler fixes as #75-#80, the runtime rows as
#82-#87 (rebased into the first musl-domain runtime stack, on the QEMU-lock fixes of #81) -- and
the registry entries the port opened (C-51..C-58) are in `docs/ref/ISSUES.md` with those
numbers. The buildroot branch is #97. The port itself is re-landed on `dev` by
`cpython/8-reintegration` (this branch): the same interpreter directory, patches and results,
rebuilt on `dev`'s compiler and runtime; `cpython/6-integration` keeps the history of how it got
there.

| branch | what | gates |
|---|---|---|
| `compiler/c51-ptrmask-capability` | C-51: `llvm.ptrmask` on a capability; sub-word atomics with capability addresses | lit, QEMU 18/18, CoreMark, BEEBS |
| `compiler/c54-capability-atomics` | C-54: capability-valued atomics stay capabilities; runtime `atomic_libcalls.c` | lit, QEMU 7/7 + control, CoreMark, BEEBS |
| `compiler/c55-cascaded-select` | C-55: physical null capability out of the cascaded-select PHI | lit, reproducer, CoreMark, BEEBS |
| `compiler/c52-frame-base-capability` | C-52: local-stack-slot base register as a GPCR | lit, reproducer, CoreMark, BEEBS |
| `compiler/c50-assignment-tracking-index-width` | C-50: Assignment Tracking at the index width | lit, reproducer, CoreMark, BEEBS |
| `compiler/c58-no-speculative-cap-arith` | C-58: MachineLICM does not hoist trapping capability arithmetic out of a guarded block | lit, CoreMark, BEEBS |
| `runtime/c56-weak-at-exit` | the runtime's at-exit hook defined, not weak-declared (C-56) | QEMU matched pair |
| `runtime/r2-dir-read` | hostcall `DIR_READ`: `getdents64` in a domain | QEMU 7/7 + control |
| `runtime/hostcall-bounce-buffer` | file reads and writes through a bounce buffer: 9p cannot copy into the payload mapping (`EFAULT`) | QEMU `large-read` + control |
| `runtime/stdio-fstat` | `fstat` on stdout/stderr, `EBADF` from `fcntl` on a closed descriptor | QEMU `stdio-fstat` + control |
| caplifive-buildroot `domain/1-cma-large-domains` | domain blocks beyond 4 MiB from CMA | QEMU matched pair; the 6.4 (board) branch not built yet |
| `cpython/3`..`6` | the port itself | survey, link, boots |

The buildroot branch is a submodule branch; `cpython/6-integration` pins the submodule at it
(`2b8ad05`), which is where `run-cpython-domain.sh` builds the module from by default.

## Compiler requirements

| ID | feature | status |
|---|---|---|
| A1 | `llvm.ptrmask` on a capability (C-51) | **fixed**, branch above |
| A2 | atomics on capabilities (C-54) | **fixed** through the generic libcalls, one-hart runtime; lock-free inline load/store later |
| A3 | Greedy allocator on `compiler_visit_stmt` (C-52) | **fixed**: the cause was the frame base register's class |
| A4 | Assignment Tracking at the index width (C-50) | **fixed**; `capstone-cc` no longer disables it |
| B1/B2 | provenance through integers | **B2 in plain C**, no new builtins needed: pointer arithmetic keeps the capability, `__builtin_align_down/up` work since C-51. Used by patches 0008 and 0009 |
| B3 | capability TLS (C-47) | open; patch 0006 serves a one-thread domain |
| B4 | inline-asm `"m"` inputs (C-53) | open; two `configure` probes crash on it, both answering correctly anyway |
| C1 | `-Os` (C-55) | **fixed**; worth only 5.4 % |
| C2/C3 | code density, hardware float | not started |
| D1/D2 | jump tables, computed goto | not started; the interpreter runs a compare chain |
| — | capability arithmetic hoisted above its NULL guard (C-58) | **fixed** for MachineLICM; IR-level GEP speculation not covered, not seen |
| — | a null capability as a load base selects `$x0` (C-57) | open, verifier-only as far as seen |
| — | an undefined weak symbol's address is not NULL in a domain (C-56) | the runtime no longer depends on it; toolchain side open |

## Port patches

0001-0006 as before (address width, freed-pointer address, radix tree, pointer hash, qsbr
padding, thread-locals as globals). New today:

| patch | what | found by |
|---|---|---|
| 0007 | `PyLong_FromVoidPtr`/`AsVoidPtr` at the address width (`longobject.o` compiles) | the survey |
| 0008 | GC list links as pointers (`_gc_next`/`_gc_prev`), flag bits by pointer arithmetic | reading; tested natively with the representation forced (20 suites, 4952 tests) and a mutation that must crash, which does |
| 0009 | pymalloc arenas keep their pointer; `_Py_ALIGN_DOWN/UP` via the clang builtins | reading; tested natively as 0008 |
| 0010 | a dict's entries start on a pointer boundary | boot 2: "Unaligned cap access" in `insert_to_emptydict` |
| 0011 | no specialization: inline caches cannot hold a capability | boot 4: untagged pointer read from a cache in the eval loop |
| 0012 | dtoa's Bigints carved on a pointer boundary | boot 20: "Unaligned cap access" in `_PyDtoa_Fini` |
| 0013 | the parser/compiler arena (`pyarena.c`) aligns to a pointer, not 8 | boot 24: "Unaligned cap access" in `_PyPegen_update_memo` |

0010, 0012 and 0013 are one class: an allocator that aligns to 8 because that was enough for a
pointer. A grep for other 8-byte alignment constants in
`Python/`, `Objects/`, `Parser/`, `Include/` and `Modules/` found none left.

## Runtime and port requirements

| ID | item | status |
|---|---|---|
| R1 | a domain larger than 4 MiB | **done**: CMA-backed blocks (buildroot branch above); boots use a 128 MiB block |
| R2 | directory listing | **done**: `DIR_READ` (`runtime/r2-dir-read`, merged here); the boots import from `lib/python313.zip`, which needs no listing |
| R3 | obmalloc without `mmap` | **done**: `ac_cv_func_mmap=no`, gated |
| R4 | `rt_sigaction` refused | reached: 12 `rt_sigaction` and 2 `rt_sigprocmask` calls at startup, all refused; startup continues. Ctrl-C and signal handlers do not exist in a domain |
| R5 | pointer <-> int | **done**: patch 0007 |
| R6 | the round-trip sites | GC and pymalloc done (0008/0009); 42 sites left, none on the startup path read so far |
| R7 | `getcwd` | not served; `domain_entry.c` gives an absolute `argv[0]` so `getpath` does not need it |
| R8 | large file reads | **done**: bounce buffer (branch above); before it a read of more than a few hundred bytes failed with `EFAULT`, and zipimport's first read of the stdlib zip (526 entries, 9.9 MB) is 65 KB |
| R9 | `sys.stdout`/`stderr` | **done**: `stdio-fstat` (branch above); CPython set them to `None` when `fstat(1)` failed |
| R10 | `getrandom`, `readlinkat` | refused at startup; startup continues. The hash seed comes from CPython's fallback, `/dev/urandom` through the hostcall file service: without either, startup ends in a fatal error |
| R11 | a domain's memory back after it ends | open: the module never frees a domain's block, so a boot holds one 128 MiB CMA block per CPython run; the run script sizes `cma=` for it |

## Boots (QEMU, `run-cpython-domain.sh`, 2026-09-23)

| boot | image | result |
|---|---|---|
| 1 | patches 0001-0009 | halted, cause 24, in `Py_InitializeFromConfig`: QEMU re-fabricated gp as a LINEAR capability at every `cjalr`, and the compiler's `movc` of a function pointer moved it. `CAPSTONE_GP_NONLIN=1` (QEMU's own documented switch) fixes it |
| 2 | same, with that switch | halted, cause 4, in `insert_to_emptydict` → patch 0010 |
| 3, 6, 10, 14 | | the guest stalled before the domain started or at the module swap, vCPU at 100 %; the run script now copies the image to tmpfs first and can use a module already in the rootfs (`CPY_MODULE_IN_ROOTFS=1`), and retries |
| 4 | + 0010 | halted, cause 24, in the eval loop, on a pointer read from an inline cache → patch 0011 |
| 5 | + 0011 | **ran the frozen `getpath` in bytecode**; it failed on `getcwd` ("failed to make path absolute") → absolute `argv[0]` |
| 7-18 | + absolute `argv[0]`, some with `-v` or probes | halted, cause 24, on the fatal-error path, reading an untagged `PyStatus.err_msg` (see below), so the error itself was never printed. With `-v` (boot 18) the last import before it is `zipimport`; the error was zipimport's 65 KB read failing with `EFAULT` → bounce buffer. Boots 9 and 15-17 were lost to the runner taking `-v`'s "# " for the prompt |
| 19 | + bounce buffer | halted, cause 24, `cincoffsetimm` of NULL in `_PyArg_UnpackKeywordsWithVararg` → C-58 |
| 20 | + C-58 compiler | halted, cause 4, in `_PyDtoa_Fini` → patch 0012 |
| 21 | + 0012 | **exit status 0**, but nothing printed: `sys.stdout` was `None` because `fstat(1)` failed → `runtime/stdio-fstat` |
| 23 | + stdio-fstat | **`hello.py` runs**: prints the version, `2 ** 100`, integer and float arithmetic, a 1000-entry dict; status 0 |
| 24 | `checks.py` | halted, cause 4, in `_PyPegen_update_memo`, storing into a 64-byte arena allocation at an 8-aligned address → patch 0013 |
| 25 | + 0013 | lost, three attempts of three, no domain output. The runner had matched the "# " in the echo of the script's own `sed 's/# /...'` and typed its exit-code probe early (fixed: `#[ ]`, and a command containing "# " is refused); whether that is why the domain printed nothing is not established, see boot 27 |
| 26 | same image | **`checks.py` 6/6**: GC of 200 reference cycles with weakref callbacks, 20 rounds of 5000 tuples of pymalloc churn, `id()`, exceptions and generators, `json`/`re`/`struct`, `math.factorial(200)`; status 0, 10 s |
| 27 | same image, same script | **hung**: the domain started and printed nothing for 9 minutes, QEMU at 100 %; stopped by hand. Not reproduced since, see below |
| 29, 30, 31 | | 29: the run script died of SIGPIPE in its vermagic check (fixed). 30, 31: the guest stalled in its own boot (at 0.6 s, and after init); both had the monitor socket, 31 was sampled: kernel code at varying PCs, no `wfi`, `mip` 0 |
| 32, 33 | batches | lost to the batch harness: a typed command cut at 1022 characters, then one 128 MiB block per run from a 256 MiB CMA area; both fixed (a run script on the share; `cma=` by run count) |
| 34 | 8 runs in one boot | **8/8** `checks.py` 6/6 under `PYTHONHASHSEED=0..5` and two random seeds |
| 35 | 12 runs | **12/12**, random seeds, buffered and unbuffered output alternating |
| 36 | 12 runs of boot 25-27's image, rebuilt byte for byte | **12/12** |

`checks.py` passes natively 6/6 too; it is kept outside the tree with `hello.py`, as scripts the
run takes as an argument.

**Batches.** `run-cpython-domain.sh` runs several images or environments in one boot
(`CPY_DOM`, `CPY_ENV`); the environment goes through a file the domain reads at start, not into
the image, because a changed string moves the whole link (44743 bytes for one digit). A run is
about 10 s and a boot a minute or more, so variants belong in one boot. The domain hashes strings
with FNV, not SipHash: configure found aligned access required.

**Not explained: the boot-27 hang.** One run in 34 of the same `checks.py` hung with no output
(boot 25 may be a second). It did not recur in 33 runs across the image before and after
`domain_entry.c` changed, fixed and random hash seeds, buffered and unbuffered. With
`PYTHONUNBUFFERED=1` a recurrence would name the check it hung in; with the monitor socket it can
be sampled. Related, also unexplained: **the guest clock freezes** during batches. In boot 35 every
run begins and ends at the same second though each takes ~10 s; in boot 36 the clock stopped after
two runs; in boot 34 it ran through seven. While it is frozen, `lt.user`'s `alarm()` limit cannot
fire, which would explain why boot 27 was never cut off at its 900 s limit.

**Not explained: the untagged `err_msg`.** On the fatal path (boots 7-18) `Py_ExitStatusException`
read `PyStatus.err_msg` without a tag. Struct copies were checked and keep tags (`ldc`/`stc`); a
yield across the hostcall keeps them (small and large tests); the tag map holds 2^20 capabilities;
a QEMU tag watch on the message's slot saw no store clear it. With probes compiled in, the fault
moved and then disappeared, so it depends on layout. It is off the path now that startup
succeeds, and not understood.

## Re-run on dev (2026-09-24, branch `cpython/8-reintegration`)

The port alone, on `dev` `7ee228cdb67b` with the merged fixes, dev's compiler (`a378789289cd`, no
compiler source changes to the tip) and dev's runtime: survey 250 of 250 compiled, link 0 undefined
symbols (`results/*-2026-09-24-a378789289cd.txt`); boots `hello.py` (status 0, 122 rounds),
`startup.py` (0, 121), `checks.py` (0, 584, `CPY-CHECKS 6/6`), `diag.py` (0, 125) -- one boot each,
the module of `domain/1-cma-large-domains` in the `br-snap` rootfs, as the histogram run below.

## Where the rounds go (2026-09-24)

`lt.user` now prints, after its `LT-RESULT` line, one `LT-HIST` line for each HostCall opcode (rounds,
bytes requested and moved, rounds that reached the end of the payload region) and one `LT-FILE`
line per opened path. The first `LT-HIST` line checks its total against `LT-RESULT`'s round count
(`MATCH`); `WRITE_STDOUT`'s bytes were checked against the output (`hello.py`: 149 = the 89 bytes
of its three lines + the 60 of the domain's `UNSERVED` line). Measured on the boot-36 image, one
boot per script; a run of `pass` alone is 121 rounds, `hello.py` adds one `WRITE_STDOUT`.

| | `hello.py` | `checks.py` |
|---|---:|---:|
| rounds | 122 | 584 |
| `FILE_READ` (of them at the 4064-byte chunk ceiling) | 69 (43) | 403 (275) |
| `FILE_OPEN` / `FILE_STAT_BASIC` / `FILE_CLOSE` | 21 / 15 / 14 | 65 / 57 / 56 |
| `WRITE_STDOUT`, `CLOCK_GETTIME` | 2, 1 | 2, 1 |
| bytes read | 157 KB | 1.11 MB |
| of it `python313.zip` | 9 opens, 62 reads, 157 KB | 51 opens, 394 reads, 1.107 MB |

Three things follow. **Every round is an import.** After the last import the interpreter makes
no HostCall: the two `WRITE_STDOUT` rounds and the one `CLOCK_GETTIME` are the only ones that are
not the loading of a module, and the eval loop never crosses the boundary. **The stdlib zip is
the path.** 51 of 65 opens, 394 of 403 reads and 1.107 of 1.113 MB are `python313.zip`, and with
its `fstat` and `close` about 550 of the 584 rounds; `zipimport` opens the archive again for every
module it loads (`Lib/zipimport.py:617`, `_get_data`), three rounds before a byte of data. **A
4 KiB read is two rounds.** The chunk is 4064 bytes, the region less the 32-byte request header
(`hostcall.c`, `max_chunk = HC_PAYLOAD_SIZE - data_off`), and CPython reads in 4096-byte pieces;
275 of the 403 read rounds sit at that ceiling.

So the boundary's cost for CPython is at startup, and the change that removes most of it is in the
runtime, not in the protocol or the monitor: the zip shared once at start as a read-only region and
served by `hostcall.c` as a memory-backed file, so that `open`, `fstat`, `close` and `read` on that
path make no round. Not measured: what one round costs on silicon. The copying alone is 272 chunks
of 4 KiB at the ~14,000 cycles `hostcall.c` cites for CVA6, about 3.8 M cycles per `checks.py`
start; on QEMU the rounds are not where the ~10 s of a run go.

## Next action

Port: drop `-S` (import `site`), then run a larger part of the stdlib test suite in a domain, one
module per run, batched into one boot (`CPY_ENV`/`CPY_DOM`). Runtime: `rt_sigaction` (R4), `getrandom` and
`readlinkat` (R10), `getcwd` (R7); once a round's cost on silicon is known, the zip as a region
(above). Compiler: C-57, IR-level GEP speculation (C-58's residual),
C-53, C-56's toolchain side, then B3 (capability TLS) for threads. The `err_msg` fault above if it
comes back.
