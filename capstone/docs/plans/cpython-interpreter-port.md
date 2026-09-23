# CPython interpreter port — what stands between here and a running interpreter

Branch `cpython/3-full-port`. Component `capstone/ports/cpython/interpreter/`
([README](../../ports/cpython/interpreter/README.md)). Written 2026-09-23 against clang
`d030df93d4a4` (= `origin/dev`); every "measured" below is from that compiler and the results
files in the component's `results/`.

## Where it stands

| step | result |
|---|---|
| compile survey | 222 of 253 objects (upstream as released: 27); six header patches |
| first link | 206 objects + 3 archives; 388 undefined symbols, 385 from the absent objects, 3 are C-54 |
| image size | 8.37 MiB without the 29 absent objects; ~11.5 MiB whole (estimate); a domain gets 4 MiB |
| runtime | nothing has run; 53 linked syscalls are unserved, four matter at startup (see R-items) |

Owners: **compiler** = the compiler lane (`llvm/`); **port** = whoever holds this branch;
**runtime** = domain runtime / hostcall / kernel module; **lead** = the project lead's decision.

## Compiler requirements

### A. For a complete link

| ID | feature | today | unblocks | owner | evidence |
|---|---|---|---|---|---|
| A1 | lower `llvm.ptrmask` on a capability: mask the address, keep the metadata | **C-51**, crashes isel | every 8/16-bit atomic; `PyMutex`; 26 objects; `__builtin_align_down` | compiler | `tests/compiler-repros/C51-ptrmask-on-capability/` |
| A2 | lock-free atomics on capabilities (load, store, exchange, compare-exchange at 16 bytes) | **C-54**, become `__atomic_*_16` calls nothing provides | `_Py_atomic_*_ptr`; 3 symbols now, more once A1 lands | compiler (design) | `tests/compiler-repros/C54-capability-atomics-libcalls/` |
| A3 | Greedy register allocator on `compiler_visit_stmt` | **C-52**, SIGSEGV in `SplitEditor` | `Python/compile.c` without `-regalloc=basic` | compiler | `tests/compiler-repros/C52-greedy-regalloc-segfault/` (208 instructions) |
| A4 | Assignment Tracking at the index width | **C-50**, one-line candidate fix, not through lit/QEMU | `-g` with optimisation without the workaround | compiler | `tests/compiler-repros/C50-assignment-tracking-index-width/` |

Interim for A2, **port**: a domain has one hart and no clone, so the runtime could define
`__atomic_*_16` as plain capability-preserving loads, stores and compares. Correct only while
nothing can start a thread; not done.

### B. For the interpreter to run correctly

| ID | feature | why | owner |
|---|---|---|---|
| B1 | an integer type that carries a capability (`uintptr_t` of 16 bytes, as CHERI's `__uintcap_t`), with arithmetic that keeps the tag | the lever on the **62 integer→pointer sites** the survey counts, above all the GC's `_gc_next`/`_gc_prev` flag bits; without it each site is rewritten by hand. An ABI change that reaches musl | **lead** decides, compiler implements |
| B2 | capability builtins for C: get/set address, bounds, tag test, align | lets port patches keep provenance explicitly (GC: set a flag bit in the address, keep the capability). Today the runtime writes raw `.insn` encodings in inline asm; `BuiltinsCapstone.td` has none | compiler |
| B3 | capability TLS model | **C-47**; worked around by patch 0006 for a one-thread domain, needed for real threads | compiler (design) |
| B4 | inline-asm `"m"` input operands | **C-53**; blocks nothing today | compiler |

B1 and B2 are alternatives for the same problem, not a sequence: B1 keeps CPython's code, B2
changes it. Choosing is the first design decision this port needs.

### C. The 4 MiB wall, compiler share

| ID | feature | today | owner |
|---|---|---|---|
| C1 | working `-Os` / `-Oz` | **crash, not yet registered**: `Objects/dictobject.c` at `-Os` asserts in `LiveVariables` ("getVarInfo: not a virtual register") on `dict___contains__`; `-O3` compiles | compiler; reduce and register first |
| C2 | code density: capstone64 objects are 1.30× their x86-64 counterparts (measured over the 222) | where the 30 % goes is NOT measured; candidates: capability loads, cap-table indirection, soft-float calls | port measures, compiler acts |
| C3 | hardware-float ABI instead of soft-float | every float operation is a compiler-rt call | **lead** (does the silicon have an FPU?) |

The limit itself is not the compiler's: a domain is one contiguous kernel allocation capped by
`MAX_ORDER` (`ports/musl-capstone/README.md`, `sscanf_long`). The mruby port's tests stop at
image size too (commit `10322a2fb795`, by its subject), so this is not CPython's problem alone.

### D. Speed, later

| ID | feature | today |
|---|---|---|
| D1 | jump tables that work with capabilities | built with `-fno-jump-tables`; CPython's opcode `switch` becomes a compare chain |
| D2 | computed-goto label tables with capability-init records | known gap (`docs/history/05-08-2026_06-00-00_gp-captable-lua-bringup.md`); CPython configured `--without-computed-gotos` |

## Runtime and port requirements (not compiler)

| ID | item | owner | evidence |
|---|---|---|---|
| R1 | a domain larger than 4 MiB | **runtime/lead** | size above |
| R2 | `getdents64` in the hostcall service, or freeze / zip the modules imported at startup | runtime or port | native `strace -k`: 4 calls from `os_listdir` at startup |
| R3 | obmalloc arenas without `mmap`: configure with `HAVE_MMAP` answered no | port | 3 `mmap` calls from `_PyMem_ArenaAlloc` |
| R4 | startup with `rt_sigaction` refused (66 calls from `PyOS_getsig`/`setsig`) | port, measure first | not known whether CPython tolerates it |
| R5 | `PyLong_FromVoidPtr`/`AsVoidPtr` (and so `id()`) without an integer that holds a pointer | port decision, or B1 | the survey's MUST_FAIL control |
| R6 | the 62 round-trip sites, GC first | port, after B1/B2 | survey report |

## Order

1. **A1 + A2** — together they are what stands between the survey and a complete link.
2. **C1** — cheap to reduce, and size is the wall.
3. **B1 or B2** — a decision for the lead; it sets how much of CPython the port rewrites.
4. **R1** in parallel — without it nothing runs, whatever the compiler does.
5. Then R2-R4 and a first boot of the smallest interpreter that links.

## Next action

Port: reduce the `-Os` crash (C1) and register it. Compiler lane: A1 and A2, reproducers above.
Lead: B1 vs B2, and R1.
