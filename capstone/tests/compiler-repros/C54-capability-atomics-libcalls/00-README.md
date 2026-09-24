# C-54 — an atomic operation on a pointer compiles to `__atomic_*_16`, which nothing provides

**Fixed** by `c7f0de349b4b`, the commit before this folder on its branch. Measured 2026-09-23: `./run.sh` reports ABSENT with that compiler and PRESENT with `dev`'s (`1a08706b6344`, whose compiler sources are those of `d030df93d4a4`). Regression test: `llvm/test/CodeGen/Capstone/atomic-capability-value-libcall.ll` and `clang/test/CodeGen/capstone-atomic-pointer.c`. In QEMU, that tags survive: `capstone/tests/runtime-qemu/capability-atomics/run.sh`. Everything below is the investigation as it was recorded before the fix.

**A COMPILER gap, recorded because a port now reaches it.** Found 2026-09-23 by the first link of
the CPython interpreter (`capstone/ports/cpython/interpreter/`): after everything the survey
could compile was linked, the only undefined symbols not explained by an absent CPython object
were `__atomic_compare_exchange_16`, `__atomic_load_16` and `__atomic_store_16`, from
`Python/getargs.c` and `Modules/signalmodule.c`. Siblings: C-50 to C-53.

## Reproducer

    void *f(void **p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }

    warning: large atomic operation may incur significant performance penalty; the access
             size (16 bytes) exceeds the max lock-free size (8 bytes) [-Watomic-alignment]
    -> an undefined reference to __atomic_load_16

`./run.sh` compiles load, store, compare-exchange and exchange on a `void **` and lists each
object's undefined symbols; the same four on a `long` are the control and must need none.

| | `d030df93d4a4` (dev) | `d5b5f11cae8f` | `f7b50f081ca4` |
|---|---|---|---|
| pointer load/store/cas/xchg | `__atomic_*_16` each | same | same |
| `long` control | inline | inline | does not compile (predates `d5b5f11cae8f`) |

## What it is

A capability is 16 bytes and the target declares 8 as its largest lock-free atomic, so clang
lowers every capability-valued atomic to the generic `__atomic_*_16` library call. There is no
libatomic for capstone64 and compiler-rt's builtins do not define these. `d5b5f11cae8f`, which
added capability-ADDRESS atomics for 32 and 64 bits, states the edge: "Subword and
capability-valued atomics are outside this change." C-51 is the subword half; this is the other.

The ISA loads and stores capabilities (`ldc`/`stc`); whether those are single-copy atomic, and
whether a capability compare-exchange is expressible at all, was not checked here and is a design
question for the compiler lane.

## Impact on CPython, and a possible interim

CPython's `_Py_atomic_*_ptr` family (Include/cpython/pyatomic.h) uses these in the linked
objects above and, by source count, more in the objects C-51 still keeps from compiling
(`Python/ceval_gil.c`, `Objects/typeobject.c`, `Python/pystate.c`, `Objects/unicodeobject.c`).
A domain runs on one hart with no clone, so a runtime providing `__atomic_*_16` as plain
capability-preserving loads, stores and compares would be correct THERE; it would not be for any
build that can start a thread. Not implemented.
