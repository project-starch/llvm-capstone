/* Capstone pure-capability syscall layer for musl.
 *
 * Overlaid onto a copy of musl's arch/riscv64 (see prepare-musl-capstone.sh).
 * Only this one file differs; everything else in arch/capstone64 is upstream
 * riscv64 verbatim, so the diff a reader has to audit is this file.
 *
 * TWO deviations from arch/riscv64, both forced by the capability ABI:
 *
 * 1. `syscall_arg_t` is capability-width, not `long`.
 *
 *    Upstream marshals every syscall argument through `((long)(X))`
 *    (src/internal/syscall.h:22). Under -target capstone64 a pointer IS a
 *    capability, so that cast destroys it, and the backend then has to
 *    reconstitute one from an integer -- which it refuses, correctly:
 *
 *      fatal error: error in backend: Capstone PureCap: Cannot materialize
 *      arbitrary >64-bit constants as capabilities; capabilities are unforgeable
 *
 *    That single cast accounted for 27 of the 118 files that failed to compile
 *    in the 2026-08-14 survey (fopen, fstatat, mkdir, lchown, sem_timedwait,
 *    ...). musl anticipates the override: its definition is guarded by
 *    `#ifndef __scc`, so defining __scc here suppresses both the macro and the
 *    `typedef long syscall_arg_t`, and pointers reach the boundary intact.
 *
 * 2. No `ecall`.
 *
 *    A domain cannot trap to Linux: its caller is a user process, not a kernel,
 *    and the trap vector belongs to the monitor. So the boundary is an ordinary
 *    call to __capstone_hostcall(), which the hostcall implementation provides.
 *    Keeping it an extern call (rather than inline asm) is what lets the whole
 *    libc compile before any hostcall transport exists.
 *
 * NOTE for whoever writes __capstone_hostcall: `a`..`f` arrive as capabilities
 * for pointer arguments and as integers widened to capability width otherwise.
 * The callee must not assume it can tell them apart from the value alone; the
 * syscall number `n` is what says which arguments are pointers.
 */
#define __SYSCALL_LL_E(x) (x)
#define __SYSCALL_LL_O(x) (x)

typedef __UINTPTR_TYPE__ syscall_arg_t;
#define __scc(X) ((syscall_arg_t)(X))

long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                         syscall_arg_t c, syscall_arg_t d,
                         syscall_arg_t e, syscall_arg_t f);

static inline long __syscall0(long n)
{ return __capstone_hostcall(n, 0, 0, 0, 0, 0, 0); }

static inline long __syscall1(long n, syscall_arg_t a)
{ return __capstone_hostcall(n, a, 0, 0, 0, 0, 0); }

static inline long __syscall2(long n, syscall_arg_t a, syscall_arg_t b)
{ return __capstone_hostcall(n, a, b, 0, 0, 0, 0); }

static inline long __syscall3(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c)
{ return __capstone_hostcall(n, a, b, c, 0, 0, 0); }

static inline long __syscall4(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d)
{ return __capstone_hostcall(n, a, b, c, d, 0, 0); }

static inline long __syscall5(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e)
{ return __capstone_hostcall(n, a, b, c, d, e, 0); }

static inline long __syscall6(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e, syscall_arg_t f)
{ return __capstone_hostcall(n, a, b, c, d, e, f); }

/* No vDSO: there is no kernel-provided page in a domain's address space. */

#define IPC_64 0
