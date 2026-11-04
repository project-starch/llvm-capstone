/* SPDX-License-Identifier: MIT */
#include <capstone/native-syscall-abi.h>
#if !defined(CAPSTONE_RUNTIME_VIRTUAL) || !defined(CAPSTONE_NATIVE_SYSCALL_ABI)
#error "native syscall stubs require explicit virtual native ABI selection"
#endif
/* long __capstone_native_syscall(long nr, void *a, void *b, void *c,
 *                               void *d, void *e, void *f);
 * Preserve all six argument tags. Only the number and selector are scalar.
 * The native kernel backend must return a scalar result with its tag cleared.
 * Capability-returning mapping services require a separate typed interface. */
.text
.global __capstone_native_syscall
.type __capstone_native_syscall, @function
__capstone_native_syscall:
    mv t0, a0
    movc a0, a1
    movc a1, a2
    movc a2, a3
    movc a3, a4
    movc a4, a5
    movc a5, a6
    mv a6, t0
    li a7, NS_SERVICE_NATIVE
    ecall
    ret
.size __capstone_native_syscall, .-__capstone_native_syscall
