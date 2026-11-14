/* Linux syscall ABI. MOVC preserves all six argument references.
 * Scalar calls return long. The reference-return alias is used by mmap so
 * no C integer return or cast discards the result tag. */
.text
.global __capstone_native_syscall
.global __capstone_linux_syscall_ref
.type __capstone_native_syscall, @function
.type __capstone_linux_syscall_ref, @function
__capstone_native_syscall:
__capstone_linux_syscall_ref:
    mv a7, a0
    movc a0, a1
    movc a1, a2
    movc a2, a3
    movc a3, a4
    movc a4, a5
    movc a5, a6
    ecall
    ret
.size __capstone_native_syscall, .-__capstone_native_syscall
.size __capstone_linux_syscall_ref, .-__capstone_linux_syscall_ref

#ifdef CAPSTONE_RUNTIME_TRAP
/* fn, stack, flags, arg, parent_tid, tls, child_tid. The stack frame contains
 * tagged copies, and Linux receives stack and TLS as reference arguments. */
.global __capstone_linux_clone
.type __capstone_linux_clone, @function
__capstone_linux_clone:
    cincoffsetimm a1, a1, -32
    stc a0, 0(a1)
    stc a3, 16(a1)
    mv a0, a2
    movc a2, a4
    movc a3, a5
    movc a4, a6
    li a7, 220
    ecall
    bnez a0, 1f
    ldc t0, 0(sp)
    ldc a0, 16(sp)
    cincoffsetimm sp, sp, 32
    jalr ra, 0(t0)
    li a7, 93
    ecall
    ebreak
1:  ret
.size __capstone_linux_clone, .-__capstone_linux_clone

.global __unmapself
.type __unmapself, @function
__unmapself:
    li a7, 215
    ecall
    li a0, 0
    li a7, 93
    ecall
    ebreak
.size __unmapself, .-__unmapself
#endif
