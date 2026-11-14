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
