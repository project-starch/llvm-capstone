#if defined(__riscv) && __riscv_xlen == 64
/* long capstone_stub_syscall(long nr, const long args[6], const _Atomic uint64_t *head,
 *                            const uint64_t *tail, int can_publish);
 *
 * One Linux system call with a checkable program counter. Between the ring
 * check and the ecall a signal's trampoline finds the pc inside
 * [capstone_stub_begin, capstone_stub_ecall] and moves it to
 * capstone_stub_retry, so the call is not made; the same range catches the
 * kernel's own restart (epc rewound to the ecall, a0 restored). A pc at
 * capstone_stub_after with a0 == -EINTR is a call the signal interrupted.
 */
    .text
    .globl capstone_stub_syscall
    .type capstone_stub_syscall, %function
    .globl capstone_stub_begin
    .globl capstone_stub_ecall
    .globl capstone_stub_after
    .globl capstone_stub_retry
capstone_stub_syscall:
    mv a7, a0
capstone_stub_begin:
    beqz a4, 1f
    ld t0, 0(a2)
    ld t1, 0(a3)
    bne t0, t1, capstone_stub_retry
1:
    ld a5, 40(a1)
    ld a4, 32(a1)
    ld a3, 24(a1)
    ld a2, 16(a1)
    ld a0, 0(a1)
    ld a1, 8(a1)
capstone_stub_ecall:
    ecall
capstone_stub_after:
    ret
capstone_stub_retry:
    li a0, -0x10000
    ret
    .size capstone_stub_syscall, .-capstone_stub_syscall
#endif

#if defined(__ELF__) && defined(__linux__)
    .section .note.GNU-stack,"",%progbits
#endif
