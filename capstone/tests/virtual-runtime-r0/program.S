/* Copied to an ordinary executable Linux mapping; no physical addresses. */
.option norvc
.text
.globl r0_program, r0_fault, r0_ecall, r0_program_end
r0_program:
    li t0, 0x123
    sd t0, 0(s2)
    sd t0, -32(sp)
    sd t0, 0(tp)
    li t1, 4096
    .insn r 0x5b, 1, 0x0c, s2, s2, t1
r0_fault:
    sd t0, 0(s2)
    li a0, 42
    .insn s 0x5b, 4, s2, -16(sp)
r0_ecall:
    ecall
r0_program_end:

.globl r0_stale, r0_stale_end
r0_stale:
    .insn i 0x5b, 3, t2, sp, -16
    ld t0, 0(t2)
    ecall
r0_stale_end:
