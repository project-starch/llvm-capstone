/* Diagnostic: regs.S's setup only. Exit status = -errno of SUBLET_NR_TEST_ENTER, 200 when it
 * returned 0 and a0 is a reference (LCC type 1), 201 when it returned >= 0 without one, 90 when mmap failed. */
#define LCC(rd, src, sel) .insn r 0x5b, 1, 4, rd, src, sel
    .text
    .globl _start
_start:
    li a0, 0; li a1, 0x10000; li a2, 3; li a3, 0x22; li a4, -1; li a5, 0; li a7, 222; ecall
    bltz a0, mfail
    mv s0, a0
    li a0, 0; li a1, 0x10000; li a2, 3; li a3, 0x22; li a4, -1; li a5, 0; li a7, 222; ecall
    bltz a0, mfail
    mv s1, a0
    mv a0, s0; li a1, 0x10000; mv a2, s1; li a3, 0x10000
    li a7, 0x5b100
    ecall
    bltz a0, neg
    LCC(t0, a0, x1)
    li t1, 1
    li a0, 201
    bne t0, t1, out
    li a0, 200
    j out
neg: sub a0, zero, a0
    j out
mfail: li a0, 90
out: li a7, 93
    ecall
