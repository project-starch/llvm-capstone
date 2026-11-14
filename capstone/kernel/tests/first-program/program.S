/* The trap route's first program (plan v5 step 4), a .dom image without the SDK.
 *
 * The loader enters it with references in sp (the stack, cursor at the top), gp (the image,
 * cursor at its base), a0 (metadata), a1 (exchange) and a2 (the startup block), and every
 * other register an integer 0. MODE selects what it does:
 *   0 (ok)     checks those registers and its own data through gp, calls getpid through the
 *              selector ABI and exits with 42, or with 10 + the number of the failed check
 *   1 (stale)  derives a child of the metadata block, revokes it and loads through it: the
 *              kernel prints the fault line with cause 25 and the program dies with SIGSEGV
 *   2 (spin)   spins for a few seconds, then exits with 43, so a test can inspect it
 *   3 (bounds) loads one doubleword past the end of the metadata block: the kernel prints the
 *              fault line with cause 28 and the program dies with SIGSEGV */
#define LCC(rd, src, sel) .insn r 0x5b, 1, 4, rd, src, sel
#define INC(rd, src, off) .insn r 0x5b, 1, 0x0c, rd, src, off
#define DERIVE(rd, parent, size) .insn r 0x5b, 1, 0x51, rd, parent, size
#define KILL(obj) .insn r 0x5b, 1, 0x52, zero, obj, zero
#define SELECTOR 0x43534e31
#define CHECK_TAGGED(reg, n) LCC(t0, reg, x1); li t1, 1; li s11, n; bne t0, t1, fail

    .text
    .globl _start
_start:
    CHECK_TAGGED(sp, 1)
    CHECK_TAGGED(gp, 2)
    CHECK_TAGGED(a0, 3)
    CHECK_TAGGED(a1, 4)
    CHECK_TAGGED(a2, 5)
    /* a register the loader left alone is an integer 0 */
    li s11, 6
    bnez s1, fail
    /* the image's data, through gp */
    lui t0, %hi(__data_off)
    addi t0, t0, %lo(__data_off)
    INC(t1, gp, t0)
    ld t2, 0(t1)
    li t3, 0x5ab1e7da7a
    li s11, 7
    bne t2, t3, fail
    /* the stack, through sp */
    li t0, 0x77
    sd t0, -16(sp)
    ld t1, -16(sp)
    li s11, 8
    bne t0, t1, fail
#if MODE == 1
    li t0, 64
    DERIVE(t1, a0, t0)
    KILL(t1)
    ld t2, 0(t1)
    li a0, 99
    j finish
#endif
#if MODE == 3
    li t0, 16384
    INC(t1, a0, t0)
    ld t2, 0(t1)
    li a0, 98
    j finish
#endif
    /* getpid through the selector ABI: the number in a6, the selector in a7 */
    li a6, 172
    li a7, SELECTOR
    ecall
    li s11, 9
    blez a0, fail
#if MODE == 2
    li t0, 4000000000
1:  addi t0, t0, -1
    bnez t0, 1b
    li a0, 43
    j finish
#endif
    li a0, 42
    j finish
fail:
    addi a0, s11, 10
finish:
    li a7, 93
    ecall

    .data
    .balign 8
    .dword 0x5ab1e7da7a

    .section .capstone_virtual, "aw"
    .dword 0x374d56564e4f5043
    .section .capstone_application, "aw"
    .dword 0x315050414e4f5043, 2, 3, 65536, 0, 4096, 0
