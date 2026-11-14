/* The trap route's first program (plan v5 step 4), a .dom image without the SDK.
 *
 * The loader enters it with references in sp (the stack block, cursor at the exec stack of
 * the 16-byte pointer ABI: argc, argv cells, envp cells, auxv) and gp (the image, cursor at
 * its base), and every other register an integer 0. MODE selects what it does:
 *   0 (ok)     checks those registers, argc, argv[0] and its own data through gp, calls getpid
 *              and exits with 42, or with 10 + the number of the failed check
 *   1 (stale)  derives a child of the stack block, revokes it and loads through it: the
 *              kernel prints the fault line with cause 25 and the program dies with SIGSEGV
 *   2 (spin)   spins for a few seconds, then exits with 43, so a test can inspect it
 *   3 (bounds) loads the doubleword below the image block's base: the kernel prints the fault
 *              line with cause 28 and the program dies with SIGSEGV
 *   4 (map)    maps a page, which comes back as a reference, stores and loads through it,
 *              unmaps it through the reference and exits with 44; a failed munmap exits with
 *              100 + its error number
 *   5 (unmapped) the same, then loads through the reference it unmapped: the munmap of the
 *              block's last page revoked it, so the kernel prints the fault line with cause 25
 *   6 (signal) registers a handler for SIGUSR1 and sends itself SIGUSR1: the kernel builds no
 *              signal frame for a protected program and kills it with SIGUSR1 instead
 *   7 (gup)    asks the test hook to pin its stack page as direct I/O would: 45 when the
 *              kernel refuses (EFAULT), 46 when the page was pinned
 *   8 (exec)   execs /bin/busybox, an ordinary ELF: 108 when the kernel refuses (ENOEXEC);
 *              otherwise busybox runs unprotected and exits with its own status
 *   9 (sockopt) sets SO_ATTACH_FILTER, whose value holds a pointer, on a Unix socket: 47 when
 *              the binder refuses (ENOPROTOOPT) after an allowed SO_SNDBUF passed, else
 *              100 + the error number
 *  10 (argv)   loads the byte 32 bytes below argv[0], inside the stack block: the reference
 *              covers only the string, so the load faults with 28; 48 if the load passed */
#define LCC(rd, src, sel) .insn r 0x5b, 1, 4, rd, src, sel
#define INC(rd, src, off) .insn r 0x5b, 1, 0x0c, rd, src, off
#define DERIVE(rd, parent, size) .insn r 0x5b, 1, 0x51, rd, parent, size
#define KILL(obj) .insn r 0x5b, 1, 0x52, zero, obj, zero
#define LDC(rd, off, base) .insn i 0x5b, 3, rd, base, off
#define STC(src, off, base) .insn s 0x5b, 4, src, off(base)
#define SUBLET_NR_TEST_GUP 0x5b102
#define CHECK_TAGGED(reg, n) LCC(t0, reg, x1); li t1, 1; li s11, n; bne t0, t1, fail

    .text
    .globl _start
_start:
    CHECK_TAGGED(sp, 1)
    CHECK_TAGGED(gp, 2)
    /* a register the loader left alone is an integer 0 */
    li s11, 3
    bnez a0, fail
    li s11, 4
    bnez s1, fail
    /* argc, and argv[0] a reference to a string that starts with '.' */
    ld t0, 0(sp)
    li t1, 1
    li s11, 5
    bne t0, t1, fail
    LDC(t2, 16, sp)
    CHECK_TAGGED(t2, 6)
    lbu t0, 0(t2)
    li t1, 0x2e
    li s11, 6
    bne t0, t1, fail
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
    DERIVE(t1, sp, t0)
    KILL(t1)
    ld t2, 0(t1)
    li a0, 99
    j finish
#endif
#if MODE == 3
    ld t2, -8(gp)
    li a0, 98
    j finish
#endif
#if MODE == 4 || MODE == 5
    li a0, 0
    li a1, 4096
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    li a7, 222
    ecall
    CHECK_TAGGED(a0, 11)
    li t0, 0x77
    sd t0, 8(a0)
    ld t1, 8(a0)
    li s11, 12
    bne t0, t1, fail
    INC(s2, a0, zero)
    li a1, 4096
    li a7, 215
    ecall
    beqz a0, 1f
    neg a0, a0
    addi a0, a0, 100
    j finish
1:
#if MODE == 5
    ld t0, 8(s2)
    li a0, 97
    j finish
#endif
    li a0, 44
    j finish
#endif
#if MODE == 6
    li t0, -64
    INC(t1, sp, t0)
    li t2, 0x1234
    sd t2, 0(t1)
    sd zero, 8(t1)
    sd zero, 16(t1)
    li a0, 10
    INC(a1, t1, zero)
    li a2, 0
    li a3, 8
    li a7, 134
    ecall
    li s11, 13
    bnez a0, fail
    li a7, 172
    ecall
    li a1, 10
    li a7, 129
    ecall
    li a0, 96
    j finish
#endif
#if MODE == 7
    INC(a0, sp, zero)
    li a7, SUBLET_NR_TEST_GUP
    ecall
    li t0, -14
    li a1, 45
    beq a0, t0, 1f
    li a1, 46
1:  mv a0, a1
    j finish
#endif
#if MODE == 8
    /* "/bin/busybox" at sp - 64, argv = {path, NULL} at sp - 128, envp = argv + 16 */
    li t0, -64
    INC(t1, sp, t0)
    li t2, 0x7375622f6e69622f     /* "/bin/bus" */
    sd t2, 0(t1)
    li t2, 0x786f6279             /* "ybox" and the terminating zero */
    sd t2, 8(t1)
    li t0, -128
    INC(t2, sp, t0)
    STC(t1, 0, t2)
    sd zero, 16(t2)
    sd zero, 24(t2)
    sd zero, 32(t2)
    sd zero, 40(t2)
    INC(a0, t1, zero)
    INC(a1, t2, zero)
    li t0, 16
    INC(a2, t2, t0)
    li a7, 221
    ecall
    neg a0, a0
    addi a0, a0, 100
    j finish
#endif
#if MODE == 9
    li a0, 1
    li a1, 1
    li a2, 0
    li a7, 198
    ecall
    li s11, 14
    bltz a0, fail
    mv s2, a0
    /* SO_SNDBUF 8192 from sp - 32: allowed */
    li t0, -32
    INC(t1, sp, t0)
    li t2, 8192
    sd t2, 0(t1)
    sd zero, 8(t1)
    mv a0, s2
    li a1, 1
    li a2, 7
    INC(a3, t1, zero)
    li a4, 4
    li a7, 208
    ecall
    li s11, 15
    bnez a0, fail
    /* SO_ATTACH_FILTER with a 16-byte sock_fprog: refused */
    mv a0, s2
    li a1, 1
    li a2, 26
    INC(a3, t1, zero)
    li a4, 16
    li a7, 208
    ecall
    li t0, -92
    li a1, 47
    beq a0, t0, 1f
    neg a1, a0
    addi a1, a1, 100
1:  mv a0, a1
    j finish
#endif
#if MODE == 10
    LDC(t2, 16, sp)
    li t0, -32
    INC(t3, t2, t0)
    lbu t4, 0(t3)
    li a0, 48
    j finish
#endif
    /* getpid */
    li a7, 172
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

