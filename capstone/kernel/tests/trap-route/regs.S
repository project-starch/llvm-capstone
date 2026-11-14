/* trap-route test program (plan v5 step 3), freestanding, RV64.
 *
 * Exit status: 0 when every register kept its reference; n when register xn came back
 * without its tag or with another lifetime; 100 + c when the test hook's reference faulted
 * with cause c; 90 to 95 when the setup failed. a0 carries getpid's result and a7 the call
 * number, so those two are not checked. */
#define LDC(rd, off, base) .insn i 0x5b, 3, rd, base, off
#define STC(src, off, base) .insn s 0x5b, 4, src, off(base)
#define LCC(rd, src, sel) .insn r 0x5b, 1, 4, rd, src, sel
#define INC(rd, src, off) .insn r 0x5b, 1, 0x0c, rd, src, off
#define SUBLET_NR_TEST_ENTER 0x5b100
#define SUBLET_NR_TEST_HOLD 0x5b101
#define MAGIC 0x5ab1e7c0ffee
#define SPIN 40000000

    .text
    .globl _start
_start:
    /* Two anonymous mappings of 64 KiB: A, whose references fill the registers, and B, which
     * holds the dump, the golden id and the test hook's two pages. */
    li a0, 0
    li a1, 0x10000
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    li a7, 222
    ecall
    bltz a0, setup_failed
    mv s0, a0
    li a0, 0
    li a1, 0x10000
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    li a7, 222
    ecall
    bltz a0, setup_failed
    mv s1, a0
    /* The test loader: a0 and a1 come back as references to A and B. */
    mv a0, s0
    li a1, 0x10000
    mv a2, s1
    li a3, 0x10000
    li a7, SUBLET_NR_TEST_ENTER
    ecall
    LCC(t0, a0, x1)
    li t1, 1
    bne t0, t1, not_protected
    /* From here on the program is protected and touches memory only through references. */
    LCC(t0, a0, x8)
    sd t0, 0(a1)
    li t0, 0x2000
    INC(t1, a1, t0)
    li t2, MAGIC
    sd t2, 0(t1)
    /* Every register but a0, a1 and a7: a reference to A with its own cursor. */
    li a7, 16
    INC(x1, a0, a7)
    li a7, 32
    INC(x2, a0, a7)
    li a7, 48
    INC(x3, a0, a7)
    li a7, 64
    INC(x4, a0, a7)
    li a7, 80
    INC(x5, a0, a7)
    li a7, 96
    INC(x6, a0, a7)
    li a7, 112
    INC(x7, a0, a7)
    li a7, 128
    INC(x8, a0, a7)
    li a7, 144
    INC(x9, a0, a7)
    li a7, 192
    INC(x12, a0, a7)
    li a7, 208
    INC(x13, a0, a7)
    li a7, 224
    INC(x14, a0, a7)
    li a7, 240
    INC(x15, a0, a7)
    li a7, 256
    INC(x16, a0, a7)
    li a7, 288
    INC(x18, a0, a7)
    li a7, 304
    INC(x19, a0, a7)
    li a7, 320
    INC(x20, a0, a7)
    li a7, 336
    INC(x21, a0, a7)
    li a7, 352
    INC(x22, a0, a7)
    li a7, 368
    INC(x23, a0, a7)
    li a7, 384
    INC(x24, a0, a7)
    li a7, 400
    INC(x25, a0, a7)
    li a7, 416
    INC(x26, a0, a7)
    li a7, 432
    INC(x27, a0, a7)
    li a7, 448
    INC(x28, a0, a7)
    li a7, 464
    INC(x29, a0, a7)
    li a7, 480
    INC(x30, a0, a7)
    li a7, 496
    INC(x31, a0, a7)
    /* getpid */
    li a7, 172
    ecall
    /* timer ticks */
    li a0, SPIN
1:  addi a0, a0, -1
    bnez a0, 1b
    /* a page fault from U, on a page of B never touched */
    li a7, 0x8000
    INC(a0, a1, a7)
    sd zero, 0(a0)
    /* the test hook: a0 references B + 0x2000, whose next page is untouched */
    li a7, 0x2000
    INC(a0, a1, a7)
    li a7, SUBLET_NR_TEST_HOLD
    ecall
    li a7, MAGIC
    bne a0, a7, hook_failed
    /* timer ticks again */
    li a0, SPIN
1:  addi a0, a0, -1
    bnez a0, 1b
    /* Dump every checked register with its tag, then check each: tagged, A's lifetime. */
    STC(x1, 16, a1)
    STC(x2, 32, a1)
    STC(x3, 48, a1)
    STC(x4, 64, a1)
    STC(x5, 80, a1)
    STC(x6, 96, a1)
    STC(x7, 112, a1)
    STC(x8, 128, a1)
    STC(x9, 144, a1)
    STC(x11, 176, a1)
    STC(x12, 192, a1)
    STC(x13, 208, a1)
    STC(x14, 224, a1)
    STC(x15, 240, a1)
    STC(x16, 256, a1)
    STC(x18, 288, a1)
    STC(x19, 304, a1)
    STC(x20, 320, a1)
    STC(x21, 336, a1)
    STC(x22, 352, a1)
    STC(x23, 368, a1)
    STC(x24, 384, a1)
    STC(x25, 400, a1)
    STC(x26, 416, a1)
    STC(x27, 432, a1)
    STC(x28, 448, a1)
    STC(x29, 464, a1)
    STC(x30, 480, a1)
    STC(x31, 496, a1)
    ld t2, 0(a1)
    li t3, 1
    LDC(t0, 16, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_1
    LCC(t1, t0, x8)
    bne t1, t2, lost_1
    LDC(t0, 32, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_2
    LCC(t1, t0, x8)
    bne t1, t2, lost_2
    LDC(t0, 48, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_3
    LCC(t1, t0, x8)
    bne t1, t2, lost_3
    LDC(t0, 64, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_4
    LCC(t1, t0, x8)
    bne t1, t2, lost_4
    LDC(t0, 80, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_5
    LCC(t1, t0, x8)
    bne t1, t2, lost_5
    LDC(t0, 96, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_6
    LCC(t1, t0, x8)
    bne t1, t2, lost_6
    LDC(t0, 112, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_7
    LCC(t1, t0, x8)
    bne t1, t2, lost_7
    LDC(t0, 128, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_8
    LCC(t1, t0, x8)
    bne t1, t2, lost_8
    LDC(t0, 144, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_9
    LCC(t1, t0, x8)
    bne t1, t2, lost_9
    LDC(t0, 176, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_11
    LCC(t1, t0, x8)
    bne t1, t2, lost_11
    LDC(t0, 192, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_12
    LCC(t1, t0, x8)
    bne t1, t2, lost_12
    LDC(t0, 208, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_13
    LCC(t1, t0, x8)
    bne t1, t2, lost_13
    LDC(t0, 224, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_14
    LCC(t1, t0, x8)
    bne t1, t2, lost_14
    LDC(t0, 240, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_15
    LCC(t1, t0, x8)
    bne t1, t2, lost_15
    LDC(t0, 256, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_16
    LCC(t1, t0, x8)
    bne t1, t2, lost_16
    LDC(t0, 288, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_18
    LCC(t1, t0, x8)
    bne t1, t2, lost_18
    LDC(t0, 304, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_19
    LCC(t1, t0, x8)
    bne t1, t2, lost_19
    LDC(t0, 320, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_20
    LCC(t1, t0, x8)
    bne t1, t2, lost_20
    LDC(t0, 336, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_21
    LCC(t1, t0, x8)
    bne t1, t2, lost_21
    LDC(t0, 352, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_22
    LCC(t1, t0, x8)
    bne t1, t2, lost_22
    LDC(t0, 368, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_23
    LCC(t1, t0, x8)
    bne t1, t2, lost_23
    LDC(t0, 384, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_24
    LCC(t1, t0, x8)
    bne t1, t2, lost_24
    LDC(t0, 400, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_25
    LCC(t1, t0, x8)
    bne t1, t2, lost_25
    LDC(t0, 416, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_26
    LCC(t1, t0, x8)
    bne t1, t2, lost_26
    LDC(t0, 432, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_27
    LCC(t1, t0, x8)
    bne t1, t2, lost_27
    LDC(t0, 448, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_28
    LCC(t1, t0, x8)
    bne t1, t2, lost_28
    LDC(t0, 464, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_29
    LCC(t1, t0, x8)
    bne t1, t2, lost_29
    LDC(t0, 480, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_30
    LCC(t1, t0, x8)
    bne t1, t2, lost_30
    LDC(t0, 496, a1)
    LCC(t1, t0, x1)
    bne t1, t3, lost_31
    LCC(t1, t0, x8)
    bne t1, t2, lost_31
    li a0, 0
    j finish
lost_1:
    li a0, 1
    j finish
lost_2:
    li a0, 2
    j finish
lost_3:
    li a0, 3
    j finish
lost_4:
    li a0, 4
    j finish
lost_5:
    li a0, 5
    j finish
lost_6:
    li a0, 6
    j finish
lost_7:
    li a0, 7
    j finish
lost_8:
    li a0, 8
    j finish
lost_9:
    li a0, 9
    j finish
lost_11:
    li a0, 11
    j finish
lost_12:
    li a0, 12
    j finish
lost_13:
    li a0, 13
    j finish
lost_14:
    li a0, 14
    j finish
lost_15:
    li a0, 15
    j finish
lost_16:
    li a0, 16
    j finish
lost_18:
    li a0, 18
    j finish
lost_19:
    li a0, 19
    j finish
lost_20:
    li a0, 20
    j finish
lost_21:
    li a0, 21
    j finish
lost_22:
    li a0, 22
    j finish
lost_23:
    li a0, 23
    j finish
lost_24:
    li a0, 24
    j finish
lost_25:
    li a0, 25
    j finish
lost_26:
    li a0, 26
    j finish
lost_27:
    li a0, 27
    j finish
lost_28:
    li a0, 28
    j finish
lost_29:
    li a0, 29
    j finish
lost_30:
    li a0, 30
    j finish
lost_31:
    li a0, 31
    j finish
hook_failed:
    bgez a0, 1f
    neg a0, a0
    addi a0, a0, 100
    j finish
1:  li a0, 95
    j finish
setup_failed:
    li a0, 90
    j finish
not_protected:
    li a0, 91
finish:
    li a7, 93
    ecall
