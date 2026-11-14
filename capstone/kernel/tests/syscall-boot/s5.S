/* The system-call test program of plan v5 step 5, freestanding, RV64.
 *
 * It maps two 64 KiB regions A and B, fills A by plain stores, and becomes protected through
 * the test loader of step 3 (a0 and a1 come back as references to A and B). From then on it
 * passes references to system calls, which go through the binder and the routines for user
 * memory. Each check compares a result; the first mismatch exits with the check's number.
 * Exit 0: every check passed, the last one (C27) by exec of a shell that prints "S5: exec ok"
 * and exits 0. The bounds check (C20) needs the processor to check an S access through t6
 * (plan v5 step 1, rule 1); it does not decide the status but prints its outcome:
 * "S5: rule1 enforced", "S5: rule1 missing" or "S5: rule1 inconclusive". A reference moves
 * between registers by CINCOFFSET with offset zero: an integer instruction such as mv clears
 * the tag. */
#define LCC(rd, src, sel) .insn r 0x5b, 1, 4, rd, src, sel
#define INC(rd, src, off) .insn r 0x5b, 1, 0x0c, rd, src, off
#define SUBLET_NR_TEST_ENTER 0x5b100
#define STC(src, off, base) .insn s 0x5b, 4, src, off(base)
#define AT_FDCWD -100

/* rd = reference s2 (A) moved to offset off */
.macro refa rd, off
    li t0, \off
    INC(\rd, s2, t0)
.endm
.macro refb rd, off
    li t0, \off
    INC(\rd, s3, t0)
.endm
/* store an iovec {reference A + off, len} at A + at, through A's reference */
.macro iova at, off, len
    refa t2, \off
    refa t3, \at
    STC(t2, 0, t3)
    refa t3, \at + 16
    li t2, \len
    sd t2, 0(t3)
.endm
/* store the reference A + off at A + at, through A's reference */
.macro ptra at, off
    refa t2, \off
    refa t3, \at
    STC(t2, 0, t3)
.endm
/* fail with status n unless a0 == value */
.macro expect n, value
    li t0, \value
    li s8, \n
    bne a0, t0, fail
.endm
.macro sys nr
    li a7, \nr
    ecall
.endm
/* plain copy of len bytes from label to A + off, before protection */
.macro put off, label, len
    li t0, \off
    add t0, s0, t0
    la t1, \label
    li t2, \len
1:  lb t3, 0(t1)
    sb t3, 0(t0)
    addi t0, t0, 1
    addi t1, t1, 1
    addi t2, t2, -1
    bnez t2, 1b
.endm

    .text
    .globl _start
_start:
    li a0, 0
    li a1, 0x10000
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    sys 222
    li s8, 90
    bltz a0, fail
    mv s0, a0
    li a0, 0
    li a1, 0x10000
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    sys 222
    li s8, 91
    bltz a0, fail
    mv s1, a0

    put 0x000, msg_ok, 12
    put 0x100, path_null, 10
    put 0x120, path_file, 5
    put 0xeff6, path_zero, 10
    put 0x200, ones, 16
    put 0x300, ones, 16
    put 0x310, ones, 16
    put 0x400, msg_enforced, 20
    put 0x420, msg_missing, 20
    put 0x440, msg_inconclusive, 25
    put 0x8e0, path_sh, 8
    put 0x900, arg_sh, 3
    put 0x910, arg_c, 3
    put 0x920, arg_echo, 18
    li t0, 0x100
    add t0, s1, t0
    li t1, -1
    sd t1, 0(t0)
    sd t1, 8(t0)
    li t0, 0x210
    add t0, s0, t0
    li t1, 1
    sw t1, 0(t0)

    mv a0, s0
    li a1, 0x10000
    mv a2, s1
    li a3, 0x10000
    sys SUBLET_NR_TEST_ENTER
    LCC(t0, a0, x1)
    li t1, 1
    li s8, 92
    bne t0, t1, fail
    INC(s2, a0, zero)
    INC(s3, a1, zero)
    /* From here on the program touches memory only through references. */

    /* C1: write through a reference. */
    li a0, 1
    INC(a1, s2, zero)
    li a2, 12
    sys 64
    expect 1, 12
    /* C2: a forged integer that names a3's slot (number 4), with A's reference in a3. */
    li a0, 1
    li a1, 4
    slli a1, a1, 40
    or a1, a1, s0
    li a2, 12
    INC(a3, s2, zero)
    sys 64
    expect 2, -14
    /* C3: a plain address in a pointer position names no reference. */
    li a0, 1
    mv a1, s0
    li a2, 12
    sys 64
    expect 3, -14
    /* C4: a call outside the allow-list (ptrace). */
    li a0, 0
    li a1, 0
    sys 117
    expect 4, -38
    /* C5: munmap the last page but one of A, through the hand-added position. */
    refa a0, 0xf000
    li a1, 0x1000
    sys 215
    expect 5, 0
    /* C6: openat of a path that ends in the last byte before the hole. */
    li a0, AT_FDCWD
    refa a1, 0xeff6
    li a2, 0
    li a3, 0
    sys 56
    li s8, 6
    bltz a0, fail
    mv s4, a0
    /* C7: read 32 bytes of /dev/zero (clear_user), check both ends. */
    mv a0, s4
    refa a1, 0x300
    li a2, 32
    sys 63
    expect 7, 32
    refa t1, 0x300
    ld a0, 0(t1)
    expect 8, 0
    refa t1, 0x318
    ld a0, 0(t1)
    expect 9, 0
    /* C10: a read that runs into the hole returns the bytes before it (clear_user residual). */
    mv a0, s4
    refa a1, 0xeff0
    li a2, 32
    sys 63
    expect 10, 16
    /* C11: getrandom into the hole (copy_to_user residual). */
    refa a0, 0xeff0
    li a1, 32
    li a2, 0
    sys 278
    expect 11, 16
    /* C12: a file write from the hole (copy_from_user residual). */
    li a0, AT_FDCWD
    refa a1, 0x120
    li a2, 0x241
    li a3, 0600
    sys 56
    li s8, 12
    bltz a0, fail
    mv s5, a0
    mv a0, s5
    refa a1, 0xeff0
    li a2, 32
    sys 64
    expect 13, 16
    /* C14: getresuid through three pointer positions (put_user). */
    refa a0, 0x200
    refa a1, 0x204
    refa a2, 0x208
    sys 148
    expect 14, 0
    refa t1, 0x200
    lw a0, 0(t1)
    expect 15, 0
    refa t1, 0x208
    lw a0, 0(t1)
    expect 16, 0
    /* C17: ioctl FIONBIO reads an int (get_user); an unknown command is refused. */
    mv a0, s4
    li a1, 0x5421
    refa a2, 0x210
    sys 29
    expect 17, 0
    mv a0, s4
    li a1, 0x1234
    li a2, 0
    sys 29
    expect 18, -25
    /* C19: openat of /dev/null by a path at the start of A. */
    li a0, AT_FDCWD
    refa a1, 0x100
    li a2, 1
    li a3, 0
    sys 56
    li s8, 19
    bltz a0, fail

    /* C21: writev of two entries whose bases are references (the compat-style loader). */
    iova 0x600, 0x000, 4
    iova 0x620, 0x004, 8
    mv a0, s5
    refa a1, 0x600
    li a2, 2
    sys 66
    expect 21, 12
    /* C22: readv into B through an iovec that lies in A: the embedded reference, not the
     * outer one, reaches B. */
    refb t2, 0x100
    refa t3, 0x640
    STC(t2, 0, t3)
    refa t3, 0x650
    li t2, 16
    sd t2, 0(t3)
    mv a0, s4
    refa a1, 0x640
    li a2, 1
    sys 65
    expect 22, 16
    refb t1, 0x100
    ld a0, 0(t1)
    expect 23, 0
    refb t1, 0x108
    ld a0, 0(t1)
    expect 24, 0
    /* C25: an untagged base in an iovec names no reference. */
    refa t3, 0x660
    li t2, 0x700
    add t2, s0, t2
    sd t2, 0(t3)
    refa t3, 0x670
    li t2, 8
    sd t2, 0(t3)
    mv a0, s4
    refa a1, 0x660
    li a2, 1
    sys 65
    expect 25, -14
    /* C26: an iovec entry that runs into the hole reads the bytes before it. */
    iova 0x680, 0xeff8, 16
    mv a0, s4
    refa a1, 0x680
    li a2, 1
    sys 65
    expect 26, 8

    /* C20 (rule 1): getrandom of 32 bytes at B + 0xfff0 runs 16 bytes past B's bounds. With
     * B right below A the bytes past the bounds are mapped, so only the processor's check of
     * the S access through t6 stops the copy at 16. */
    li t1, 0x10000
    add t1, s1, t1
    refa a1, 0x440
    li a2, 25
    bne t1, s0, report
    refb a0, 0xfff0
    li a1, 32
    li a2, 0
    sys 278
    refa a1, 0x400
    li a2, 20
    li t0, 16
    beq a0, t0, report
    refa a1, 0x420
    li t0, 32
    beq a0, t0, report
    li s8, 20
    j fail
report:
    li a0, 1
    sys 64
    /* C27: execve with argv references; the shell prints "S5: exec ok" and exits 0. */
    ptra 0x800, 0x900
    ptra 0x810, 0x910
    ptra 0x820, 0x920
    refa a0, 0x8e0
    refa a1, 0x800
    li a2, 0
    sys 221
    li s8, 27
fail:
    mv a0, s8
    sys 93

    .section .rodata
msg_ok:           .ascii "s5 write ok\n"
path_null:        .asciz "/dev/null"
path_file:        .asciz "/s5f"
path_zero:        .asciz "/dev/zero"
ones:             .fill 16, 1, 0xff
msg_enforced:     .ascii "S5: rule1 enforced \n"
msg_missing:      .ascii "S5: rule1 missing  \n"
msg_inconclusive: .ascii "S5: rule1 inconclusive \n\n"
path_sh:          .asciz "/bin/sh"
arg_sh:           .asciz "sh"
arg_c:            .asciz "-c"
arg_echo:         .asciz "echo S5: exec ok"
