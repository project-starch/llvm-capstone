/* One Linux U-mode task, one tagged register, ordinary mmap and scheduling. */
    .option norvc
    .section .text,"ax"
    .globl _start
_start:
    /* mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS) */
    li a0, 0
    li a1, 4096
    li a2, 3
    li a3, 0x22
    li a4, -1
    li a5, 0
    li a7, 222
    ecall
    bltz a0, fail
    mv s0, a0

    /* Linux binds this one page to the experimental lifetime context and
     * delivers its tagged authority in the saved s2 context slot. */
    li a0, 0x43535410
    mv a1, s0
    li t1, 4096
    add a2, s0, t1
    li a3, 0
    li a4, 0
    li a5, 0
    li a7, 167
    ecall
    bnez a0, fail

#ifdef CAPSTONE_STRIP_S2
    /* Negative control: the same virtual address without the saved tag. */
    mv s2, s0
#endif
#ifdef CAPSTONE_WRONG_FAULT_SITE
    /* Runner control: the correct cause at a different instruction fails. */
    sd zero, 0(zero)
#endif
    li t0, 0x5566
    .globl protected_first_store
protected_first_store:
    sd t0, 0(s2)        /* First touch also exercises a Linux page fault. */
    li a7, 172          /* getpid */
    ecall
    blez a0, fail
    ld t1, 0(s2)
    bne t1, t0, fail

    /* clone(SIGCHLD, NULL, ...) creates a child with a distinct mm.  The
     * parent then blocks in wait4, forcing a switch away and back. */
    li a0, 17
    li a1, 0
    li a2, 0
    li a3, 0
    li a4, 0
    li a7, 220
    ecall
    bltz a0, fail
    beqz a0, child_exit
    li a1, 0
    li a2, 0
    li a3, 0
    li a7, 260
    ecall
    blez a0, fail
    li a0, 0x43535411   /* Count switches away from this mm. */
    li a1, 0
    li a2, 0
    li a3, 0
    li a4, 0
    li a5, 0
    li a7, 167
    ecall
    blez a0, fail
    ld t1, 0(s2)
    bne t1, t0, fail
    li t1, 0x7788
    sd t1, 8(s2)
    ld t2, 8(s2)
    bne t1, t2, fail

    li a0, 0
    j exit
child_exit:
    li a0, 0
    j exit
fail:
    li a0, 1
exit:
    li a7, 93
    ecall
    j .
