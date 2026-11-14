/* Linux exec entry. sp and gp are references supplied by the kernel.
 * The first 16-byte stack cell holds argc, followed by tagged argv cells. */
.section .text.entry,"ax"
.global _start
.type _start, @function
_start:
    movc a0, sp
    call __capstone_linux_start
    ebreak
.size _start, .-_start

/* Compiler-generated pointer globals still need their initializers. These
 * are image relocations, independent of the removed launcher grants. */
.text
.global __capstone_init_globals
.type __capstone_init_globals, @function
__capstone_init_globals:
    cincoffsetimm sp, sp, -32
    stc ra, 0(sp)
    lla t0, __capstone_cap_init_start
    sd t0, 16(sp)
1:  ld t0, 16(sp)
    lla t1, __capstone_cap_init_end
    bgeu t0, t1, 2f
    scc t2, gp, t0
    ld t3, 0(t2)
    add t3, t0, t3
    jalr ra, t3, 0
    ld t0, 16(sp)
    addi t0, t0, 8
    sd t0, 16(sp)
    j 1b
2:  ldc ra, 0(sp)
    cincoffsetimm sp, sp, 32
    ret
.size __capstone_init_globals, .-__capstone_init_globals
.section .gct,"a",@progbits
.balign 16
.global __capstone_gct_start
.hidden __capstone_gct_start
__capstone_gct_start:
