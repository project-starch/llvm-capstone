/* Virtual application startup: all authority arrives in explicit registers.
 * gp is an image capability with a representable cursor; gp-free codegen
 * uses SCC for globals and ordinary calls within PCC. */
.section .text.entry,"ax"
.global _start
_start:
    cincoffsetimm sp, sp, -80
    stc a0, 0(sp)
    stc a1, 16(sp)
    stc a2, 32(sp)
    stc a3, 48(sp)
    lla t0, __capstone_cap_init_start
    sd t0, 64(sp)
1:  ld t0, 64(sp)
    lla t1, __capstone_cap_init_end
    bgeu t0, t1, 2f
    scc t2, gp, t0
    ld t3, 0(t2)
    add t3, t0, t3
    jalr ra, t3, 0
    ld t0, 64(sp)
    addi t0, t0, 8
    sd t0, 64(sp)
    j 1b
2:  ldc a0, 0(sp)
    li a1, 1
    call domain_main
    ldc a0, 16(sp)
    li a1, 1
    call domain_main
    ldc a0, 32(sp)
    li a1, 1
    call domain_main
    ld t0, 48(sp)
    beqz t0, 3f
    ldc a0, 48(sp)
    li a1, 1
    call domain_main
3:  li a0, 0
    li a1, 0
    call domain_main
    ebreak
.text
.global __capstone_yield
__capstone_yield:
    li a7, 0
    ecall
    ret
.global __capstone_vm_map
__capstone_vm_map:
    li a7, 1
    ecall
    ret
.global __capstone_vm_unmap
__capstone_vm_unmap:
    li a7, 2
    ecall
    ret
.global __capstone_vm_thread_create_frame
__capstone_vm_thread_create_frame:
    li a7, 3
    ecall
    ret
.global __capstone_vm_thread_exit
__capstone_vm_thread_exit:
    li a7, 4
    ecall
    ebreak
.section .gct
.balign 16
.global __capstone_gct_start
.hidden __capstone_gct_start
__capstone_gct_start:
.section .capstone_virtual,"",@progbits
.quad 0x314d56564e4f5043
