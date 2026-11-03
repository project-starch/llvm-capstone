#include "vm-abi.h"
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
/* Table ABI v1: four 32-bit scalar words per record. Slot and target are
 * signed displacements from the record; size zero retains image bounds;
 * addend moves the cursor AFTER narrowing. All values inherit gp's NONLIN
 * type, permissions and lifetime. No DELIN, sealing or new authority.
 * a0 is the table capability, a1 the count. Caller-saved registers only.
 * The symbol makes objects compiled for this ABI fail to link with old SDKs.
 */
.global __capstone_cap_init_table_v1
.hidden __capstone_cap_init_table_v1
.type __capstone_cap_init_table_v1,@function
__capstone_cap_init_table_v1:
    beqz a1, 3f
1:  lw t0, 0(a0)
    lw t1, 4(a0)
    lwu t2, 8(a0)
    lw t3, 12(a0)
    mv t4, a0
    add t0, t4, t0
    add t1, t4, t1
    scc t5, gp, t1
    beqz t2, 2f
    add t2, t1, t2
    shrink t5, t1, t2
2:  cincoffset t5, t5, t3
    scc t6, gp, t0
    stc t5, 0(t6)
    cincoffsetimm a0, a0, 16
    addi a1, a1, -1
    bnez a1, 1b
3:  ret
.size __capstone_cap_init_table_v1, .-__capstone_cap_init_table_v1
.global __capstone_yield
.global __capstone_vm_delegate_transport
.global __capstone_vm_thread_self
__capstone_vm_thread_self:
    li a7, CV_SERVICE_THREAD_SELF
    ecall
    ret
__capstone_vm_delegate_transport:
    li a7, CV_SERVICE_THREAD_DELEGATE
    ecall
    ret
__capstone_yield:
    li a7, CV_SERVICE_DELEGATE
    ecall
    ret
.global __capstone_vm_map
__capstone_vm_map:
    mv a1, a0
    li a2, 3
    li a3, 6
    li a4, 0
    li a7, CV_SERVICE_MAP
    ecall
    bgez a0, 1f
    li a0, 0
1:  ret
.global __capstone_vm_acquire
__capstone_vm_acquire:
    li a7, CV_SERVICE_MAP
    ecall
    ret
.global __capstone_vm_remap
__capstone_vm_remap:
    li a7, CV_SERVICE_REMAP
    ecall
    ret
.global __capstone_vm_unmap
.global __capstone_vm_nodes
__capstone_vm_nodes:
    li a7, CV_SERVICE_NODES
    ecall
    ret

__capstone_vm_unmap:
    li a7, CV_SERVICE_UNMAP
    ecall
    ret
.global __capstone_vm_transport_detach
__capstone_vm_transport_detach:
    li a7, CV_SERVICE_TRANSPORT_DETACH
    ecall
    ret
.global __capstone_vm_thread_create_frame
__capstone_vm_thread_create_frame:
    li a7, CV_SERVICE_THREAD_CREATE
    ecall
    ret
.global __capstone_vm_thread_exit
__capstone_vm_thread_exit:
    li a0, 0
    li a1, 0
    li a2, 0
    li a3, 0
    li a7, CV_SERVICE_THREAD_EXIT
    ecall
    ebreak
.global __capstone_vm_thread_exit_clear
__capstone_vm_thread_exit_clear:
    li a7, CV_SERVICE_PTHREAD_EXIT
    ecall
    ebreak
.global __capstone_vm_thread_join
__capstone_vm_thread_join:
    li a7, CV_SERVICE_THREAD_JOIN
    ecall
    ret
.global __capstone_vm_protect
__capstone_vm_protect:
    li a7, CV_SERVICE_PROTECT
    ecall
    ret
.global __capstone_vm_wait
__capstone_vm_wait:
    li a7, CV_SERVICE_WAIT
    ecall
    ret
.global __capstone_vm_futex
__capstone_vm_futex:
    li a7, CV_SERVICE_FUTEX
    ecall
    ret
.section .gct,"a",@progbits
.balign 16
.global __capstone_gct_start
.hidden __capstone_gct_start
__capstone_gct_start:
.section .capstone_virtual,"",@progbits
.quad CV_IMAGE_MAGIC

/* The same non-executable cancellation markers used by start-musl.S. */
.section .rodata.capstone_cancel_points,"a",@progbits
.global __cp_begin
.global __cp_end
.global __cp_cancel
__cp_begin:
    .byte 0
__cp_end:
    .byte 0
__cp_cancel:
    .byte 0
