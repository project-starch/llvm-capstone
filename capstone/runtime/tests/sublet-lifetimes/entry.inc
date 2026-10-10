    .insn r 0x5b, 1, 3, s2, zero, zero
    .insn r 0x5b, 1, 3, s4, zero, zero
    .insn r 0x5b, 1, 4, t0, s4, x4
    .insn r 0x5b, 1, 5, sp, s4, t0
    .insn r 0x5b, 1, 0x0a, a0, s2, zero
    jal ra, wrapper_code
    bnez a0, wrapper_fail
#ifndef BAD_OFFSET
fault_site:
    ebreak
#else
    .set fault_site, wrapper_code+WRAPPER_FAULT_OFFSET
#endif
wrapper_fail:
    .word 0
.balign 4
wrapper_code:
    .incbin WRAPPER_BINARY
