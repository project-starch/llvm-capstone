    .insn r 0x5b, 1, 3, s2, zero, zero
    .insn r 0x5b, 1, 3, s4, zero, zero
    .insn r 0x5b, 1, 4, t0, s4, x4
    .insn r 0x5b, 1, 5, sp, s4, t0
#ifdef LINEAR_PARENT
    .insn r 0x5b, 1, 0x0a, a0, s3, zero /* s3 is still linear: moved into a0 */
#else
    .insn r 0x5b, 1, 0x0a, a0, s2, zero
#endif
    jal ra, wrapper_code
#if defined(STALE_ALIAS) || defined(RENEW_STALE)
fault_site:
    lbu t0, 0(a0) /* the child, whose lifetime a release or a renewal ended */
#else
    bnez a0, wrapper_fail
#if defined(BAD_OFFSET) || defined(LINEAR_PARENT) || defined(LEGACY_FORM)
    .set fault_site, wrapper_code+WRAPPER_FAULT_OFFSET
#else
fault_site:
    ebreak
#endif
#endif
wrapper_fail:
    .word 0
.balign 4
wrapper_code:
    .incbin WRAPPER_BINARY
